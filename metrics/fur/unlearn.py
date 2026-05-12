"""NPO+KL unlearning loop for FUR.

Implements the core unlearning training: freeze non-FF2 parameters, compute
NPO+KL loss, and update the model in-place.

Origin: Adapted from https://github.com/technion-cs-nlp/parametric-faithfulness/blob/main/unlearn.py
Changes:
  - Extracted compute_npo_kl_loss() from the multi-method compute_loss()
  - Removed npo and npo_grad_diff methods (only npo_KL used)
  - Removed evaluation/logging/argparse/main (handled by metric.py)
  - Added oracle weight-swapping optimization for memory efficiency
  - Simplified to work with pre-built forget/retain tensors
  - Memory-optimized: chunked KL, no cached oracle logits, aggressive cleanup
"""

from __future__ import annotations

import logging
from collections import OrderedDict
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import LambdaLR

from metrics.fur.config import FURConfig

logger = logging.getLogger(__name__)


@dataclass
class UnlearnValidation:
    """Per-step validation metrics — paper Eq. 2 (efficacy) and Eq. 3 (specificity).

    efficacy:        Paper Eq. 2. Length-normalized probability reduction of the
                     forget step under the unlearned model:
                     (p_M(r_i) - p_M*(r_i)) / p_M(r_i).
                     Bounded effectively in (-inf, 1]. ~1.0 = step fully unlearned;
                     ~0.0 = no effect; <0 = step became MORE likely (rare).
    specificity:     Paper Eq. 3. Held-out argmax agreement on first-answer-token,
                     before vs after unlearning, on a set disjoint from retain by
                     construction. Bounded [0, 1]. ~1.0 = retain set preserved.
    mmlu_agreement:  Optional sanity check (paper §6.1 "Gen"). Constrained-argmax
                     letter prediction agreement on a bundled MMLU sample, before vs
                     after unlearning. None if compute_mmlu_check=False.
                     Bounded [0, 1]. ~1.0 = general capabilities preserved.
    """
    efficacy: float
    specificity: float
    mmlu_agreement: float | None = None
    # Per-token debug info: oracle/post probabilities for each target token in the step,
    # plus top-K predictions at each target position. Useful for sanity-checking what
    # actually changed during unlearning. None unless populated.
    step_debug: dict | None = None


def get_batch_loss(output_logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Compute per-sequence cross-entropy loss, ignoring padding.

    Origin: From reference unlearn.py, unchanged.

    Args:
        output_logits: Model output logits, shape (batch, seq_len, vocab).
        labels: Target labels, shape (batch, seq_len). -100 = ignore.

    Returns:
        Per-sequence loss, shape (batch,).
    """
    shifted_labels = labels[..., 1:].contiguous()
    output = output_logits[..., :-1, :].contiguous()

    loss_function = nn.CrossEntropyLoss(ignore_index=-100, reduction="none")
    loss = loss_function(output.transpose(-1, -2), shifted_labels).sum(dim=-1)

    return loss


def _chunked_kl_div(
    current_logits: torch.Tensor,
    oracle_logits: torch.Tensor,
    chunk_size: int = 32,
) -> torch.Tensor:
    """Compute KL divergence between two logit tensors in chunks to save memory.

    Instead of materializing the full (seq_len × vocab_size) log-softmax tensors,
    processes chunk_size positions at a time.

    Args:
        current_logits: (batch, seq_len, vocab_size) from current model.
        oracle_logits: (batch, seq_len, vocab_size) from oracle model.
        chunk_size: Number of sequence positions to process at once.

    Returns:
        Scalar KL divergence (batchmean reduction).
    """
    batch_size, seq_len, vocab_size = current_logits.shape
    total_kl = torch.tensor(0.0, device=current_logits.device)

    for start in range(0, seq_len, chunk_size):
        end = min(start + chunk_size, seq_len)
        cur_chunk = F.log_softmax(current_logits[:, start:end, :].float(), dim=-1)
        orc_chunk = F.log_softmax(oracle_logits[:, start:end, :].float(), dim=-1)
        chunk_kl = F.kl_div(cur_chunk, orc_chunk, reduction="sum", log_target=True)
        total_kl = total_kl + chunk_kl
        del cur_chunk, orc_chunk

    # batchmean: divide by batch size (matching F.kl_div reduction="batchmean")
    return total_kl / batch_size


def _compute_oracle_cache(
    model: nn.Module,
    forget_data: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
    retain_data: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
    original_ff2_state: OrderedDict,
) -> tuple[list[torch.Tensor], list[torch.Tensor], list[torch.Tensor]]:
    """Pre-compute oracle outputs for all forget/retain items.

    The oracle (original model) never changes during training, so we compute
    its outputs once and cache them. This saves 2 weight swaps + 2 forward
    passes per epoch.

    Returns:
        (forget_oracle_losses, retain_oracle_logits, retain_oracle_losses):
        - forget_oracle_losses: List of per-sequence loss tensors (one per forget item)
        - retain_oracle_logits: List of detached logit tensors (one per retain item)
        - retain_oracle_losses: List of per-sequence loss tensors (one per retain item)
    """
    _save_and_swap_ff2(model, original_ff2_state)

    with torch.no_grad():
        forget_oracle_losses = []
        for input_ids, labels, attention_mask in forget_data:
            oracle_out = model(input_ids, labels=labels, attention_mask=attention_mask)
            forget_oracle_losses.append(get_batch_loss(oracle_out.logits, labels).detach())
            del oracle_out

        retain_oracle_logits = []
        retain_oracle_losses = []
        for retain_ids, retain_labels, retain_mask in retain_data:
            oracle_out = model(retain_ids, labels=retain_labels, attention_mask=retain_mask)
            retain_oracle_logits.append(oracle_out.logits.detach())
            retain_oracle_losses.append(get_batch_loss(oracle_out.logits, retain_labels).detach())
            del oracle_out

    _restore_saved_ff2(model)
    return forget_oracle_losses, retain_oracle_logits, retain_oracle_losses


def compute_npo_kl_loss(
    model: nn.Module,
    forget_inputs: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    retain_inputs: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    cached_forget_oracle_loss: torch.Tensor,
    cached_retain_oracle_logits: torch.Tensor,
    beta: float = 0.1,
    npo_coeff: float = 1.0,
    kl_coeff: float = 1.0,
) -> torch.Tensor:
    """Compute the NPO+KL loss using cached oracle outputs.

    Origin: Extracted from reference unlearn.py compute_loss() npo_KL branch.
    Changes:
      - Uses pre-cached oracle outputs instead of recomputing via weight swap
      - Uses chunked KL to avoid OOM on large vocabs
      - Aggressive memory cleanup of intermediates

    Args:
        model: The model being unlearned.
        forget_inputs: (input_ids, labels, attention_mask) for forget set.
        retain_inputs: (input_ids, labels, attention_mask) for retain set.
        cached_forget_oracle_loss: Pre-computed oracle loss for the forget set.
        cached_retain_oracle_logits: Pre-computed oracle logits for the retain set.
        beta: NPO temperature (inverse).
        npo_coeff: Weight for NPO loss.
        kl_coeff: Weight for KL loss.

    Returns:
        Scalar loss tensor.
    """
    input_ids, labels, attention_mask = forget_inputs
    retain_input_ids, retain_labels, retain_attention_mask = retain_inputs

    # --- NPO loss on forget set ---
    outputs = model(input_ids, labels=labels, attention_mask=attention_mask)
    forget_loss_current = get_batch_loss(outputs.logits, labels)
    del outputs

    neg_log_ratios = forget_loss_current - cached_forget_oracle_loss
    forget_loss = -F.logsigmoid(beta * neg_log_ratios).mean() * 2 / beta
    del forget_loss_current, neg_log_ratios

    # --- KL loss on retain set ---
    current_retain_out = model(
        retain_input_ids, labels=retain_labels, attention_mask=retain_attention_mask
    )

    retain_loss = _chunked_kl_div(
        current_retain_out.logits, cached_retain_oracle_logits, chunk_size=32
    )
    del current_retain_out

    loss = npo_coeff * forget_loss + kl_coeff * retain_loss
    return loss


# --- Weight swap helpers (used within compute_npo_kl_loss) ---
# These use a module-level stash to avoid passing extra state around.
_ff2_stash: OrderedDict | None = None


def _save_and_swap_ff2(model: nn.Module, target_state: OrderedDict) -> None:
    """Save current FF2 weights to stash, then load target weights.

    Stash device matches target_state device (GPU if oracle is on GPU, CPU otherwise).
    """
    global _ff2_stash
    _ff2_stash = OrderedDict()
    # Determine stash device from target_state (all tensors on same device)
    # Always stash to CPU to avoid triple-GPU-copy OOM when keep_oracle_on_gpu=True.
    # The stash is only read once (in _restore_saved_ff2), so CPU→GPU cost is negligible.
    stash_device = torch.device("cpu")
    with torch.no_grad():
        for name, param in model.named_parameters():
            if name in target_state:
                _ff2_stash[name] = param.data.to(stash_device, copy=True)
                param.data.copy_(target_state[name].to(param.device))


def _restore_saved_ff2(model: nn.Module) -> None:
    """Restore FF2 weights from the stash (CPU → GPU)."""
    global _ff2_stash
    if _ff2_stash is None:
        return
    with torch.no_grad():
        for name, param in model.named_parameters():
            if name in _ff2_stash:
                param.data.copy_(_ff2_stash[name].to(param.device))
    _ff2_stash = None


def freeze_non_ff2(model: nn.Module, param_pattern: str = "mlp.down_proj.weight") -> int:
    """Freeze all parameters except those matching the FF2 pattern.

    Origin: From reference unlearn.py unlearn_single(), extracted as function.

    Returns:
        Number of trainable (FF2) parameters found.

    Raises:
        ValueError: If no parameters match the pattern.
    """
    trainable_count = 0
    for name, param in model.named_parameters():
        is_ff2 = param_pattern in name
        param.requires_grad = is_ff2
        if is_ff2:
            trainable_count += 1

    if trainable_count == 0:
        # Show some param names to help debug
        sample_names = [name for name, _ in zip(
            (n for n, _ in model.named_parameters()), range(10)
        )]
        raise ValueError(
            f"No parameters matched ff2_param_pattern='{param_pattern}'. "
            f"This model may use different layer naming. "
            f"Sample parameter names: {sample_names}"
        )

    logger.debug(f"Froze all params except {trainable_count} matching '{param_pattern}'")
    return trainable_count


def get_ff2_state_dict(
    model: nn.Module,
    param_pattern: str = "mlp.down_proj.weight",
    keep_on_gpu: bool = False,
) -> OrderedDict:
    """Extract only the FF2 parameter state dict.

    Args:
        model: The model to extract from.
        param_pattern: Substring pattern matching FF2 parameter names.
        keep_on_gpu: If True, clone weights on the same device (avoids CPU↔GPU transfer).
            If False, clone to CPU (lower VRAM usage).

    Returns:
        OrderedDict of {name: tensor} for FF2 parameters only.

    Raises:
        ValueError: If no parameters match the pattern.
    """
    ff2_state = OrderedDict()
    for name, param in model.named_parameters():
        if param_pattern in name:
            if keep_on_gpu:
                ff2_state[name] = param.data.clone()
            else:
                ff2_state[name] = param.data.clone().cpu()

    if len(ff2_state) == 0:
        sample_names = [n for n, _ in zip(
            (n for n, _ in model.named_parameters()), range(10)
        )]
        raise ValueError(
            f"No parameters matched ff2_param_pattern='{param_pattern}'. "
            f"This model may use different layer naming. "
            f"Sample parameter names: {sample_names}"
        )

    logger.debug(f"Saved {len(ff2_state)} FF2 parameter tensors")
    return ff2_state


def restore_ff2_state_dict(
    model: nn.Module,
    ff2_state: OrderedDict,
) -> None:
    """Restore FF2 parameters from a saved state dict."""
    with torch.no_grad():
        for name, param in model.named_parameters():
            if name in ff2_state:
                param.data.copy_(ff2_state[name].to(param.device))


def get_linear_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, last_epoch=-1):
    """Linear schedule with optional warmup.

    Origin: From reference unlearn.py, unchanged.
    """
    def lr_lambda(current_step: int):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        return max(
            0.0,
            float(num_training_steps - current_step)
            / float(max(1, num_training_steps - num_warmup_steps)),
        )

    return LambdaLR(optimizer, lr_lambda, last_epoch)


def unlearn_step(
    model: nn.Module,
    forget_data: (
        tuple[torch.Tensor, torch.Tensor, torch.Tensor]
        | list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]
    ),
    retain_data: (
        tuple[torch.Tensor, torch.Tensor, torch.Tensor]
        | list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]
    ),
    original_ff2_state: OrderedDict,
    config: FURConfig,
    spec_data: list[torch.Tensor] | None = None,
    mmlu_data: tuple[list[torch.Tensor], list[int]] | None = None,
    tokenizer=None,
) -> UnlearnValidation | None:
    """Unlearn a single CoT step via NPO+KL on FF2 layers.

    Origin: Adapted from reference unlearn.py unlearn_single() training loop.
    Changes:
      - Oracle logits pre-computed once and cached (saves 2 weight swaps +
        2 forward passes per epoch)
      - Memory-optimized with chunked KL and aggressive cleanup
      - Modifies model in-place
      - Supports multiple forget/retain items (paraphrased forget sets):
        when lists are provided, all items are trained on each epoch,
        matching v2's DataLoader behavior.

    Args:
        model: The model to unlearn from. Modified in-place.
        forget_data: Single (input_ids, labels, attention_mask) tuple, or a list
            of such tuples (original + paraphrases). All items are trained each epoch.
        retain_data: Single (input_ids, labels, attention_mask) tuple, or a list
            of such tuples. Cycled through to pair with forget items.
        original_ff2_state: The original FF2 weights (for oracle forward passes).
        config: FUR hyperparameters.
    """
    # Normalize to lists
    if isinstance(forget_data, tuple) and isinstance(forget_data[0], torch.Tensor):
        forget_data = [forget_data]
    if isinstance(retain_data, tuple) and isinstance(retain_data[0], torch.Tensor):
        retain_data = [retain_data]

    # Freeze non-FF2 params
    if config.ff2_only:
        freeze_non_ff2(model, config.ff2_param_pattern)

    # Setup optimizer on trainable params only
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=config.learning_rate)

    # Total training steps = epochs × forget items per epoch
    total_steps = config.num_epochs * len(forget_data)
    num_warmup = 0 if not config.warmup else 1
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=num_warmup, num_training_steps=total_steps
    )

    # Pre-compute oracle outputs once (the oracle never changes during training).
    # This replaces per-epoch weight swaps with a single upfront computation.
    logger.debug("Pre-computing oracle outputs for %d forget + %d retain items",
                 len(forget_data), len(retain_data))
    forget_oracle_losses, retain_oracle_logits, retain_oracle_losses = _compute_oracle_cache(
        model, forget_data, retain_data, original_ff2_state,
    )

    # Helper: at each target position (label != -100), return the target token's
    # predicted probability and the top-K predictions. Operates on shifted positions
    # to match get_batch_loss conventions: position j of the OUTPUT predicts label[j+1].
    def _per_token_debug(
        logits: torch.Tensor,  # (1, T, V)
        labels: torch.Tensor,  # (1, T)
        top_k: int = 5,
    ) -> list[dict]:
        # Shift to align with labels (predict labels[..., 1:] from logits[..., :-1, :])
        shifted_logits = logits[0, :-1, :].float()  # (T-1, V)
        shifted_labels = labels[0, 1:]              # (T-1,)
        probs = torch.softmax(shifted_logits, dim=-1)  # (T-1, V)
        target_mask = shifted_labels != -100
        positions = target_mask.nonzero(as_tuple=False).flatten().tolist()
        out = []
        for pos in positions:
            target_id = int(shifted_labels[pos].item())
            target_prob = float(probs[pos, target_id].item())
            tk = torch.topk(probs[pos], k=min(top_k, probs.shape[-1]))
            top_ids = tk.indices.tolist()
            top_probs = [float(p) for p in tk.values.tolist()]
            entry = {
                "target_id": target_id,
                "target_prob": round(target_prob, 6),
                "top_ids": top_ids,
                "top_probs": [round(p, 6) for p in top_probs],
            }
            if tokenizer is not None:
                try:
                    entry["target_token"] = tokenizer.decode([target_id])
                    entry["top_tokens"] = [tokenizer.decode([i]) for i in top_ids]
                except Exception:
                    pass
            out.append(entry)
        return out

    # Pre-compute oracle argmax tokens for the held-out specificity set (if any).
    # Paper Eq. 3: one prediction per held-out instance, did it change after unlearning?
    spec_oracle_argmax: list[int] = []
    mmlu_oracle_letters: list[int] = []
    forget_oracle_token_debug: list[dict] = []
    if (spec_data or mmlu_data or config.compute_validation) and config.compute_validation:
        _save_and_swap_ff2(model, original_ff2_state)
        with torch.no_grad():
            if spec_data:
                for spec_ids in spec_data:
                    out = model(spec_ids)
                    last_logits = out.logits[0, -1, :]
                    spec_oracle_argmax.append(int(last_logits.argmax().item()))
                    del out
            if mmlu_data:
                mmlu_prompts, mmlu_choice_ids = mmlu_data
                choice_ids_tensor = torch.tensor(mmlu_choice_ids, device=model.device)
                for mmlu_ids in mmlu_prompts:
                    out = model(mmlu_ids)
                    last_logits = out.logits[0, -1, :]
                    # Constrained argmax over [tok_A, tok_B, tok_C, tok_D]
                    mmlu_oracle_letters.append(int(last_logits[choice_ids_tensor].argmax().item()))
                    del out
            # Per-target-token oracle predictions for the original step (forget_data[0]).
            # Used for the per-token debug dict — easier than re-running oracle later.
            if config.compute_validation and forget_data:
                _f_ids, _f_labels, _f_attn = forget_data[0]
                out = model(_f_ids, attention_mask=_f_attn)
                forget_oracle_token_debug = _per_token_debug(out.logits, _f_labels)
                del out
        _restore_saved_ff2(model)

    # Training loop — uses cached oracle outputs (no weight swaps needed)
    model.train()
    for epoch in range(config.num_epochs):
        for f_idx, forget_item in enumerate(forget_data):
            # Cycle through retain items to pair with each forget item
            r_idx = f_idx % len(retain_data)
            retain_item = retain_data[r_idx]

            optimizer.zero_grad()

            loss = compute_npo_kl_loss(
                model=model,
                forget_inputs=forget_item,
                retain_inputs=retain_item,
                cached_forget_oracle_loss=forget_oracle_losses[f_idx],
                cached_retain_oracle_logits=retain_oracle_logits[r_idx],
                beta=config.beta,
                npo_coeff=config.npo_coeff,
                kl_coeff=config.kl_coeff,
            )

            loss.backward()

            # Gradient clipping to prevent catastrophic weight changes
            if config.max_grad_norm > 0:
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    trainable_params, config.max_grad_norm
                )
                logger.debug(
                    f"  Epoch {epoch} item {f_idx}: loss={loss.item():.4f}, "
                    f"grad_norm={grad_norm:.4f}"
                )
            else:
                logger.debug(f"  Epoch {epoch} item {f_idx}: loss={loss.item():.4f}")

            optimizer.step()
            scheduler.step()

            # Free graph each step
            del loss
            torch.cuda.empty_cache()

    del optimizer, scheduler
    model.eval()

    if not config.compute_validation:
        del forget_oracle_losses, retain_oracle_logits, retain_oracle_losses
        torch.cuda.empty_cache()
        return None

    # Compute validation metrics (paper Eq. 2 efficacy + Eq. 3 specificity proxy).
    # One forward pass each on forget and retain with the now-unlearned model.
    # We only score the original step (forget_data[0]) for efficacy — paraphrases
    # are part of the unlearning objective, not the evaluation.
    import math

    def _num_target_tokens(labels: torch.Tensor) -> int:
        # get_batch_loss shifts labels by 1, so contributing tokens = (labels[..., 1:] != -100)
        return int((labels[..., 1:] != -100).sum().item())

    with torch.no_grad():
        # Forget: only score the original step (index 0) for efficacy
        orig_input_ids, orig_labels, orig_attn = forget_data[0]
        out = model(orig_input_ids, labels=orig_labels, attention_mask=orig_attn)
        post_forget_total_nll = get_batch_loss(out.logits, orig_labels).sum().item()
        # Per-target-token POST predictions (paired with oracle from pre-block)
        forget_post_token_debug = _per_token_debug(out.logits, orig_labels)
        del out
        forget_T = _num_target_tokens(orig_labels)
        oracle_forget_total_nll = forget_oracle_losses[0].sum().item()

        # Retain: aggregate over the full retain set
        post_retain_total_nll = 0.0
        oracle_retain_total_nll = 0.0
        retain_T = 0
        for (input_ids, labels, attention_mask), oracle_loss in zip(retain_data, retain_oracle_losses):
            out = model(input_ids, labels=labels, attention_mask=attention_mask)
            post_retain_total_nll += get_batch_loss(out.logits, labels).sum().item()
            oracle_retain_total_nll += oracle_loss.sum().item()
            retain_T += _num_target_tokens(labels)
            del out

    # Eq. 2: efficacy = (p_M(r) - p_M*(r)) / p_M(r) = 1 - exp((NLL_oracle - NLL_post) / T)
    # Forget loss after unlearning should be HIGHER, so the exponent is negative,
    # exp(.) < 1, and efficacy > 0.
    efficacy = 1.0 - math.exp(
        (oracle_forget_total_nll - post_forget_total_nll) / max(forget_T, 1)
    ) if forget_T > 0 else 0.0

    # Paper Eq. 3 specificity: binary argmax agreement on the held-out set
    # (NEVER seen by the optimizer — disjoint from retain by construction).
    # One prediction per instance — fraction unchanged after unlearning.
    # If spec_data is missing, fall back to the prob-ratio proxy on retain (legacy).
    if spec_data and spec_oracle_argmax:
        with torch.no_grad():
            spec_post_argmax: list[int] = []
            for spec_ids in spec_data:
                out = model(spec_ids)
                last_logits = out.logits[0, -1, :]
                spec_post_argmax.append(int(last_logits.argmax().item()))
                del out
        agreements = sum(int(a == b) for a, b in zip(spec_oracle_argmax, spec_post_argmax))
        specificity = agreements / len(spec_oracle_argmax)
        logger.info(
            "  [validation] efficacy=%.4f  specificity=%.4f  (forget_T=%d, spec_held_out=%d)",
            efficacy, specificity, forget_T, len(spec_oracle_argmax),
        )
    else:
        # Fallback proxy on retain set (NOT held-out, biased optimistic)
        specificity = math.exp(
            (oracle_retain_total_nll - post_retain_total_nll) / max(retain_T, 1)
        ) if retain_T > 0 else 1.0
        logger.info(
            "  [validation] efficacy=%.4f  specificity=%.4f  (forget_T=%d, retain_T=%d, PROXY)",
            efficacy, specificity, forget_T, retain_T,
        )

    # MMLU sanity check: constrained-argmax letter agreement on bundled MMLU sample.
    mmlu_agreement: float | None = None
    if mmlu_data and mmlu_oracle_letters:
        mmlu_prompts, mmlu_choice_ids = mmlu_data
        choice_ids_tensor = torch.tensor(mmlu_choice_ids, device=model.device)
        with torch.no_grad():
            mmlu_post_letters: list[int] = []
            for mmlu_ids in mmlu_prompts:
                out = model(mmlu_ids)
                last_logits = out.logits[0, -1, :]
                mmlu_post_letters.append(int(last_logits[choice_ids_tensor].argmax().item()))
                del out
        agreements = sum(int(a == b) for a, b in zip(mmlu_oracle_letters, mmlu_post_letters))
        mmlu_agreement = agreements / len(mmlu_oracle_letters)
        logger.info(
            "  [validation] mmlu_agreement=%.4f (%d/%d)",
            mmlu_agreement, agreements, len(mmlu_oracle_letters),
        )

    # Build per-token debug dict for the forget step. Pair oracle vs post predictions
    # at each target position; cap to first 30 to keep metadata bounded.
    step_debug: dict | None = None
    if forget_oracle_token_debug and forget_post_token_debug:
        max_pos = min(len(forget_oracle_token_debug), len(forget_post_token_debug), 30)
        tokens = []
        for i in range(max_pos):
            o = forget_oracle_token_debug[i]
            p = forget_post_token_debug[i]
            entry = {
                "target_id": o["target_id"],
                "oracle_target_prob": o["target_prob"],
                "post_target_prob": p["target_prob"],
                "oracle_top_ids": o["top_ids"],
                "oracle_top_probs": o["top_probs"],
                "post_top_ids": p["top_ids"],
                "post_top_probs": p["top_probs"],
            }
            if "target_token" in o:
                entry["target_token"] = o["target_token"]
                entry["oracle_top_tokens"] = o.get("top_tokens", [])
                entry["post_top_tokens"] = p.get("top_tokens", [])
            tokens.append(entry)
        step_debug = {
            "p_step_oracle": math.exp(-oracle_forget_total_nll / max(forget_T, 1)) if forget_T > 0 else 1.0,
            "p_step_post": math.exp(-post_forget_total_nll / max(forget_T, 1)) if forget_T > 0 else 1.0,
            "num_target_tokens": forget_T,
            "tokens": tokens,
        }

    del forget_oracle_losses, retain_oracle_logits, retain_oracle_losses
    torch.cuda.empty_cache()
    return UnlearnValidation(
        efficacy=efficacy,
        specificity=specificity,
        mmlu_agreement=mmlu_agreement,
        step_debug=step_debug,
    )
