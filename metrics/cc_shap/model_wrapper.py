"""Model wrappers bridging the SHAP library with chat-template forward passes.

Provides two classes:
- FixedTargetGenerator: supplies pre-known target text (skips generation).
- QuestionTeacherForcing: reconstructs full chat-template prompts from masked
  question text and computes teacher-forced log-odds for the SHAP explainer.
"""

from __future__ import annotations

import logging
from typing import Callable

import numpy as np
from numpy.typing import NDArray

logger = logging.getLogger(__name__)


class FixedTargetGenerator:
    """Returns pre-known target text instead of generating.

    Used by OutputComposite to supply the target for teacher forcing.
    The paper's flow generates the target from unmasked input first;
    since we already have the target (answer/CoT from data), we skip generation.
    """

    def __init__(self, target_text: str) -> None:
        self.target_text = target_text
        self.explanation_row = 0  # OutputComposite expects this

    def __call__(self, X: NDArray) -> NDArray[np.str_]:
        """Return the fixed target regardless of input."""
        self.explanation_row += 1
        return np.array([self.target_text])


class QuestionTeacherForcing:
    """Teacher-forcing model wrapper that reconstructs chat-template prompts.

    The SHAP library's Text masker operates on question text only (our adaptation).
    This wrapper:
    1. Receives masked question strings from the Text masker
    2. Reconstructs the full chat-template prompt via build_prompt()
    3. Teacher-forces the target (answer or CoT)
    4. Returns log-odds matching the paper's TeacherForcing.get_logodds()

    Args:
        model: HuggingFace CausalLM model.
        tokenizer: HuggingFace tokenizer.
        prompt_builder: Callable that takes question text and returns full prompt string.
        batch_size: Max batch size for forward passes.
        auto_batch: Auto-shrink batch size based on free GPU memory (capped by batch_size).
    """

    def __init__(
        self,
        model: object,
        tokenizer: object,
        prompt_builder: Callable[[str], str],
        batch_size: int = 8,
        auto_batch: bool = True,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.prompt_builder = prompt_builder
        self.batch_size = batch_size
        self.auto_batch = auto_batch
        # Required attributes for SHAP library compatibility
        self.output_names: list[str] | None = None
        self.output: NDArray | None = None

    def update_output_names(self, output: NDArray) -> None:
        """Update output token names (for SHAP Explanation labels)."""
        if (self.output is None) or (not np.array_equal(self.output, output)):
            self.output = output
            # Tokenize the target to get output token names
            if output.dtype.type is np.str_:
                output_ids = self.tokenizer(
                    output[0], add_special_tokens=False
                ).input_ids
                self.output_names = [
                    self.tokenizer.decode([x]).strip() for x in output_ids
                ]
            else:
                self.output_names = [
                    self.tokenizer.decode([x]).strip() for x in output[0]
                ]
            logger.debug(
                "Updated output names: %d tokens, first few: %s",
                len(self.output_names),
                self.output_names[:5],
            )

    def __call__(self, X: NDArray, Y: NDArray) -> NDArray[np.float64]:
        """Compute log-odds for masked question inputs with known targets.

        Args:
            X: np.ndarray of masked question strings, shape (B,)
            Y: np.ndarray of target strings/ids, shape (B,)

        Returns:
            np.ndarray of log-odds, shape (B, T) where T = number of target tokens
        """
        self.update_output_names(Y[:1])

        # Get target token IDs
        if Y.dtype.type is np.str_:
            target_ids = self.tokenizer(
                Y[0].strip(), add_special_tokens=False
            ).input_ids
        else:
            target_ids = Y[0].tolist()

        logger.debug(
            "Teacher forcing: %d masked inputs, %d target tokens, batch_size=%d",
            len(X),
            len(target_ids),
            self.batch_size,
        )

        T = len(target_ids)
        effective_batch_size = self.batch_size

        # Auto-batch: measure free GPU memory and pick the largest batch that fits.
        import torch
        auto_on = self.auto_batch and torch.cuda.is_available()
        try:
            device = next(self.model.parameters()).device
        except StopIteration:
            device = torch.device("cpu")
        if auto_on and device.type != "cuda":
            auto_on = False

        if auto_on:
            # Estimate per-sample GPU memory: logits tensor (B, prompt_len+T, vocab) in
            # float32 after .float() conversion in _teacher_forced_logits, plus activations.
            # Use the longest prompt in this call as the per-sample sequence length proxy.
            sample_prompts = [self.prompt_builder(q) for q in X[: min(len(X), 4)].tolist()]
            max_prompt_tokens = max(
                (len(self.tokenizer(p, add_special_tokens=False).input_ids)
                 for p in sample_prompts),
                default=0,
            )
            vocab_size = int(getattr(self.tokenizer, "vocab_size", 0)) or 32000
            # Logits tensor (B, seq_len, vocab) in float32 dominates; flash SDP keeps
            # attention ~linear in T. 2.0× covers activation peaks; measured usage at
            # T=22k, V=151k was ~18GB/sample vs 13GB logits → ~1.4× real overhead.
            SAFETY = 2.0
            seq_len = max_prompt_tokens + T
            mem_per_sample = seq_len * vocab_size * 4 * SAFETY  # bytes
            free_bytes, _total = torch.cuda.mem_get_info(device)
            budget = free_bytes * 0.75
            auto_bs = max(1, int(budget // max(mem_per_sample, 1)))
            chosen = min(effective_batch_size, auto_bs)
            if chosen < effective_batch_size:
                logger.info(
                    "CC-SHAP AUTO-BATCH ON: free_gpu=%.2fGB, target_T=%d, seq_len=%d, "
                    "mem/sample=%.0fMB — tight, shrinking batch %d → %d (cap=%d)",
                    free_bytes / 1e9, T, seq_len, mem_per_sample / 1e6,
                    effective_batch_size, chosen, self.batch_size,
                )
            else:
                logger.info(
                    "CC-SHAP AUTO-BATCH ON: free_gpu=%.2fGB, target_T=%d, seq_len=%d, "
                    "mem/sample=%.0fMB → batch_size=%d (cap=%d)",
                    free_bytes / 1e9, T, seq_len, mem_per_sample / 1e6,
                    chosen, self.batch_size,
                )
            effective_batch_size = chosen
        else:
            reason = "disabled via config" if not self.auto_batch else "no CUDA device"
            logger.info(
                "CC-SHAP AUTO-BATCH OFF (%s): batch_size=%d (cap=%d, target_T=%d)",
                reason, effective_batch_size, self.batch_size, T,
            )

        output_batch: NDArray[np.float64] | None = None
        for start in range(0, len(X), effective_batch_size):
            end = min(start + effective_batch_size, len(X))
            X_batch = X[start:end]

            logger.debug(
                "Processing batch [%d:%d] of %d inputs", start, end, len(X)
            )

            logits = self._teacher_forced_logits(X_batch, target_ids)
            logodds = self._logits_to_logodds(logits, target_ids)

            if output_batch is None:
                output_batch = logodds
            else:
                output_batch = np.concatenate((output_batch, logodds))

        return output_batch  # type: ignore[return-value]

    def _teacher_forced_logits(
        self, X_batch: NDArray, target_ids: list[int]
    ) -> NDArray[np.float64]:
        """Forward pass: build full prompts, concat targets, get logits.

        Args:
            X_batch: Masked question strings, shape (B,).
            target_ids: Token IDs of the target sequence.

        Returns:
            Logits array of shape (B, T, vocab_size).
        """
        import torch

        device = next(self.model.parameters()).device

        # Build full prompts from masked questions
        full_prompts = [self.prompt_builder(q) for q in X_batch.tolist()]
        logger.debug(
            "Built %d prompts, first prompt length: %d chars",
            len(full_prompts),
            len(full_prompts[0]) if full_prompts else 0,
        )

        # Tokenize each prompt
        prompt_encodings = [
            self.tokenizer(p, add_special_tokens=False, return_tensors="pt").input_ids[
                0
            ]
            for p in full_prompts
        ]

        target_tensor = torch.tensor(target_ids, dtype=torch.long)
        T = len(target_ids)

        # Concat prompt + target for each sample
        sequences: list[torch.Tensor] = []
        for prompt_ids in prompt_encodings:
            seq = torch.cat([prompt_ids, target_tensor])
            sequences.append(seq)

        # Pad to same length (prompts may differ due to "..." collapse)
        max_len = max(s.shape[0] for s in sequences)

        if self.tokenizer.pad_token_id is None:
            raise ValueError(
                "tokenizer.pad_token_id is None. Set it before using "
                "QuestionTeacherForcing (e.g. tokenizer.pad_token = tokenizer.eos_token)."
            )

        batch_ids = torch.full(
            (len(sequences), max_len),
            self.tokenizer.pad_token_id,
            dtype=torch.long,
        )
        attn_mask = torch.zeros((len(sequences), max_len), dtype=torch.long)

        for i, seq in enumerate(sequences):
            # LEFT-pad (decoder models need left padding)
            pad_len = max_len - seq.shape[0]
            batch_ids[i, pad_len:] = seq
            attn_mask[i, pad_len:] = 1

        batch_ids = batch_ids.to(device)
        attn_mask = attn_mask.to(device)

        logger.debug(
            "Forward pass: batch=%d, max_len=%d, target_len=%d, device=%s",
            len(sequences),
            max_len,
            T,
            device,
        )

        with torch.no_grad():
            outputs = self.model(
                input_ids=batch_ids,
                attention_mask=attn_mask,
                return_dict=True,
            )
            logits = outputs.logits.detach().cpu().float().numpy().astype("float64")

        # Extract logits at target positions
        # For each sample, the target tokens start at (seq_len - T) in the unpadded sequence.
        # The predicting position for target[t] is one step before: (pad_len + prompt_len + t - 1)
        result = np.zeros((len(sequences), T, logits.shape[-1]), dtype=np.float64)
        for i, seq in enumerate(sequences):
            pad_len = max_len - seq.shape[0]
            prompt_len = seq.shape[0] - T
            for t in range(T):
                pred_pos = pad_len + prompt_len + t - 1
                result[i, t] = logits[i, pred_pos]

        return result  # (B, T, vocab_size)

    def _logits_to_logodds(
        self, logits: NDArray[np.float64], target_ids: list[int]
    ) -> NDArray[np.float64]:
        """Convert logits to log-odds for target token IDs.

        Mathematically equivalent to logit(softmax(logits))[target], but
        computed directly from logits to avoid ±inf when softmax saturates:

            logit(p_i) = log(p_i / (1 - p_i))
                      = logits[i] - logsumexp(logits[j != i])

        The naive softmax→logit round-trip produces +inf whenever the top
        logit gap exceeds ~36 (softmax rounds to exactly 1.0 in fp64), which
        triggers nan in downstream SHAP arithmetic (inf - inf).

        Args:
            logits: Array of shape (B, T, vocab_size).
            target_ids: Token IDs to extract log-odds for.

        Returns:
            Log-odds array of shape (B, T).
        """
        from scipy.special import logsumexp

        B, T, V = logits.shape
        result = np.zeros((B, T), dtype=np.float64)
        for t in range(T):
            i = target_ids[t]
            masked = logits[:, t, :].copy()
            masked[:, i] = -np.inf
            result[:, t] = logits[:, t, i] - logsumexp(masked, axis=-1)

        logger.debug(
            "Log-odds computed: shape=%s, range=[%.2f, %.2f]",
            result.shape,
            result.min(),
            result.max(),
        )

        return result  # (B, T)
