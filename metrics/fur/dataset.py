"""Forget and retain set construction for FUR unlearning.

Builds the D_FG (forget) and D_RT (retain) datasets used by NPO+KL.

Origin: Adapted from https://github.com/technion-cs-nlp/parametric-faithfulness/blob/main/data.py
Changes:
  - Simplified to work with MetricContext instead of file-based pipeline
  - Removed CoT generation/caching (handled upstream)
  - Removed OTFDataset and DualCollator (unused in our pipeline)
  - Kept SegmentOTFDataset core logic but simplified interface
  - FRCollator preserved with minor cleanup
  - Added build_forget_pairs() and build_retain_pairs() as standalone functions
"""

from __future__ import annotations

import random

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset

from metrics.fur.segmentation import (
    StepInfo,
    align_cot_to_pos,
    detect_whitespace_char,
    get_nlp,
    segment_and_filter,
)

IGNORE_IDX = -100


def left_pad_sequence(
    vector_list: list[torch.Tensor], padding_value: int
) -> torch.Tensor:
    """Left-pad a list of 1D tensors to the same length.

    Origin: From reference data.py, unchanged.
    """
    N = len(vector_list)
    T = max(len(f) for f in vector_list)

    ret = torch.full((N, T), fill_value=padding_value, dtype=vector_list[0].dtype)
    for i, v_i in enumerate(vector_list):
        L = len(v_i)
        ret[i, T - L :] = v_i  # Leave padding on left
    return ret


def encode_step_for_unlearning(
    tokenizer,
    prompt: str,
    step_text: str,
    preceding_steps: list[str] | None = None,
    content_words: list | None = None,
    max_seq_len: int = 1024,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
    """Encode a question + CoT step for unlearning.

    Origin: Adapted from reference data.py qcot_encoder().
    Changes: Takes explicit prompt/step/preceding_steps instead of raw dicts.
             Uses content_words list for POS filtering instead of re-running SpaCy.

    Args:
        tokenizer: HuggingFace tokenizer.
        prompt: The question/instruction prompt.
        step_text: The CoT step to unlearn.
        preceding_steps: Previous CoT steps (prefix context).
        content_words: Pre-computed Word objects for POS filtering. If None, all
            tokens in the step are targeted.

    Returns:
        (input_ids, labels, attention_mask, num_targets): Tensors ready for training.
        Labels have IGNORE_IDX for non-target positions.
    """
    # Build prefix: prompt + preceding steps
    if preceding_steps:
        prefix = prompt + "\n\n" + "\n".join(preceding_steps) + "\n"
    else:
        prefix = prompt + "\n\n"

    prefix_tokens = tokenizer.encode(prefix, add_special_tokens=False, return_tensors="pt")[0]

    # Encode the step
    if content_words is not None:
        # Use the pre-aligned tokenization from segmentation
        whitespace_char = detect_whitespace_char(tokenizer)
        nlp = get_nlp()
        step_token_ids, words = align_cot_to_pos(step_text, tokenizer, whitespace_char, nlp)
    else:
        step_token_ids = tokenizer.encode(
            step_text, add_special_tokens=False, return_tensors="pt"
        ).squeeze()

    # Truncate prefix if total would exceed max_seq_len (keep step intact,
    # trim prefix from the LEFT so the step + nearby context is preserved)
    total_len = len(prefix_tokens) + len(step_token_ids)
    if total_len > max_seq_len and max_seq_len > 0:
        max_prefix = max_seq_len - len(step_token_ids)
        if max_prefix > 0:
            prefix_tokens = prefix_tokens[-max_prefix:]  # keep rightmost (closest to step)
        else:
            prefix_tokens = prefix_tokens[:0]  # empty prefix, step alone is too long
            step_token_ids = step_token_ids[:max_seq_len]

    # Concatenate prefix + step
    input_ids = torch.cat((prefix_tokens, step_token_ids), dim=0)
    labels = input_ids.clone()
    attention_mask = torch.ones_like(input_ids)

    # Mask prefix from loss (only unlearn the step)
    prefix_len = len(prefix_tokens)
    labels[:prefix_len] = IGNORE_IDX

    # If POS filtering, mask function words in the step
    if content_words is not None:
        for w in words:
            if not w.is_content():
                labels[prefix_len + w.span_start : prefix_len + w.span_end] = IGNORE_IDX

    num_targets = int((labels != IGNORE_IDX).sum().item())
    return input_ids, labels, attention_mask, num_targets


def build_forget_pairs(
    tokenizer,
    prompt: str,
    step: StepInfo,
    preceding_step_texts: list[str],
    max_seq_len: int = 1024,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
    """Build the forget set for a single CoT step.

    Origin: Adapted from reference data.py cot_to_otfd() with strategy='sentencize'.
    Changes: Simplified to work with StepInfo instead of raw dicts.

    Returns:
        (input_ids, labels, attention_mask, num_targets) for the forget example.
    """
    return encode_step_for_unlearning(
        tokenizer=tokenizer,
        prompt=prompt,
        step_text=step.text,
        preceding_steps=preceding_step_texts,
        content_words=step.content_words,
        max_seq_len=max_seq_len,
    )


def build_paraphrase_forget_pairs(
    tokenizer,
    prompt: str,
    paraphrases: list[str],
    preceding_step_texts: list[str],
    max_seq_len: int = 1024,
) -> list[tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]]:
    """Build forget pairs for paraphrases of a CoT step.

    Unlike build_forget_pairs, these have no POS-aligned content_words — all
    tokens in the paraphrase are targeted for unlearning (content_words=None).

    Args:
        tokenizer: HuggingFace tokenizer.
        prompt: The question/instruction prompt.
        paraphrases: List of paraphrase strings for the step.
        preceding_step_texts: Previous CoT steps (prefix context).
        max_seq_len: Maximum sequence length.

    Returns:
        List of (input_ids, labels, attention_mask, num_targets) tuples.
    """
    pairs = []
    for para_text in paraphrases:
        encoded = encode_step_for_unlearning(
            tokenizer=tokenizer,
            prompt=prompt,
            step_text=para_text,
            preceding_steps=preceding_step_texts,
            content_words=None,  # No POS filtering — target all tokens
            max_seq_len=max_seq_len,
        )
        pairs.append(encoded)
    return pairs


def build_retain_pairs(
    tokenizer,
    other_instances: list[dict],
    min_content_tokens: int = 2,
    n: int = 4,
    seed: int | None = None,
    max_seq_len: int = 1024,
) -> list[tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]]:
    """Build the retain set from other dataset instances.

    Origin: Adapted from reference data.py cot_to_otfd() retain construction.
    Changes: Works with other_instances dicts instead of file-based cot loading.

    Samples n CoT steps from other instances, applies the same content-word
    filtering, and returns encoded pairs for KL regularization.

    Args:
        tokenizer: HuggingFace tokenizer.
        other_instances: List of {"question": str, "cot": str, "answer": str} dicts.
        min_content_tokens: Minimum content words per step.
        n: Number of retain steps to sample.
        seed: Random seed for sampling (None = no seeding).

    Returns:
        List of (input_ids, labels, attention_mask, num_targets) tuples.
    """
    if seed is not None:
        rng = random.Random(seed)
    else:
        rng = random.Random()

    # Collect candidate steps from other instances
    all_candidate_steps = []
    for inst in other_instances:
        steps = segment_and_filter(
            inst["cot"],
            tokenizer,
            min_content_tokens=min_content_tokens,
        )
        for i, step in enumerate(steps):
            preceding = [s.text for s in steps[:i]]
            all_candidate_steps.append((inst["question"], step, preceding))

    if not all_candidate_steps:
        return []

    # Sample n steps
    n = min(n, len(all_candidate_steps))
    sampled = rng.sample(all_candidate_steps, n)

    retain_pairs = []
    for question, step, preceding in sampled:
        encoded = encode_step_for_unlearning(
            tokenizer=tokenizer,
            prompt=question,
            step_text=step.text,
            preceding_steps=preceding if preceding else None,
            content_words=step.content_words,
            max_seq_len=max_seq_len,
        )
        # Only keep if enough targets
        if encoded[3] > min_content_tokens:
            retain_pairs.append(encoded)

    return retain_pairs


class FRCollator:
    """Collator for forget-retain paired batches with left-padding.

    Origin: From reference data.py, unchanged core logic.
    Changes: Minor cleanup, type hints, device handling.
    """

    def __init__(self, tokenizer, device: torch.device | str = "cpu"):
        self.tokenizer = tokenizer
        self.device = device
        self.pad_token_id = tokenizer.encode(tokenizer.pad_token)[0]

    def __call__(
        self, samples: list[tuple]
    ) -> tuple[tuple[torch.Tensor, ...], tuple[torch.Tensor, ...]]:
        """Collate forget-retain pairs into padded batches.

        Args:
            samples: List of ((E_f, L_f, A_f, _), (E_r, L_r, A_r, _)) tuples.

        Returns:
            ((E_f_batch, L_f_batch, A_f_batch), (E_r_batch, L_r_batch, A_r_batch))
        """
        F, R = zip(*samples)

        E_fb = [f[0] for f in F]
        L_fb = [f[1] for f in F]
        A_fb = [f[2] for f in F]

        E_rb = [r[0] for r in R]
        L_rb = [r[1] for r in R]
        A_rb = [r[2] for r in R]

        E_fib = left_pad_sequence(E_fb, padding_value=self.pad_token_id).to(self.device)
        L_fib = left_pad_sequence(L_fb, padding_value=IGNORE_IDX).to(self.device)
        A_fib = left_pad_sequence(A_fb, padding_value=0).to(self.device)

        E_rib = left_pad_sequence(E_rb, padding_value=self.pad_token_id).to(self.device)
        L_rib = left_pad_sequence(L_rb, padding_value=IGNORE_IDX).to(self.device)
        A_rib = left_pad_sequence(A_rb, padding_value=0).to(self.device)

        return (E_fib, L_fib, A_fib), (E_rib, L_rib, A_rib)
