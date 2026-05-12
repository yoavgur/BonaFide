"""Truncation utilities for the Early Answering metric.

Splits CoT into sentences and builds progressive truncations at sampled
sentence boundaries.
"""

from __future__ import annotations

import numpy as np


def _split_into_sentences(cot: str) -> list[str]:
    """Lazy-import wrapper for split_into_sentences from isolate_steps.

    Avoids pulling in the heavyweight isolate_steps module (which requires
    GEMINI_API_KEY, torch, sentence-transformers, etc.) at import time.
    """
    from sentence_splitting import split_into_sentences

    return split_into_sentences(cot)


def build_truncated_cots(
    cot: str,
    max_points: int,
) -> tuple[list[str], list[int], list[str]]:
    """Split a CoT into sentences and build progressive truncations.

    Args:
        cot: The full chain-of-thought text.
        max_points: Maximum number of truncation points to sample.
            If the CoT has fewer sentences than this, all boundaries
            are used. Otherwise, evenly-spaced boundaries are sampled.

    Returns:
        A tuple of:
        - sentences: The individual sentences from the CoT.
        - truncation_indices: The sentence indices used as truncation
          points. Index k means "include sentences 0..k-1" (i.e.,
          k=0 means empty CoT, k=1 means first sentence only, etc.).
          Does NOT include len(sentences) (the full CoT).
        - truncated_cots: The truncated CoT strings at each index.
          truncated_cots[i] corresponds to truncation_indices[i].
    """
    sentences = _split_into_sentences(cot)
    n = len(sentences)

    if n == 0:
        return [], [], []

    # Select truncation indices: 0 through n-1 (excluding n = full CoT).
    # k=0 means "no meaningful CoT" — the metric handles this by using a
    # filler token with answer prefix to prevent thinking models from
    # restarting reasoning.
    if n == 1:
        # Only one sentence — the only truncation point is k=0 (no CoT)
        truncation_indices = [0]
    elif n <= max_points:
        # Use all sentence boundaries from 0 to n-1
        truncation_indices = list(range(n))
    else:
        # Sample evenly-spaced indices from 0 to n-1
        truncation_indices = sorted(
            set(np.linspace(0, n - 1, max_points, dtype=int).tolist())
        )

    # Build truncated CoT strings
    truncated_cots = []
    for k in truncation_indices:
        truncated_cots.append(" ".join(sentences[:k]))

    return sentences, truncation_indices, truncated_cots


def find_step_index(
    cot: str,
    step_span: tuple[int, int],
    sentences: list[str],
) -> int:
    """Map a character span in the CoT to the sentence index containing it.

    Args:
        cot: The full chain-of-thought text.
        step_span: (char_start, char_end) into cot identifying the step.
        sentences: The sentences from split_into_sentences(cot).

    Returns:
        The index of the sentence that contains the step's start position.

    Raises:
        ValueError: If the span doesn't map to any sentence.
    """
    char_start, char_end = step_span

    if char_start < 0 or char_end > len(cot) or char_start >= char_end:
        raise ValueError(
            f"Invalid step_span ({char_start}, {char_end}) for CoT of "
            f"length {len(cot)}"
        )

    # Find each sentence's position in the original CoT
    search_start = 0
    for i, sentence in enumerate(sentences):
        pos = cot.find(sentence, search_start)
        if pos == -1:
            raise ValueError(
                f"Could not locate sentence {i} ({sentence[:50]!r}...) "
                f"in the CoT starting from position {search_start}"
            )
        sent_end = pos + len(sentence)
        # Check if the step's start falls within this sentence
        if pos <= char_start < sent_end:
            return i
        search_start = sent_end

    raise ValueError(
        f"step_span ({char_start}, {char_end}) does not fall within any "
        f"sentence in the CoT. The CoT has {len(sentences)} sentences."
    )
