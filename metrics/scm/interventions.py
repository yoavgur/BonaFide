"""Intervention functions for the SCM causal analysis metric.

H1 — CoT corruption via sentence swap from a donor instance.
H2 — Instruction modification via role prepending.
"""

from __future__ import annotations

import math

from sentence_splitting import split_into_sentences


def corrupt_cot_swap(cot: str, donor_cot: str, ratio: float = 1.0) -> str:
    """Replace the last *ratio* fraction of CoT sentences with donor sentences.

    With the default ratio=1.0 (full-CoT corruption), this matches Bao et al.'s
    "random CoT" intervention: the entire CoT is replaced with an unrelated
    donor CoT. Smaller ratios were used in earlier versions but lacked
    citation; we now default to full-CoT corruption.

    Args:
        cot: The original chain-of-thought text.
        donor_cot: A CoT from a different instance (used as replacement).
        ratio: Fraction of sentences to replace (from the end). 1.0 = full CoT.

    Returns:
        The corrupted CoT string.

    Raises:
        ValueError: If either CoT is empty or produces no sentences.
    """
    if not cot.strip():
        raise ValueError("cot is empty")
    if not donor_cot.strip():
        raise ValueError("donor_cot is empty")

    cot_sents = split_into_sentences(cot)
    donor_sents = split_into_sentences(donor_cot)

    if not cot_sents:
        raise ValueError("cot produced no sentences after tokenization")
    if not donor_sents:
        raise ValueError("donor_cot produced no sentences after tokenization")

    n_replace = max(1, math.ceil(ratio * len(cot_sents)))
    n_keep = len(cot_sents) - n_replace

    # Take the last n_replace sentences from the donor (or all if donor is shorter)
    replacement = donor_sents[-n_replace:] if len(donor_sents) >= n_replace else donor_sents

    corrupted_sents = cot_sents[:n_keep] + replacement
    return " ".join(corrupted_sents)


def modify_instruction_role(question: str, role: str) -> str:
    """Prepend a role instruction to the question.

    Tests H2: whether changing the instruction wording (while keeping the CoT
    constant) changes the model's answer.  If it does, the instruction is
    bypassing the CoT to directly influence the answer.

    Args:
        question: The original question / instruction text.
        role: A professional role (e.g. "detective", "chef").

    Returns:
        The modified question string.
    """
    return f"Imagine you are a {role}. {question}"
