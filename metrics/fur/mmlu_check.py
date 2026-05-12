"""MMLU sanity-check sample for FUR.

10 hand-picked MMLU-style questions across diverse subjects. Used as the paper's
"general capabilities" check (§6.1, "Gen" column in Table 1): we predict the
model's answer letter (constrained argmax over A/B/C/D) before AND after each
step's unlearning, and report the fraction of unchanged predictions.

If unlearning is precise, this should stay near 1.0 — i.e. unlearning a single
reasoning step shouldn't break the model's general knowledge.
"""

from __future__ import annotations

import logging

import torch

logger = logging.getLogger(__name__)


# 10 MMLU-style questions across diverse subjects. Hand-picked to be
# unambiguous and answerable by any competent LLM, so any change in
# prediction post-unlearning indicates damage rather than ambiguity.
MMLU_QUESTIONS: list[dict] = [
    {
        "subject": "geography",
        "question": "What is the capital of France?",
        "choices": ["London", "Berlin", "Paris", "Madrid"],
        "answer": 2,
    },
    {
        "subject": "elementary math",
        "question": "What is 12 multiplied by 13?",
        "choices": ["146", "156", "166", "176"],
        "answer": 1,
    },
    {
        "subject": "biology",
        "question": "Which of these is the powerhouse of the cell?",
        "choices": ["Nucleus", "Ribosome", "Mitochondrion", "Golgi apparatus"],
        "answer": 2,
    },
    {
        "subject": "physics",
        "question": "What is the SI unit of electric current?",
        "choices": ["Volt", "Watt", "Ampere", "Ohm"],
        "answer": 2,
    },
    {
        "subject": "chemistry",
        "question": "What is the chemical symbol for gold?",
        "choices": ["Go", "Gd", "Au", "Ag"],
        "answer": 2,
    },
    {
        "subject": "history",
        "question": "In what year did World War II end?",
        "choices": ["1943", "1945", "1947", "1950"],
        "answer": 1,
    },
    {
        "subject": "literature",
        "question": "Who wrote the play 'Romeo and Juliet'?",
        "choices": ["Charles Dickens", "Jane Austen", "William Shakespeare", "Mark Twain"],
        "answer": 2,
    },
    {
        "subject": "computer science",
        "question": "What does HTTP stand for?",
        "choices": [
            "Hyper Transfer Text Protocol",
            "Hypertext Transfer Protocol",
            "Hypertext Transit Protocol",
            "High Transfer Text Protocol",
        ],
        "answer": 1,
    },
    {
        "subject": "astronomy",
        "question": "Which planet is closest to the Sun?",
        "choices": ["Venus", "Earth", "Mars", "Mercury"],
        "answer": 3,
    },
    {
        "subject": "anatomy",
        "question": "How many chambers does a human heart have?",
        "choices": ["Two", "Three", "Four", "Five"],
        "answer": 2,
    },
]


def _format_prompt(item: dict) -> str:
    """Format an MMLU item as a multiple-choice prompt ending in 'Answer:'."""
    letters = ["A", "B", "C", "D"]
    choices = "\n".join(f"{l}. {c}" for l, c in zip(letters, item["choices"]))
    return (
        f"The following is a multiple choice question.\n\n"
        f"{item['question']}\n\n"
        f"{choices}\n\n"
        f"Answer:"
    )


def _resolve_choice_token_ids(tokenizer) -> list[int]:
    """Get the token IDs for ' A', ' B', ' C', ' D' (with leading space).

    Most tokenizers split letters following whitespace into a separate token.
    We take the *last* token of the encoded ' X' to handle BOS/special tokens.
    """
    ids = []
    for letter in ["A", "B", "C", "D"]:
        # Use leading space — that's what comes after "Answer:"
        toks = tokenizer.encode(" " + letter, add_special_tokens=False)
        if not toks:
            raise ValueError(f"Tokenizer produced empty encoding for ' {letter}'")
        ids.append(toks[-1])
    if len(set(ids)) != 4:
        logger.warning(
            "MMLU choice tokens are not all distinct: %s. "
            "Constrained argmax may behave unexpectedly.",
            ids,
        )
    return ids


def build_mmlu_tensors(
    tokenizer,
    device: torch.device,
    max_n: int = 10,
) -> tuple[list[torch.Tensor], list[int]]:
    """Build tokenized MMLU prompts and the 4 choice-letter token IDs.

    Args:
        tokenizer: HF tokenizer.
        device: Where to place the prompt tensors.
        max_n: How many MMLU questions to use (capped at the bundled set size).

    Returns:
        (prompts, choice_token_ids):
        - prompts: list of (1, T) input_ids tensors, one per question.
        - choice_token_ids: [tok_A, tok_B, tok_C, tok_D] — same for all prompts.
    """
    n = min(max_n, len(MMLU_QUESTIONS))
    items = MMLU_QUESTIONS[:n]
    prompts: list[torch.Tensor] = []
    for item in items:
        prompt_str = _format_prompt(item)
        ids = tokenizer(prompt_str, return_tensors="pt").input_ids.to(device)
        prompts.append(ids)
    choice_ids = _resolve_choice_token_ids(tokenizer)
    return prompts, choice_ids


def constrained_argmax_letter(
    last_logits: torch.Tensor,
    choice_token_ids: list[int],
) -> int:
    """Pick the letter index (0-3) with highest logit among the 4 choice tokens.

    Args:
        last_logits: Shape (vocab_size,) — logits at the final position.
        choice_token_ids: [tok_A, tok_B, tok_C, tok_D].

    Returns:
        Index 0-3 corresponding to A/B/C/D.
    """
    sub = last_logits[choice_token_ids]
    return int(sub.argmax().item())
