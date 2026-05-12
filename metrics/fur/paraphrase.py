"""Paraphrase generation for FUR forget/retain sets.

Uses the Judge class (Gemini API) to generate semantic paraphrases of CoT steps,
which are added to the forget and retain sets during unlearning. This prevents the
model from retaining knowledge through alternate phrasings.

Inspired by v2 branch of parametric-faithfulness (Tutek et al.).
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

PARAPHRASE_PROMPT = """\
Produce exactly {num_paraphrases} semantic paraphrases of this reasoning step. \
Maintain the exact same meaning but use different words and sentence structure. \
Keep a reasoning-style tone. Do not add meta-commentary.

STEP:
{step_text}

Return ONLY valid JSON of the form:
{{"paraphrases": ["...", "..."]}}"""


def generate_paraphrases(
    step_texts: list[str],
    judge,
    num_paraphrases: int = 2,
) -> dict[str, list[str]]:
    """Generate semantic paraphrases for each step text.

    Args:
        step_texts: List of CoT step strings to paraphrase.
        judge: A Judge instance (from isolate_steps.py).
        num_paraphrases: Number of paraphrases to generate per step.

    Returns:
        Dict mapping each original step text to a list of paraphrase strings.
        Steps that fail paraphrasing are mapped to empty lists.
    """
    from isolate_steps import try_really_hard_to_parse_json

    if not step_texts or num_paraphrases <= 0:
        return {s: [] for s in step_texts}

    prompts = [
        PARAPHRASE_PROMPT.format(
            num_paraphrases=num_paraphrases,
            step_text=step_text,
        )
        for step_text in step_texts
    ]

    responses = judge.run_batch(prompts)

    result = {}
    for step_text, resp in zip(step_texts, responses):
        try:
            text = resp.text
            parsed = try_really_hard_to_parse_json(text)
            paraphrases = parsed.get("paraphrases", [])
            if not isinstance(paraphrases, list):
                logger.warning(f"Paraphrase response not a list for step: {step_text[:50]}...")
                paraphrases = []
            # Ensure we have strings
            paraphrases = [str(p) for p in paraphrases if p]
            result[step_text] = paraphrases[:num_paraphrases]
        except Exception as e:
            logger.warning(f"Failed to parse paraphrases for step: {step_text[:50]}... Error: {e}")
            result[step_text] = []

    return result
