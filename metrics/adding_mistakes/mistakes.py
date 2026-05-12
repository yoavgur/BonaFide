"""Mistake generation via Gemini for the Adding Mistakes metric.

Uses the Judge class from isolate_steps.py to generate mistaken versions
of reasoning steps via a structured prompt.
"""

from __future__ import annotations

import logging
from typing import Any

from metrics.adding_mistakes.prompts import format_mistake_prompt

logger = logging.getLogger(__name__)


def _get_judge(model_name: str) -> Any:
    """Lazy-import and create a Judge instance.

    Avoids pulling in the heavyweight isolate_steps module at import time.
    """
    from isolate_steps import Judge

    return Judge(model_name=model_name)


def _dump_response_debug(response: Any, index: int | None = None) -> None:
    """Log detailed info about a Gemini response for debugging truncation."""
    prefix = f"[response {index}] " if index is not None else ""

    # finish_reason
    finish_reason = None
    if hasattr(response, "candidates") and response.candidates:
        candidate = response.candidates[0]
        finish_reason = getattr(candidate, "finish_reason", None)
    print(f"  {prefix}finish_reason: {finish_reason}")

    # token counts
    usage = getattr(response, "usage_metadata", None)
    if usage:
        print(f"  {prefix}tokens — prompt: {getattr(usage, 'prompt_token_count', '?')}, "
              f"candidates: {getattr(usage, 'candidates_token_count', '?')}, "
              f"thoughts: {getattr(usage, 'thoughts_token_count', '?')}, "
              f"total: {getattr(usage, 'total_token_count', '?')}")

    # all parts (thought vs text)
    if hasattr(response, "candidates") and response.candidates:
        parts = response.candidates[0].content.parts
        for j, part in enumerate(parts):
            kind = "thought" if getattr(part, "thought", False) else "text"
            text = getattr(part, "text", "") or ""
            print(f"  {prefix}part[{j}] ({kind}, {len(text)} chars): {text[:150]!r}{'...' if len(text) > 150 else ''}")

    # raw .text
    raw_text = response.text if hasattr(response, "text") else str(response)
    print(f"  {prefix}response.text ({len(raw_text)} chars):")
    print(raw_text)


_START_DELIM = "<<<MISTAKE>>>"
_END_DELIM = "<<<END>>>"


def _parse_mistake_response(response: Any, index: int | None = None) -> str:
    """Extract the mistaken step from a Gemini response using delimiters.

    Expects the mistaken step between <<<MISTAKE>>> and <<<END>>> delimiters.

    Raises:
        ValueError: If the delimiters are missing or the content is empty.
    """
    text = response.text if hasattr(response, "text") else str(response)

    start = text.find(_START_DELIM)
    end = text.find(_END_DELIM)

    if start < 0 or end < 0 or end <= start:
        raise ValueError(
            f"Missing <<<MISTAKE>>>...<<<END>>> delimiters in Gemini response: {text!r}"
        )

    result = text[start + len(_START_DELIM):end].strip()
    if not result:
        raise ValueError(
            f"Empty content between delimiters in Gemini response: {text!r}"
        )

    return result


def generate_mistakes(
    sentences: list[str],
    question: str,
    mistake_model: str,
) -> tuple[list[str], float]:
    """Generate mistaken versions of reasoning steps via Gemini.

    Args:
        sentences: The original reasoning steps to corrupt.
        question: The original question (for context in the prompt).
        mistake_model: Gemini model name for the Judge.

    Returns:
        (mistakes, api_cost_usd): List of mistaken step strings and total API cost.

    Raises:
        ValueError: If any response cannot be parsed.
    """
    if not sentences:
        return [], 0.0

    judge = _get_judge(mistake_model)

    # Format prompts for all sentences
    prompts = [
        format_mistake_prompt(question, sentence)
        for sentence in sentences
    ]

    logger.info(
        "Generating %d mistakes via %s", len(prompts), mistake_model,
    )

    # Batch generation via Judge
    responses = judge.run_batch(prompts)

    # Parse responses
    mistakes = []
    for i, response in enumerate(responses):
        try:
            mistake = _parse_mistake_response(response, index=i)
        except ValueError as e:
            print(f"  === DEBUG: failed response {i} ===")
            _dump_response_debug(response, index=i)
            raise ValueError(
                f"Failed to parse mistake for sentence {i} "
                f"({sentences[i][:50]!r}...): {e}"
            ) from e
        mistakes.append(mistake)

    cost = judge.total_cost
    logger.info("Generated %d mistakes successfully (cost=$%.4f)", len(mistakes), cost)
    return mistakes, cost


def generate_single_mistake(
    sentence: str,
    question: str,
    mistake_model: str,
) -> str:
    """Generate a mistaken version of a single reasoning step.

    Args:
        sentence: The original reasoning step to corrupt.
        question: The original question (for context).
        mistake_model: Gemini model name for the Judge.

    Returns:
        The mistaken step string.

    Raises:
        ValueError: If the response cannot be parsed.
    """
    judge = _get_judge(mistake_model)
    prompt = format_mistake_prompt(question, sentence)

    logger.info("Generating single mistake via %s", mistake_model)
    response = judge.run(prompt)

    try:
        return _parse_mistake_response(response, index=0)
    except ValueError:
        print("  === DEBUG: failed single response ===")
        _dump_response_debug(response, index=0)
        raise
