"""Paraphrase generation via Gemini for the Paraphrasing metric.

Uses the Judge class from isolate_steps.py to paraphrase CoT text.
The paraphrasing model does NOT see the original question.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any

from metrics.paraphrasing.prompts import format_paraphrase_prompt

logger = logging.getLogger(__name__)


def _get_judge(model_name: str) -> Any:
    """Lazy-import and create a Judge instance."""
    from isolate_steps import Judge

    return Judge(model_name=model_name)


def _parse_paraphrase_response(response: Any) -> str:
    """Extract the paraphrased text from a Gemini Judge response.

    Expects JSON with key "paraphrased_text" in the response.

    Raises:
        ValueError: If the response cannot be parsed.
    """
    text = response.text if hasattr(response, "text") else str(response)
    text = text.strip()

    if not text:
        raise ValueError("Gemini returned empty paraphrase response")

    # Strip markdown code fences if present
    if text.startswith("```"):
        lines = text.split("\n")
        lines = [l for l in lines if not l.strip().startswith("```")]
        text = "\n".join(lines).strip()

    def _fix_escapes(s: str) -> str:
        """Fix invalid JSON backslash escapes (e.g. \\e from LaTeX)."""
        return re.sub(r'\\(?!["\\/bfnrtu])', r'\\\\', s)

    data = None
    for attempt_text in [text, _fix_escapes(text)]:
        try:
            data = json.loads(attempt_text)
            break
        except json.JSONDecodeError:
            continue
    if data is None:
        # Try to find JSON object in the text
        start = text.find("{")
        end = text.rfind("}") + 1
        if start >= 0 and end > start:
            for attempt_text in [text[start:end], _fix_escapes(text[start:end])]:
                try:
                    data = json.loads(attempt_text)
                    break
                except json.JSONDecodeError:
                    continue
    if data is None:
        # Last resort: recover truncated JSON like {"paraphrased_text": "some text...
        # This happens when the model hits its output token limit mid-response.
        m = re.search(r'"paraphrased_text"\s*:\s*"', text)
        if m:
            value_start = m.end()
            # Take everything after the opening quote; the closing " and } are missing
            truncated = text[value_start:]
            # Remove a trailing incomplete escape or quote if present
            truncated = truncated.rstrip('"}\n\r ')
            # Un-escape JSON string escapes so the text is usable
            try:
                truncated = json.loads('"' + truncated + '"')
            except json.JSONDecodeError:
                truncated = truncated.replace('\\"', '"').replace('\\n', '\n')
            if truncated.strip():
                logger.warning(
                    "Recovered truncated paraphrase (%d chars) — "
                    "model likely hit output token limit",
                    len(truncated),
                )
                return truncated.strip()
        logger.error(
            "=== PARAPHRASE PARSE FAILURE DIAGNOSTIC ===\n"
            "Response object: %r\n"
            "Response text length: %d\n"
            "Full response text:\n%s\n"
            "=== END DIAGNOSTIC ===",
            response, len(text), text,
        )
        raise ValueError(
            f"Could not parse JSON from Gemini paraphrase response: "
            f"{text[:200]!r}"
        )

    if "paraphrased_text" not in data:
        raise ValueError(
            f"Gemini response JSON missing 'paraphrased_text' key. "
            f"Got keys: {list(data.keys())}"
        )

    result = data["paraphrased_text"]
    if not isinstance(result, str) or not result.strip():
        raise ValueError(
            f"'paraphrased_text' is empty or not a string: {result!r}"
        )

    return result.strip()


def paraphrase_texts(
    texts: list[str],
    paraphrase_model: str,
) -> tuple[list[str], float]:
    """Paraphrase multiple CoT subsequences via Gemini (batched).

    The original question is intentionally NOT provided to the
    paraphrasing model (per the paper's design).

    Args:
        texts: List of CoT subsequence texts to paraphrase.
        paraphrase_model: Gemini model name for the Judge.

    Returns:
        (paraphrased_texts, api_cost_usd): List of paraphrased texts and total API cost.

    Raises:
        ValueError: If any response cannot be parsed.
    """
    if not texts:
        return [], 0.0

    judge = _get_judge(paraphrase_model)

    prompts = [format_paraphrase_prompt(t) for t in texts]

    logger.info(
        "Paraphrasing %d texts via %s", len(prompts), paraphrase_model,
    )

    responses = judge.run_batch(prompts, max_output_tokens=50000)

    paraphrased = []
    for i, response in enumerate(responses):
        try:
            result = _parse_paraphrase_response(response)
            paraphrased.append(result)
        except ValueError as e:
            # Retry once — truncated responses often succeed on a second attempt
            logger.warning(
                "Paraphrase parse failed for text %d (%s...): %s — retrying",
                i, texts[i][:50], e,
            )
            retry_response = judge.run(prompts[i], max_output_tokens=50000)
            result = _parse_paraphrase_response(retry_response)
            paraphrased.append(result)
            logger.info("Retry succeeded for text %d", i)

    cost = judge.total_cost
    logger.info("Paraphrased %d texts successfully (cost=$%.4f)", len(paraphrased), cost)
    return paraphrased, cost
