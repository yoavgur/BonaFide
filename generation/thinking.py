"""Extract Chain-of-Thought and answer from model output."""

from __future__ import annotations

import json
import re

from generation.model_registry import ANSWER_JSON_PATTERN, ANSWER_JSON_UNQUOTED_PATTERN


def _extract_json_answer(text: str) -> str | None:
    """Try to extract final_answer from a JSON object in the text.

    Returns the extracted answer string, or None if not found.
    """
    # Try quoted value first
    m = re.search(ANSWER_JSON_PATTERN, text, flags=re.DOTALL)
    if m:
        # Unescape JSON string escapes
        raw = m.group("json_answer")
        try:
            return json.loads(f'"{raw}"')
        except json.JSONDecodeError:
            return raw

    # Fallback: unquoted value (numbers, booleans, etc.)
    m = re.search(ANSWER_JSON_UNQUOTED_PATTERN, text, flags=re.DOTALL)
    if m:
        return m.group("json_answer").strip().strip('"')

    return None


def split_thinking(
    raw_output: str,
    thinking_tag: str | None,
    cot_pattern: str | None = None,
) -> tuple[str, str]:
    """Split raw model output into (cot, answer).

    Args:
        raw_output: Full model output text.
        thinking_tag: Tag name (e.g., "think" for <think>...</think>).
            None means model doesn't use thinking tags.
        cot_pattern: Regex with groups 'cot' and 'answer'. Legacy fallback
            if JSON parsing fails.

    Returns:
        (cot, answer) tuple. Attempts to parse {"final_answer": "..."} from
        the output first. Falls back to tag-based or pattern-based splitting.
    """
    if not raw_output:
        return "", ""

    # --- Tag-based splitting (thinking models) ---
    if thinking_tag is not None:
        close_tag = f"</{thinking_tag}>"
        open_tag = f"<{thinking_tag}>"

        parts = raw_output.split(close_tag, maxsplit=1)

        if len(parts) < 2:
            # No closing tag — model likely hit token limit mid-thinking
            cot = raw_output
            if cot.lstrip().startswith(open_tag):
                cot = cot.lstrip()[len(open_tag):]
            return cot.strip(), ""

        cot = parts[0]
        answer_part = parts[1]

        # Strip opening tag from CoT
        if cot.lstrip().startswith(open_tag):
            cot = cot.lstrip()[len(open_tag):]

        # Try JSON extraction from the answer portion
        json_answer = _extract_json_answer(answer_part)
        if json_answer is not None:
            return cot.strip(), json_answer.strip()

        return cot.strip(), answer_part.strip()

    # --- Non-thinking models ---
    # Try JSON extraction from the full output
    json_answer = _extract_json_answer(raw_output)
    if json_answer is not None:
        # Everything before the JSON object is the CoT
        m = re.search(r'\{\s*"final_answer"\s*:', raw_output)
        cot = raw_output[:m.start()].strip() if m else ""
        return cot, json_answer.strip()

    # Fallback: regex pattern (last-line extraction)
    if cot_pattern is not None:
        m = re.search(cot_pattern, raw_output, flags=re.DOTALL)
        if m:
            groups = m.groupdict()
            return groups.get("cot", "").strip(), groups.get("answer", "").strip()

    # Nothing matched — return raw as answer
    return "", raw_output.strip()
