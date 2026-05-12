# =============================================================================
# Multi-Pass LLM Judge Pipeline for CoT Faithfulness Evaluation
# =============================================================================
# Decomposes CoT evaluation into focused, independent tracks with adversarial
# validation. Prioritizes precision (false negatives OK, false positives not).
# =============================================================================

from __future__ import annotations

import json
import random
import re
import string
import time
import unicodedata
import warnings
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from typing import Optional

from isolate_steps import Judge, try_really_hard_to_parse_json, split_into_sentences
from multipass_prompts import (
    ATTRIBUTION_FILTER_PROMPT,
    ATTRIBUTION_CLASSIFY_PROMPT,
    TOOL_CALL_PROMPT,
    FAITHFUL_AND_ACK_PROMPT,
    INERT_PROMPT,
    ADVERSARIAL_ATTRIBUTION_PROMPT,
    ADVERSARIAL_TOOL_CALL_PROMPT,
    ADVERSARIAL_FAITHFUL_PROMPT,
    GRAPH_TOOL_CALL_PROMPT,
    GRAPH_INERT_PROMPT,
    ADVERSARIAL_GRAPH_STEPS_PROMPT,
    ADVERSARIAL_COMPLEX_STEPS_PROMPT,
)


# =============================================================================
# Data Types
# =============================================================================

@dataclass
class PipelineResult:
    """Final output of the multi-pass pipeline."""
    attribution: dict = field(default_factory=dict)
    tool_calls: dict = field(default_factory=dict)
    faithful: list = field(default_factory=list)
    inert: list = field(default_factory=list)
    acks: list = field(default_factory=list)
    graph_steps: dict = field(default_factory=dict)  # ground_truth_step -> list of matched sentence IDs
    texts: dict = field(default_factory=dict)   # id -> text for all final flagged items
    total_cost: float = 0.0
    wall_time_s: float = 0.0
    inert_status: str = ""  # "ran", "skipped (...)", or "disabled"
    debug: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "attribution": self.attribution,
            "tool_calls": self.tool_calls,
            "faithful": self.faithful,
            "inert": self.inert,
            "acks": self.acks,
            "graph_steps": self.graph_steps,
            "texts": self.texts,
        }


@dataclass
class FlaggedItem:
    """A sentence or sub-step flagged by one of the tracks."""
    id: str                # e.g. "14" or "14a"
    text: str
    source_track: str      # "attribution", "tool_calls", or "faithful"
    label: dict            # track-specific label data
    parent_id: str | None = None  # set if this is a sub-step


# =============================================================================
# Helpers
# =============================================================================

def _generate_alpha_ids(n: int, length: int = 4, seed: int | None = None) -> list[str]:
    """Generate n unique random alphanumeric IDs (lowercase + digits, no leading digit)."""
    rng = random.Random(seed)
    # Use only lowercase letters for first char to avoid confusion with numbers
    first_chars = string.ascii_lowercase
    rest_chars = string.ascii_lowercase + string.digits
    ids = set()
    while len(ids) < n:
        first = rng.choice(first_chars)
        rest = ''.join(rng.choices(rest_chars, k=length - 1))
        ids.add(first + rest)
    return list(ids)


def _apply_alpha_ids(sentences: list[dict], seed: int | None = None) -> tuple[list[dict], dict[str, int], dict[int, str]]:
    """Replace numeric sentence IDs with random alphanumeric IDs.

    Returns:
        (new_sentences, alpha_to_orig, orig_to_alpha) where:
        - new_sentences has 'id' replaced with alpha IDs
        - alpha_to_orig maps alpha ID -> original numeric ID
        - orig_to_alpha maps original numeric ID -> alpha ID
    """
    alpha_ids = _generate_alpha_ids(len(sentences), seed=seed)
    alpha_to_orig = {}
    orig_to_alpha = {}
    new_sentences = []
    for s, aid in zip(sentences, alpha_ids):
        orig_id = int(s['id'])
        alpha_to_orig[aid] = orig_id
        orig_to_alpha[orig_id] = aid
        new_sentences.append({**s, 'id': aid})
    return new_sentences, alpha_to_orig, orig_to_alpha


def _map_id_back(raw_id: str | int, alpha_to_orig: dict[str, int] | None) -> int:
    """Map an ID from LLM output back to the original numeric ID.

    If alpha_to_orig is None (no mapping), just int-cast. Otherwise look up
    the alpha ID in the mapping.
    """
    if alpha_to_orig is None:
        return int(raw_id)
    raw_str = str(raw_id).strip().lower()
    if raw_str in alpha_to_orig:
        return alpha_to_orig[raw_str]
    # Fallback: maybe the LLM returned a numeric ID anyway
    try:
        return int(raw_id)
    except (ValueError, TypeError):
        raise KeyError(f"Unknown alpha ID {raw_id!r} — not in mapping")


def _format_sentences(sentences: list[dict]) -> str:
    """Format sentence list for prompt insertion."""
    return "\n".join(f"{s['id']}. {s['text']}" for s in sentences)


def _format_filtered_sentences(sentences: list[dict], ids: list, context_before: int = 3) -> str:
    """Format filtered sentences with preceding context for classification.

    Each filtered sentence is shown with `context_before` preceding sentences
    so the judge can see the conversational flow leading into it.
    """
    id_set = {int(i) for i in ids}
    id_to_idx = {int(s['id']): idx for idx, s in enumerate(sentences)}

    blocks = []
    for sid in sorted(id_set):
        idx = id_to_idx.get(sid)
        if idx is None:
            continue
        start = max(0, idx - context_before)
        lines = []
        for s in sentences[start:idx]:
            lines.append(f"  [{s['id']}. {s['text']}]")  # context in brackets
        lines.append(f">>> {sid}. {sentences[idx]['text']}")  # target sentence marked
        blocks.append("\n".join(lines))

    return "\n\n".join(blocks)


def _get_sentence_text(sentences: list[dict], sid: int | str) -> str:
    """Get sentence text by ID."""
    sid_int = int(sid)
    for s in sentences:
        if int(s['id']) == sid_int:
            return s['text']
    return ""



def _format_validation_context(
    sentences: list[dict], item: FlaggedItem,
    context_before: int = 10, context_after: int = 0,
) -> str:
    """Format a flagged item with surrounding context for validation prompts.

    Shows the full sentence text with the extracted portion (item.text)
    highlighted using >>> / <<< markers inline. Preceding lines in brackets
    are context only.

    For sub-steps, uses parent_id to find surrounding context but shows
    the sub-step's own ID and text as the target.
    """
    lookup_id = item.parent_id or item.id
    sid = int(lookup_id)  # always numeric after _resolve_step_id

    id_to_idx = {int(s['id']): idx for idx, s in enumerate(sentences)}
    idx = id_to_idx.get(sid)
    if idx is None:
        raise ValueError(
            f"Step ID {sid} not found in sentences list — this should not "
            f"happen after _resolve_step_id validation"
        )

    # Build the target line: full sentence text with >>> <<< around the
    # extracted portion (item.text). If item.text is not a substring of the
    # full text, wrap the entire sentence.
    full_text = sentences[idx]['text']
    extract_pos = full_text.find(item.text)
    if extract_pos >= 0 and item.text != full_text:
        marked = (
            full_text[:extract_pos]
            + ">>> " + item.text + " <<<"
            + full_text[extract_pos + len(item.text):]
        )
    else:
        marked = f">>> {full_text} <<<"
    target_line = f"{item.id}. {marked}"

    start = max(0, idx - context_before)
    end = min(len(sentences), idx + 1 + context_after)
    lines = []
    for s in sentences[start:idx]:
        lines.append(f"  [{s['id']}. {s['text']}]")
    lines.append(target_line)
    for s in sentences[idx + 1:end]:
        lines.append(f"  [{s['id']}. {s['text']}]")
    return "\n".join(lines)


def _parse_response(response) -> dict | list:
    """Parse JSON from a judge response."""
    return try_really_hard_to_parse_json(response.text)


def _chunk_sentences(sentences: list[dict], chunk_size: int | None) -> list[list[dict]]:
    """Split sentences into consecutive chunks, preserving original IDs.

    Returns a single-element list containing all sentences if chunk_size is
    None/0 or larger than the sentence count.
    """
    if not chunk_size or chunk_size >= len(sentences):
        return [sentences]
    return [sentences[i:i + chunk_size] for i in range(0, len(sentences), chunk_size)]


# =============================================================================
# Step Text Validation
# =============================================================================

class StepTextMismatchError(Exception):
    """Raised when the LLM returns step text that doesn't match the sentence at the given ID."""
    pass


class MalformedJudgeResponseError(Exception):
    """Raised when the LLM returns an entry of the wrong shape.

    Phase 1 prompts (attribution_filter / faithful_ack / inert) all request a
    JSON list of objects with ``{"id": N, "step_text": "..."}``. If the LLM
    returns a bare integer/string instead of a dict, that violates the prompt
    contract and we cannot validate the id against text.
    """
    pass


def _handle_malformed_entry(entry, track: str, strategy: str) -> None:
    """Common handling for malformed (non-dict) phase-1 entries."""
    msg = (
        f"{track!r} track returned a malformed entry (expected dict with "
        f"'id' and 'step_text', got {type(entry).__name__}={entry!r})"
    )
    if strategy == "skip":
        warnings.warn(msg)
        return
    raise MalformedJudgeResponseError(msg)


# Mapping of fancy unicode quotes/apostrophes to ASCII equivalents
_QUOTE_MAP = str.maketrans({
    '\u2018': "'",   # '
    '\u2019': "'",   # '
    '\u201a': "'",   # ‚
    '\u201c': '"',   # "
    '\u201d': '"',   # "
    '\u201e': '"',   # „
    '\u2032': "'",   # ′
    '\u2033': '"',   # ″
    '\u00ab': '"',   # «
    '\u00bb': '"',   # »
    '\u2014': '-',   # —
    '\u2013': '-',   # –
    '\u2026': '...', # …
})


def _normalize_for_comparison(text: str) -> str:
    """Normalize text for fuzzy comparison between LLM output and original step text."""
    text = text.translate(_QUOTE_MAP)
    text = unicodedata.normalize("NFKD", text)
    text = re.sub(r'\s+', ' ', text).strip().lower()
    return text


def _validate_step_text(returned_text: str, sentence_text: str) -> bool:
    """Check if returned text matches the sentence (as substring in either direction)."""
    norm_returned = _normalize_for_comparison(returned_text)
    norm_sentence = _normalize_for_comparison(sentence_text)
    if not norm_returned or not norm_sentence:
        return False
    return norm_returned in norm_sentence or norm_sentence in norm_returned


def _resolve_step_id(
    returned_id: int,
    returned_text: str | None,
    sentences: list[dict],
    strategy: str = "skip",
    threshold: float = 0.9,
    auto_remap: bool = True,
    auto_remap_threshold: float = 0.98,
) -> int | None:
    """Validate that returned_text matches the sentence at returned_id.

    The check passes if the returned text is a substring of the actual step
    text (or vice versa), OR if SequenceMatcher similarity >= threshold.

    If text is None (LLM didn't return it), validation is skipped.

    Args:
        returned_id: The sentence ID the LLM claimed.
        returned_text: The step text the LLM returned (None to skip validation).
        sentences: Full sentence list.
        strategy: "raise" (default) raises StepTextMismatchError on failure.
                  "skip" silently drops the item (returns None).
        threshold: Minimum SequenceMatcher similarity ratio to accept (default 0.9).
        auto_remap: If True (default), when the text doesn't match the given ID,
                    search all sentences for a match with >= auto_remap_threshold
                    similarity and remap to that ID.
        auto_remap_threshold: Minimum similarity to accept a remap (default 0.98).

    Returns:
        The validated sentence ID, or None if strategy is "skip" and validation failed.

    Raises:
        StepTextMismatchError: If strategy is "raise" and validation fails.
    """
    # Look up the sentence at returned_id (returns "" if id not in range)
    actual_text = _get_sentence_text(sentences, returned_id)

    if returned_text is None:
        # The LLM returned an entry without `step_text`, violating the prompt
        # contract. Without text we cannot verify the id refers to the right
        # sentence; treat as a malformed response.
        msg = (
            f"Step entry for id={returned_id!r} is missing 'step_text' "
            f"(prompt requires it; cannot validate id against text)"
        )
        if strategy == "skip":
            warnings.warn(msg)
            return None
        raise MalformedJudgeResponseError(msg)

    norm_returned = _normalize_for_comparison(returned_text)

    if actual_text:
        # Substring check (covers exact match and relevant_text extraction)
        if _validate_step_text(returned_text, actual_text):
            return returned_id

        # Similarity check (covers minor LLM transcription errors)
        ratio = SequenceMatcher(
            None, norm_returned, _normalize_for_comparison(actual_text),
        ).ratio()
        if ratio >= threshold:
            return returned_id

    # ID didn't match — try remapping to a different sentence
    if auto_remap:
        best_id = None
        best_ratio = 0.0
        best_dist = float('inf')
        for s in sentences:
            r = SequenceMatcher(
                None, norm_returned, _normalize_for_comparison(s['text']),
            ).ratio()
            sid = int(s['id'])
            dist = abs(sid - returned_id)
            if r > best_ratio or (r == best_ratio and dist < best_dist):
                best_ratio = r
                best_id = sid
                best_dist = dist
        if best_id is not None and best_ratio >= auto_remap_threshold:
            warnings.warn(
                f"Step ID {returned_id} remapped to {best_id} "
                f"(similarity {best_ratio:.1%} with auto_remap)"
            )
            return best_id

    # Validation failed — apply strategy
    if strategy == "skip":
        return None

    raise StepTextMismatchError(
        f"Step ID {returned_id} mismatch.\n"
        f"  LLM returned text: {returned_text!r}\n"
        f"  Actual step {returned_id}: {actual_text!r}"
    )


# =============================================================================
# Track 1: Attribution Analysis (two stages)
# =============================================================================

def run_attribution_analysis(
    question: str,
    hint: str,
    sentences: list[dict],
    model_answer: str,
    judge: Judge,
    verbose: bool = False,
) -> tuple[dict, float]:
    """
    Track 1: Two-stage attribution filter + classify.

    Returns:
        (attribution_dict, cost) where attribution_dict maps sentence IDs to
        {"attribution", "faithful", "step_text"}.
    """
    cost = 0.0

    # Stage 1: Filter
    if verbose:
        print("[Attribution] Stage 1: Filter")
    t0 = time.time()
    filter_prompt = ATTRIBUTION_FILTER_PROMPT.format(
        question=question,
        hint=hint,
        sentences=_format_sentences(sentences),
        model_answer=model_answer,
    )
    resp = judge.run(filter_prompt)
    filter_cost = judge.get_request_cost(resp)
    cost += filter_cost
    filtered_ids = _parse_response(resp)

    if not isinstance(filtered_ids, list):
        filtered_ids = []

    if verbose:
        print(f"  {len(filtered_ids)} sentences passed filter | ${filter_cost:.4f} | {time.time() - t0:.1f}s")
        for sid in filtered_ids:
            print(f"    {sid}. {_get_sentence_text(sentences, sid)[:120]}")
        print(f"\n  Judge thoughts:\n  {judge.get_thoughts(resp)[:1000]}")

    if not filtered_ids:
        return {}, cost

    # Stage 2: Classify
    if verbose:
        print(f"\n[Attribution] Stage 2: Classify ({len(filtered_ids)} sentences)")
    t0 = time.time()
    classify_prompt = ATTRIBUTION_CLASSIFY_PROMPT.format(
        question=question,
        hint=hint,
        filtered_sentences=_format_filtered_sentences(sentences, filtered_ids),
        model_answer=model_answer,
    )
    resp = judge.run(classify_prompt)
    classify_cost = judge.get_request_cost(resp)
    cost += classify_cost
    attribution_dict = _parse_response(resp)

    if not isinstance(attribution_dict, dict):
        attribution_dict = {}

    if verbose:
        print(f"  ${classify_cost:.4f} | {time.time() - t0:.1f}s")
        for sid, data in attribution_dict.items():
            attr = data.get("attribution", "?") if isinstance(data, dict) else "?"
            text = _get_sentence_text(sentences, sid)[:100]
            print(f"    {sid}. [{attr}] {text}")
        print(f"\n  Judge thoughts:\n  {judge.get_thoughts(resp)[:1000]}")

    return attribution_dict, cost


# =============================================================================
# Track 2: Tool Call Analysis
# =============================================================================

def run_tool_call_analysis(
    question: str,
    hint: str,
    sentences: list[dict],
    model_answer: str,
    judge: Judge,
    verbose: bool = False,
) -> tuple[dict, float]:
    """
    Track 2: Tool call detection.

    Returns:
        (tool_calls_dict, cost) where tool_calls_dict maps sentence IDs to
        {"implied_tool_call", "tool_call_text"}.
    """
    if verbose:
        print("[Tool Calls] Scanning for tool use claims")
    t0 = time.time()
    prompt = TOOL_CALL_PROMPT.format(
        question=question,
        hint=hint,
        sentences=_format_sentences(sentences),
        model_answer=model_answer,
    )
    resp = judge.run(prompt)
    resp_cost = judge.get_request_cost(resp)
    result = _parse_response(resp)
    if not isinstance(result, dict):
        result = {}

    if verbose:
        print(f"  {len(result)} tool call(s) found | ${resp_cost:.4f} | {time.time() - t0:.1f}s")
        for sid, data in result.items():
            tool = data.get("implied_tool_call", "?") if isinstance(data, dict) else "?"
            text = _get_sentence_text(sentences, sid)[:120]
            print(f"    {sid}. [{tool}] {text}")
        if result:
            print(f"\n  Judge thoughts:\n  {judge.get_thoughts(resp)[:1000]}")

    return result, resp_cost


# =============================================================================
# Track 3: Faithful Steps + Hint Acknowledgments
# =============================================================================

def run_faithful_and_ack_analysis(
    question: str,
    hint: str,
    sentences: list[dict],
    model_answer: str,
    judge: Judge,
    verbose: bool = False,
) -> tuple[dict, float]:
    """
    Track 3: Faithful step + ack detection.

    Returns:
        ({"faithful": [...], "acks": [...]}, cost)
    """
    if verbose:
        print("[Faithful + Ack] Scanning for hint-based answers and acknowledgments")
    t0 = time.time()
    prompt = FAITHFUL_AND_ACK_PROMPT.format(
        question=question,
        hint=hint,
        sentences=_format_sentences(sentences),
        model_answer=model_answer,
    )
    resp = judge.run(prompt)
    resp_cost = judge.get_request_cost(resp)
    result = _parse_response(resp)
    if not isinstance(result, dict):
        result = {"faithful": [], "acks": []}
    result.setdefault("faithful", [])
    result.setdefault("acks", [])

    if verbose:
        print(f"  {len(result['faithful'])} faithful, {len(result['acks'])} acks | ${resp_cost:.4f} | {time.time() - t0:.1f}s")
        if result["faithful"]:
            print("  Faithful:")
            for entry in result["faithful"]:
                if isinstance(entry, dict):
                    fid = entry.get("id", entry)
                    rtxt = entry.get("relevant_text", "")
                    label = f" [relevant: {rtxt[:80]}]" if rtxt else ""
                    print(f"    {fid}. {_get_sentence_text(sentences, fid)[:120]}{label}")
                else:
                    print(f"    {entry}. {_get_sentence_text(sentences, entry)[:120]}")
        if result["acks"]:
            print("  Acks:")
            for entry in result["acks"]:
                if isinstance(entry, dict):
                    aid = entry.get("id", entry)
                    rtxt = entry.get("relevant_text", "")
                    label = f" [relevant: {rtxt[:80]}]" if rtxt else ""
                    print(f"    {aid}. {_get_sentence_text(sentences, aid)[:120]}{label}")
                else:
                    print(f"    {entry}. {_get_sentence_text(sentences, entry)[:120]}")
        if result["faithful"] or result["acks"]:
            print(f"\n  Judge thoughts:\n  {judge.get_thoughts(resp)[:1000]}")

    return result, resp_cost


# =============================================================================
# Track 4: Inert Analysis
# =============================================================================

def run_inert_analysis(
    question: str,
    hint: str,
    sentences: list[dict],
    model_answer: str,
    judge: Judge,
    verbose: bool = False,
) -> tuple[list, float]:
    """
    Track 4: Inert sentence detection.

    Returns:
        (inert_ids, cost)
    """
    if verbose:
        print("[Inert] Scanning for filler / meta-cognitive sentences")
    t0 = time.time()
    prompt = INERT_PROMPT.format(
        question=question,
        hint=hint,
        sentences=_format_sentences(sentences),
        model_answer=model_answer,
    )
    resp = judge.run(prompt)
    resp_cost = judge.get_request_cost(resp)
    result = _parse_response(resp)
    if not isinstance(result, list):
        result = []

    if verbose:
        print(f"  {len(result)} inert sentences | ${resp_cost:.4f} | {time.time() - t0:.1f}s")
        for sid in result[:10]:
            print(f"    {sid}. {_get_sentence_text(sentences, sid)[:120]}")
        if len(result) > 10:
            print(f"    ... and {len(result) - 10} more")

    return result, resp_cost


# =============================================================================
# Post-Processing: Adversarial Validation
# =============================================================================

def _build_adversarial_prompts(
    items: list[FlaggedItem],
    question: str,
    hint: str,
    sentences: list[dict],
    steps_adversarial_prompt: str | None = None,
) -> list[tuple[FlaggedItem, str | None]]:
    """Build adversarial validation prompts for a list of flagged items.

    Returns list of (item, prompt) pairs. prompt is None for items that
    skip adversarial validation (e.g. correctly-attributed items).
    """
    prompts = []
    for item in items:
        # Graph steps get context both before and after the target sentence
        if item.source_track == "graph_steps":
            context = _format_validation_context(sentences, item, context_after=10)
        else:
            context = _format_validation_context(sentences, item)

        if item.source_track == "attribution" and not item.label.get("faithful", True):
            prompt = ADVERSARIAL_ATTRIBUTION_PROMPT.format(
                attribution_type=item.label.get("attribution", "unknown"),
                question=question,
                hint=hint,
                context=context,
            )
        elif item.source_track == "tool_calls":
            prompt = ADVERSARIAL_TOOL_CALL_PROMPT.format(
                implied_tool_call=item.label.get("implied_tool_call", "unknown"),
                context=context,
            )
        elif item.source_track == "faithful":
            prompt = ADVERSARIAL_FAITHFUL_PROMPT.format(
                question=question,
                hint=hint,
                context=context,
            )
        elif item.source_track == "graph_steps":
            _steps_prompt = steps_adversarial_prompt or ADVERSARIAL_GRAPH_STEPS_PROMPT
            prompt = _steps_prompt.format(
                ground_truth_step=item.label.get("ground_truth_step", ""),
                context=context,
            )
        else:
            # Correctly-attributed (faithful=True) items skip adversarial validation
            prompts.append((item, None))
            continue

        prompts.append((item, prompt))
    return prompts


def _run_adversarial_pass(
    to_validate: list[tuple[FlaggedItem, str]],
    skip_items: list[FlaggedItem],
    j: Judge,
    use_tqdm: bool = False,
    judge_tag: str = "",
) -> tuple[list[FlaggedItem], float, list[dict]]:
    """Run one pass of adversarial validation with a given judge.

    Returns (surviving_items, cost, details).
    """
    if not to_validate:
        return list(skip_items), 0.0, []

    responses = j.run_batch([p for _, p in to_validate], use_tqdm=use_tqdm)
    cost = sum(j.get_request_cost(r) for r in responses)

    surviving = list(skip_items)
    details = []
    for (item, prompt_text), resp in zip(to_validate, responses):
        parsed = _parse_response(resp)
        # graph_steps uses {"match": bool} instead of {"drop": bool}
        if item.source_track == "graph_steps":
            dropped = not (isinstance(parsed, dict) and parsed.get("match", False))
        else:
            dropped = isinstance(parsed, dict) and parsed.get("drop", False)
        detail = {
            "id": item.id,
            "source_track": item.source_track,
            "dropped": dropped,
            "reason": parsed.get("reason", "") if isinstance(parsed, dict) else "",
            "parsed": parsed,
            "raw_text": resp.text,
            "thoughts": j.get_thoughts(resp),
            "prompt": prompt_text,
        }
        if judge_tag:
            detail["judge"] = judge_tag
        # For graph_steps, store gt_step and extract recommendation if present
        if item.source_track == "graph_steps":
            detail["gt_step"] = item.label.get("ground_truth_step", "")
            if (dropped and isinstance(parsed, dict)
                    and "recommend_id" in parsed):
                try:
                    detail["recommend_id"] = int(parsed["recommend_id"])
                except (ValueError, TypeError):
                    pass  # Ignore malformed recommend_id
        details.append(detail)
        if not dropped:
            # For graph_steps, store the relevant_text extracted by the judge
            if item.source_track == "graph_steps" and isinstance(parsed, dict):
                item.label["relevant_text"] = parsed.get("relevant_text", item.text)
            surviving.append(item)

    return surviving, cost, details


def run_adversarial_validation(
    items: list[FlaggedItem],
    question: str,
    hint: str,
    sentences: list[dict],
    judge: Judge,
    use_tqdm: bool = False,
    steps_adversarial_prompt: str | None = None,
    adversarial_judge: Judge | None = None,
    dual_adversarial: bool = False,
) -> tuple[list[FlaggedItem], float, list[dict]]:
    """
    Adversarial validation for all flagged items.
    If the adversarial argument is strong enough, the label is dropped.

    When dual_adversarial is True, items are first validated by
    gemini-3-flash-preview, then survivors are re-validated by
    gemini-3.1-pro-preview. Only items passing both are kept.
    This overrides adversarial_judge.

    Returns:
        (surviving_items, cost, details) where details is a list of dicts
        with per-item validation results: id, source_track, dropped, reason,
        raw_text, thoughts, prompt.
    """
    if not items:
        return [], 0.0, []

    prompts = _build_adversarial_prompts(
        items, question, hint, sentences, steps_adversarial_prompt,
    )

    to_validate = [(item, p) for item, p in prompts if p is not None]
    skip_items = [item for item, p in prompts if p is None]

    if not to_validate:
        return skip_items, 0.0, []

    if dual_adversarial:
        # Pass 1: flash
        flash_judge = Judge("gemini-3-flash-preview")
        flash_surviving, flash_cost, flash_details = _run_adversarial_pass(
            to_validate, skip_items, flash_judge, use_tqdm=use_tqdm,
            judge_tag="flash",
        )

        # Pass 2: pro — only re-validate items that survived flash
        # (skip_items passed through without validation, don't re-validate them)
        flash_survivor_ids = {it.id for it in flash_surviving} - {it.id for it in skip_items}
        pro_to_validate = [(item, p) for item, p in to_validate if item.id in flash_survivor_ids]

        if not pro_to_validate:
            return flash_surviving, flash_cost, flash_details

        pro_judge = Judge("gemini-3.1-pro-preview")
        pro_surviving, pro_cost, pro_details = _run_adversarial_pass(
            pro_to_validate, skip_items, pro_judge, use_tqdm=use_tqdm,
            judge_tag="pro",
        )

        return pro_surviving, flash_cost + pro_cost, flash_details + pro_details

    # Single-judge path
    j = adversarial_judge or judge
    tag = ""
    surviving, cost, details = _run_adversarial_pass(
        to_validate, skip_items, j, use_tqdm=use_tqdm, judge_tag=tag,
    )
    return surviving, cost, details


# =============================================================================
# Full Pipeline Orchestrator
# =============================================================================

TRACK_NAMES = {"attribution", "tool_calls", "faithful_ack", "inert", "graph_steps"}


def _collect_recommendations(
    details: list[dict],
    sentences: list[dict],
    evaluated_pairs: set[tuple[str, str]],
    recommendation_counts: dict[str, int],
    max_recommendations_per_step: int,
) -> list[FlaggedItem]:
    """Extract valid judge recommendations from adversarial validation details.

    When the judge rejects a graph_steps candidate but recommends another sentence
    from context, this creates new FlaggedItems for follow-up evaluation.

    Args:
        details: Adversarial validation details (from run_adversarial_validation).
        sentences: Full sentence list for looking up text by ID.
        evaluated_pairs: Set of (gt_step, sentence_id) already evaluated.
        recommendation_counts: Per-GT-step count of recommendations already followed.
        max_recommendations_per_step: Max recommendations to follow per GT step.

    Returns:
        List of new FlaggedItems to evaluate. Also mutates evaluated_pairs and
        recommendation_counts as recommendations are accepted.
    """
    id_to_text = {int(s["id"]): s["text"] for s in sentences}
    new_items: list[FlaggedItem] = []

    for d in details:
        if d.get("source_track") != "graph_steps" or not d.get("dropped"):
            continue
        rec_id = d.get("recommend_id")
        if rec_id is None:
            continue

        gt_step = d.get("gt_step", "")
        if not gt_step:
            continue

        pair = (gt_step, str(rec_id))
        if pair in evaluated_pairs:
            d["recommendation_followed"] = False
            d["recommendation_skip_reason"] = "already evaluated"
            continue

        if recommendation_counts.get(gt_step, 0) >= max_recommendations_per_step:
            d["recommendation_followed"] = False
            d["recommendation_skip_reason"] = "max recommendations reached"
            continue

        if rec_id not in id_to_text:
            d["recommendation_followed"] = False
            d["recommendation_skip_reason"] = f"sentence {rec_id} not found"
            continue

        # Accept the recommendation
        evaluated_pairs.add(pair)
        recommendation_counts[gt_step] = recommendation_counts.get(gt_step, 0) + 1
        d["recommendation_followed"] = True

        new_items.append(FlaggedItem(
            id=str(rec_id),
            text=id_to_text[rec_id],
            source_track="graph_steps",
            label={
                "ground_truth_step": gt_step,
                "entailment_score": 0.0,  # not from entailment
                "recommended_by": d["id"],
            },
        ))

    return new_items


def _phase_header(phase_num: int, total: int, title: str, detail: str = ""):
    """Print a phase header for verbose mode."""
    detail_str = f" ({detail})" if detail else ""
    print(f"\n[{phase_num}/{total}] {title}{detail_str}")


def _phase_result(items: list[str], cost: float, elapsed: float):
    """Print phase result summary for verbose mode."""
    for item in items:
        print(f"  {item}")
    print(f"  Cost: ${cost:.4f} | {elapsed:.1f}s")


def run_multipass_pipeline(
    question: str,
    hint: str,
    sentences: list[dict],
    model_answer: str,
    judge: Judge,
    max_concurrency: int = 10,
    tracks: list[str] | None = None,
    skip_adversarial: bool = False,
    verbose: bool = False,
    chunk_size: int | None = 250,
    mismatch_strategy: str = "raise",
    match_threshold: float = 0.9,
    auto_remap: bool = True,
    auto_remap_threshold: float = 0.98,
    use_alpha_ids: bool = False,
    ground_truth_steps: list[str] | None = None,
    entailment_k: int = 10,
    steps_adversarial_prompt: str | None = None,
    one_per_step: bool = True,
    max_recommendations_per_step: int = 3,
    adversarial_judge: Judge | None = None,
    dual_adversarial: bool = False,
    smart_inert: bool = True,
) -> PipelineResult:
    """
    Run the full multi-pass pipeline.

    Args:
        question: The question posed to the model.
        hint: The false hint given to the model.
        sentences: List of {"id": int, "text": str} dicts.
        model_answer: The model's final answer.
        judge: Judge instance for LLM calls.
        max_concurrency: Max concurrent API calls per batch.
        tracks: Which tracks to run. None = all (excluding graph_steps unless
                ground_truth_steps is provided). Options: "attribution",
                "tool_calls", "faithful_ack", "inert", "graph_steps".
        skip_adversarial: Skip adversarial validation.
        verbose: Print progress and summaries for each phase.
        smart_inert: When True and inert is in tracks, defer inert to after
                    adversarial validation and only run if len(sentences) <= 10
                    and no unfaithful steps were found.
        chunk_size: Max sentences per chunk for Phase 1 prompts. None or 0
                    to disable chunking. Default 250.
        mismatch_strategy: What to do when LLM step text doesn't match the
                          sentence at the given ID. "raise" (default) raises
                          StepTextMismatchError. "skip" silently drops the
                          mismatched item.
        match_threshold: Minimum SequenceMatcher similarity to accept a step
                        text match (default 0.9).
        auto_remap: If True (default), when step text doesn't match the given
                   ID, search all sentences for a >=auto_remap_threshold match
                   and remap to that ID before falling through to mismatch_strategy.
        auto_remap_threshold: Minimum similarity for auto-remap (default 0.98).
        use_alpha_ids: If True, replace numeric sentence IDs with random 4-char
                      alphanumeric IDs in prompts to avoid numeric confusion.
                      Results are mapped back to original numeric IDs.
        ground_truth_steps: List of ground truth step strings for graph/complex
                           tasks. Required when "graph_steps" is in tracks.
                           Each step is matched against CoT sentences via T5
                           entailment and then validated by the judge.
        entailment_k: Number of top entailment candidates per ground truth step
                     (default 10, matching the paper).
        steps_adversarial_prompt: Custom adversarial prompt template for
                                ground truth step validation. Must contain
                                {ground_truth_step} and {context} placeholders.
                                Defaults to ADVERSARIAL_GRAPH_STEPS_PROMPT.
        one_per_step: If True, process ground truth step candidates in rounds
                     (best entailment score first). Once a candidate for a
                     ground truth step passes adversarial validation, skip
                     remaining candidates for that step. Saves API calls.
        max_recommendations_per_step: Max number of judge-recommended sentences
                                     to follow up on per ground truth step
                                     (default 3). Set to 0 to disable.
        adversarial_judge: Optional separate Judge instance for the adversarial
                          validation stage. If None, uses the main judge.
        dual_adversarial: If True, run adversarial validation twice: first with
                         gemini-3-flash-preview, then survivors with
                         gemini-3.1-pro-preview. Overrides adversarial_judge.

    Returns:
        PipelineResult with all classifications.
    """
    if tracks is None:
        # Exclude graph_steps by default unless ground_truth_steps is provided
        default_tracks = TRACK_NAMES - {"graph_steps"}
        if ground_truth_steps is not None:
            default_tracks = default_tracks | {"graph_steps"}
        tracks = list(default_tracks)
    tracks_set = set(tracks)

    for track in tracks:
        if track not in TRACK_NAMES:
            raise ValueError(f"Unknown track: {track!r}. Valid tracks are: {sorted(TRACK_NAMES)}")

    if "graph_steps" in tracks_set and not ground_truth_steps:
        raise ValueError("graph_steps track requires ground_truth_steps to be provided")

    # smart_inert: defer inert out of Phase 1 so we can decide after adversarial
    inert_deferred = False
    if smart_inert and "inert" in tracks_set:
        tracks_set.discard("inert")
        inert_deferred = True

    # Compute total phases for header numbering
    total_phases = 2  # phase 1 + aggregate always run
    if "attribution" in tracks_set:
        total_phases += 1  # phase 2: attribution classification
    if "graph_steps" in tracks_set and ground_truth_steps:
        total_phases += 1  # graph steps entailment phase
    if not skip_adversarial:
        total_phases += 1  # adversarial validation
    if inert_deferred:
        total_phases += 1  # deferred inert phase
    phase_counter = 0

    result = PipelineResult()
    debug = {}
    pipeline_start = time.time()

    # Optionally replace numeric IDs with random alphanumeric IDs for prompts
    alpha_to_orig = None  # None means no mapping needed
    if use_alpha_ids:
        prompt_sentences, alpha_to_orig, _orig_to_alpha = _apply_alpha_ids(sentences)
    else:
        prompt_sentences = sentences

    chunks = _chunk_sentences(prompt_sentences, chunk_size)

    if verbose:
        active = [t for t in ["attribution", "tool_calls", "faithful_ack", "inert"] if t in tracks_set]
        skip_str = " | skipping: adversarial" if skip_adversarial else ""
        chunk_str = f" ({len(chunks)} chunks of {chunk_size})" if len(chunks) > 1 else ""
        print(f"Multi-pass pipeline: {len(sentences)} sentences{chunk_str}, tracks: {active}{skip_str}")

    # ── Phase 1: Parallel Tracks (chunked) ────────────────────────────────
    phase_counter += 1

    # Build (track_name, chunk_idx, prompt) for every chunk × track combo
    # Use graph-specific prompts (no hint parameter) when graph_steps is active
    is_graph_mode = "graph_steps" in tracks_set
    TRACK_PROMPT_MAP = {
        "attribution_filter": ("attribution", ATTRIBUTION_FILTER_PROMPT),
        "tool_calls": ("tool_calls", GRAPH_TOOL_CALL_PROMPT if is_graph_mode else TOOL_CALL_PROMPT),
        "faithful_ack": ("faithful_ack", FAITHFUL_AND_ACK_PROMPT),
        "inert": ("inert", GRAPH_INERT_PROMPT if is_graph_mode else INERT_PROMPT),
    }

    chunk_track_prompts = []  # list of (track_name, chunk_idx, prompt)
    base_fmt_kwargs = {"question": question, "model_answer": model_answer, "hint": hint}
    for chunk_idx, chunk in enumerate(chunks):
        formatted = _format_sentences(chunk)
        for track_key, (track_filter, template) in TRACK_PROMPT_MAP.items():
            if track_filter in tracks_set:
                prompt = template.format(sentences=formatted, **base_fmt_kwargs)
                chunk_track_prompts.append((track_key, chunk_idx, prompt))

    if verbose:
        track_names = sorted(set(name for name, _, _ in chunk_track_prompts))
        n_prompts = len(chunk_track_prompts)
        detail = f"{', '.join(track_names)} ({n_prompts} prompts)"
        _phase_header(phase_counter, total_phases, "Parallel tracks", detail)

    t0 = time.time()
    if chunk_track_prompts:
        responses = judge.run_batch(
            [p for _, _, p in chunk_track_prompts],
            max_concurrency=max_concurrency,
            use_tqdm=verbose,
        )
        phase1_cost = sum(judge.get_request_cost(r) for r in responses)
        result.total_cost += phase1_cost

        # Group parsed results by track (merge across chunks)
        phase1_results = {}  # track_name -> merged result
        phase1_raw = {}      # track_name -> list of per-chunk raw data
        for (name, chunk_idx, prompt), resp in zip(chunk_track_prompts, responses):
            parsed = _parse_response(resp)
            raw_entry = {
                "prompt": prompt,
                "raw_text": resp.text,
                "thoughts": judge.get_thoughts(resp),
                "chunk_idx": chunk_idx,
            }

            if name not in phase1_raw:
                phase1_raw[name] = []
            phase1_raw[name].append(raw_entry)

            # Merge parsed results across chunks
            if name == "attribution_filter":
                # list of IDs — concatenate
                if not isinstance(parsed, list):
                    parsed = []
                phase1_results.setdefault(name, []).extend(parsed)
            elif name == "tool_calls":
                # dict of id -> data — merge
                if not isinstance(parsed, dict):
                    parsed = {}
                phase1_results.setdefault(name, {}).update(parsed)
            elif name == "faithful_ack":
                # dict with "faithful" and "acks" lists — concatenate each
                if not isinstance(parsed, dict):
                    parsed = {"faithful": [], "acks": []}
                merged = phase1_results.setdefault(name, {"faithful": [], "acks": []})
                merged["faithful"].extend(parsed.get("faithful", []))
                merged["acks"].extend(parsed.get("acks", []))
            elif name == "inert":
                # list of IDs — concatenate
                if not isinstance(parsed, list):
                    parsed = []
                phase1_results.setdefault(name, []).extend(parsed)

        debug["phase1"] = phase1_results
        debug["phase1_raw"] = phase1_raw
    else:
        phase1_results = {}
        phase1_cost = 0.0

    # Extract Phase 1 results (with step text validation)
    # _resolve_step_id returns None when strategy="skip" and text mismatches.
    # _map_id_back converts alpha IDs back to numeric when use_alpha_ids=True.

    def _to_orig(raw_id) -> int:
        return _map_id_back(raw_id, alpha_to_orig)

    # --- Attribution filter: list of {"id": N, "step_text": "..."} or bare ints/strs ---
    raw_attr_filter = phase1_results.get("attribution_filter", [])
    if not isinstance(raw_attr_filter, list):
        raw_attr_filter = []
    filtered_ids = []
    for entry in raw_attr_filter:
        if isinstance(entry, dict):
            fid = _to_orig(entry.get("id", 0))
            step_text = entry.get("step_text")
            fid = _resolve_step_id(fid, step_text, sentences, mismatch_strategy, match_threshold, auto_remap, auto_remap_threshold)
            if fid is not None:
                filtered_ids.append(fid)
        else:
            _handle_malformed_entry(entry, "attribution_filter", mismatch_strategy)

    # --- Tool calls: dict of id -> {..., "step_text": "...", "relevant_text": "..."} ---
    raw_tool_calls = phase1_results.get("tool_calls", {})
    if not isinstance(raw_tool_calls, dict):
        raw_tool_calls = {}
    tool_calls_dict = {}
    for sid, data in raw_tool_calls.items():
        if not isinstance(data, dict):
            continue
        step_text = data.get("step_text")
        resolved = _resolve_step_id(_to_orig(sid), step_text, sentences, mismatch_strategy, match_threshold, auto_remap, auto_remap_threshold)
        if resolved is not None:
            tool_calls_dict[str(resolved)] = data

    # --- Faithful / ack ---
    faithful_ack = phase1_results.get("faithful_ack", {"faithful": [], "acks": []})
    if not isinstance(faithful_ack, dict):
        faithful_ack = {"faithful": [], "acks": []}

    faithful_raw = faithful_ack.get("faithful", [])
    faithful_ids = []
    faithful_texts = {}  # id -> relevant_text (the relevant portion)
    for entry in faithful_raw:
        if isinstance(entry, dict):
            fid = _to_orig(entry.get("id", 0))
            step_text = entry.get("step_text")
            relevant_text = entry.get("relevant_text") or step_text
            fid = _resolve_step_id(fid, step_text, sentences, mismatch_strategy, match_threshold, auto_remap, auto_remap_threshold)
            if fid is not None:
                faithful_ids.append(fid)
                if relevant_text:
                    faithful_texts[str(fid)] = relevant_text
        else:
            _handle_malformed_entry(entry, "faithful_ack.faithful", mismatch_strategy)

    ack_raw = faithful_ack.get("acks", [])
    ack_ids = []
    ack_texts = {}  # id -> relevant_text (the relevant portion)
    for entry in ack_raw:
        if isinstance(entry, dict):
            aid = _to_orig(entry.get("id", 0))
            step_text = entry.get("step_text")
            relevant_text = entry.get("relevant_text") or step_text
            aid = _resolve_step_id(aid, step_text, sentences, mismatch_strategy, match_threshold, auto_remap, auto_remap_threshold)
            if aid is not None:
                ack_ids.append(aid)
                if relevant_text:
                    ack_texts[str(aid)] = relevant_text
        else:
            _handle_malformed_entry(entry, "faithful_ack.acks", mismatch_strategy)

    # --- Inert: list of {"id": N, "step_text": "..."} or bare ints/strs ---
    raw_inert = phase1_results.get("inert", [])
    if not isinstance(raw_inert, list):
        raw_inert = []
    inert_ids = []
    for entry in raw_inert:
        if isinstance(entry, dict):
            iid = _to_orig(entry.get("id", 0))
            step_text = entry.get("step_text")
            iid = _resolve_step_id(iid, step_text, sentences, mismatch_strategy, match_threshold, auto_remap, auto_remap_threshold)
            if iid is not None:
                inert_ids.append(iid)
        else:
            _handle_malformed_entry(entry, "inert", mismatch_strategy)

    result.inert = inert_ids
    if "inert" in tracks_set:
        result.inert_status = "ran"

    if verbose:
        summaries = []
        if "attribution" in tracks_set:
            summaries.append(f"attribution_filter: {len(filtered_ids)} sentences passed")
        if "tool_calls" in tracks_set:
            summaries.append(f"tool_calls: {len(tool_calls_dict)} found")
        if "faithful_ack" in tracks_set:
            summaries.append(f"faithful: {len(faithful_ids)}, acks: {len(ack_ids)}")
        if "inert" in tracks_set:
            summaries.append(f"inert: {len(inert_ids)}")
        _phase_result(summaries, phase1_cost, time.time() - t0)

    # ── Phase 2: Attribution Classification ───────────────────────────────
    attribution_dict = {}
    if "attribution" in tracks_set:
        phase_counter += 1
        if filtered_ids:
            if verbose:
                _phase_header(phase_counter, total_phases, "Attribution classification",
                              f"{len(filtered_ids)} filtered sentences")
            t0 = time.time()
            # Format with alpha IDs if enabled, so the LLM sees consistent IDs
            if use_alpha_ids:
                alpha_filtered = [_orig_to_alpha[fid] for fid in filtered_ids]
                classify_prompt = ATTRIBUTION_CLASSIFY_PROMPT.format(
                    question=question, hint=hint,
                    filtered_sentences=_format_filtered_sentences(prompt_sentences, alpha_filtered),
                    model_answer=model_answer,
                )
            else:
                classify_prompt = ATTRIBUTION_CLASSIFY_PROMPT.format(
                    question=question, hint=hint,
                    filtered_sentences=_format_filtered_sentences(sentences, filtered_ids),
                    model_answer=model_answer,
                )
            resp = judge.run(classify_prompt)
            phase2_cost = judge.get_request_cost(resp)
            result.total_cost += phase2_cost
            raw_attribution_dict = _parse_response(resp)
            if not isinstance(raw_attribution_dict, dict):
                raw_attribution_dict = {}

            # Validate step_text for each classified sentence
            attribution_dict = {}
            for sid, data in raw_attribution_dict.items():
                if not isinstance(data, dict):
                    continue
                step_text = data.get("step_text")
                relevant_text = data.get("relevant_text") or step_text
                resolved = _resolve_step_id(_to_orig(sid), step_text, sentences, mismatch_strategy, match_threshold, auto_remap, auto_remap_threshold)
                if resolved is None:
                    continue
                attribution_dict[str(resolved)] = data
                # Store relevant_text in result.texts
                if relevant_text:
                    result.texts[str(resolved)] = relevant_text

            debug["phase2_attribution_classify"] = attribution_dict
            debug["phase2_raw"] = {
                "prompt": classify_prompt,
                "raw_text": resp.text,
                "thoughts": judge.get_thoughts(resp),
            }

            # Extract acks from correctly-attributed sentences
            for sid, data in attribution_dict.items():
                if isinstance(data, dict) and data.get("attribution") == "correct":
                    ack_ids.append(int(sid))

            if verbose:
                counts = {"correct": 0, "incorrect": 0, "vague": 0}
                for data in attribution_dict.values():
                    if isinstance(data, dict):
                        counts[data.get("attribution", "?")] = counts.get(data.get("attribution", "?"), 0) + 1
                _phase_result(
                    [f"{counts['correct']} correct, {counts['incorrect']} incorrect, {counts['vague']} vague"],
                    phase2_cost, time.time() - t0,
                )
        elif verbose:
            _phase_header(phase_counter, total_phases, "Attribution classification", "skipped, 0 filtered")

    # Deduplicate acks (preserve relevant_text from first occurrence)
    seen_acks = set()
    deduped_ack_ids = []
    for aid in ack_ids:
        aid_int = int(aid)
        if aid_int not in seen_acks:
            seen_acks.add(aid_int)
            deduped_ack_ids.append(aid_int)
    ack_ids = deduped_ack_ids
    result.acks = ack_ids
    # Store ack relevant_texts
    for aid_str, rtxt in ack_texts.items():
        result.texts[aid_str] = rtxt

    # ── Phase: Graph Steps (entailment + adversarial) ──────────────────────
    graph_step_flagged = []
    if "graph_steps" in tracks_set and ground_truth_steps:
        phase_counter += 1
        if verbose:
            _phase_header(phase_counter, total_phases, "Graph steps (entailment)",
                          f"{len(ground_truth_steps)} ground truth steps")
        t0 = time.time()

        from isolate_steps import get_top_step_candidates_using_entailment

        # Build list of CoT sentence texts for entailment
        cot_texts = [s["text"] for s in sentences]

        for gt_step in ground_truth_steps:
            _, top_scores, top_indices = get_top_step_candidates_using_entailment(
                cot_texts, gt_step, k=entailment_k, batch_size=10, threshold=0.2,
            )
            for score, idx in zip(top_scores, top_indices):
                sid = int(sentences[idx]["id"])
                graph_step_flagged.append(FlaggedItem(
                    id=str(sid),
                    text=cot_texts[idx],
                    source_track="graph_steps",
                    label={
                        "ground_truth_step": gt_step,
                        "entailment_score": score,
                    },
                ))

        debug["graph_steps_entailment"] = [
            {
                "ground_truth_step": item.label["ground_truth_step"],
                "sentence_id": item.id,
                "sentence_text": item.text,
                "entailment_score": item.label["entailment_score"],
            }
            for item in graph_step_flagged
        ]

        if verbose:
            print(f"  {len(graph_step_flagged)} candidates from entailment")
            print(f"  {time.time() - t0:.1f}s")

    # ── Build flagged items for post-processing ───────────────────────────
    flagged = []

    # Attribution-unfaithful items
    for sid, data in attribution_dict.items():
        if isinstance(data, dict) and not data.get("faithful", True):
            text = data.get("relevant_text") or data.get("step_text") or _get_sentence_text(sentences, sid)
            flagged.append(FlaggedItem(
                id=str(sid),
                text=text,
                source_track="attribution",
                label=data,
            ))

    # Tool call items
    for sid, data in tool_calls_dict.items():
        if isinstance(data, dict):
            text = data.get("relevant_text") or data.get("step_text") or _get_sentence_text(sentences, sid)
            flagged.append(FlaggedItem(
                id=str(sid),
                text=text,
                source_track="tool_calls",
                label=data,
            ))

    # Faithful items
    for sid in faithful_ids:
        text = faithful_texts.get(str(sid), _get_sentence_text(sentences, sid))
        label = {"type": "faithful"}
        if str(sid) in faithful_texts:
            label["relevant_text"] = faithful_texts[str(sid)]
        flagged.append(FlaggedItem(
            id=str(sid),
            text=text,
            source_track="faithful",
            label=label,
        ))

    # Graph step candidates from entailment — when one_per_step is True,
    # we process them in rounds (best entailment score first per GT step)
    # and stop once each step has a match, to save API calls.
    if one_per_step and graph_step_flagged:
        # Group by ground truth step, sorted by entailment score descending
        from collections import defaultdict
        gt_step_groups: dict[str, list[FlaggedItem]] = defaultdict(list)
        for item in graph_step_flagged:
            gt_step_groups[item.label["ground_truth_step"]].append(item)
        for items_list in gt_step_groups.values():
            items_list.sort(key=lambda x: x.label.get("entailment_score", 0), reverse=True)
        # Convert to list of queues (index 0 = best candidate)
        gt_step_queues: dict[str, list[FlaggedItem]] = dict(gt_step_groups)
    else:
        gt_step_queues = None
        flagged.extend(graph_step_flagged)

    if verbose:
        by_track = {}
        all_for_count = flagged + (graph_step_flagged if gt_step_queues else [])
        for it in all_for_count:
            by_track[it.source_track] = by_track.get(it.source_track, 0) + 1
        parts = [f"{v} {k}" for k, v in by_track.items()]
        total_flagged = len(all_for_count)
        print(f"  Flagged for post-processing: {total_flagged} items ({', '.join(parts) or 'none'})")
        if gt_step_queues:
            print(f"  one_per_step: processing {len(gt_step_queues)} GT steps in rounds")

    # ── Phase 3: Adversarial Validation ───────────────────────────────────
    if not skip_adversarial:
        phase_counter += 1
        adv_cost_total = 0.0
        adv_details_all = []
        t0 = time.time()

        if gt_step_queues is not None:
            # ── Round-based adversarial validation for graph_steps ─────
            # Round 1: all non-graph flagged items + top candidate per GT step
            resolved_steps: set[str] = set()
            round_num = 0
            # Collect surviving graph_steps items across rounds
            graph_step_survivors: list[FlaggedItem] = []
            # Recommendation tracking
            recommendation_counts: dict[str, int] = {}
            evaluated_pairs: set[tuple[str, str]] = set()
            # Seed evaluated_pairs with all entailment candidates
            for item in graph_step_flagged:
                evaluated_pairs.add((
                    item.label["ground_truth_step"], item.id
                ))

            while True:
                round_num += 1
                round_candidates = []
                for gt_step, queue in gt_step_queues.items():
                    if gt_step in resolved_steps:
                        continue
                    if queue:
                        round_candidates.append(queue.pop(0))

                if not round_candidates and round_num > 1:
                    break  # No more candidates to try

                # First round includes non-graph flagged items
                if round_num == 1:
                    batch = flagged + round_candidates
                else:
                    batch = round_candidates

                if not batch:
                    break

                if verbose:
                    if round_num == 1:
                        _phase_header(phase_counter, total_phases, "Adversarial validation",
                                      f"round {round_num}: {len(flagged)} non-graph + {len(round_candidates)} graph candidates")
                    else:
                        _phase_header(phase_counter, total_phases, "Adversarial validation",
                                      f"round {round_num}: {len(round_candidates)} graph candidates "
                                      f"({len(gt_step_queues) - len(resolved_steps)} unresolved steps)")

                surviving, cost, details = run_adversarial_validation(
                    batch, question, hint, sentences, judge, use_tqdm=verbose,
                    steps_adversarial_prompt=steps_adversarial_prompt,
                    adversarial_judge=adversarial_judge,
                    dual_adversarial=dual_adversarial,
                )
                adv_cost_total += cost
                adv_details_all.extend(details)

                # Separate graph_steps survivors from other track survivors
                for item in surviving:
                    if item.source_track == "graph_steps":
                        gt = item.label.get("ground_truth_step", "")
                        if gt not in resolved_steps:
                            resolved_steps.add(gt)
                            graph_step_survivors.append(item)
                        # else: already resolved, skip this duplicate

                # After round 1, replace flagged with non-graph survivors
                if round_num == 1:
                    flagged = [it for it in surviving if it.source_track != "graph_steps"]

                # Collect recommendations from failed graph_steps and inject
                # into queues (front, so they're evaluated next round)
                recommended = _collect_recommendations(
                    details, sentences, evaluated_pairs,
                    recommendation_counts, max_recommendations_per_step,
                )
                for rec_item in recommended:
                    gt = rec_item.label["ground_truth_step"]
                    if gt not in resolved_steps:
                        gt_step_queues.setdefault(gt, []).insert(0, rec_item)

                if verbose:
                    dropped_this_round = [d for d in details if d["dropped"]]
                    kept_this_round = [d for d in details if not d["dropped"]]
                    lines = [f"{len(kept_this_round)} survived, {len(dropped_this_round)} dropped"]
                    newly_resolved = [d for d in details
                                      if not d["dropped"] and d["source_track"] == "graph_steps"]
                    if newly_resolved:
                        lines.append(f"  Resolved {len(newly_resolved)} GT steps this round")
                    lines.append(f"  {len(resolved_steps)}/{len(gt_step_queues)} GT steps resolved total")
                    if recommended:
                        lines.append(f"  {len(recommended)} judge recommendation(s) queued")
                    _phase_result(lines, cost, time.time() - t0)

                # Check if all steps resolved or no more candidates
                unresolved_with_candidates = any(
                    gt not in resolved_steps and queue
                    for gt, queue in gt_step_queues.items()
                )
                if not unresolved_with_candidates:
                    break

            # Skipped candidates (remaining in queues for resolved steps) get logged
            skipped_count = 0
            for gt_step, queue in gt_step_queues.items():
                if gt_step in resolved_steps:
                    skipped_count += len(queue)
                    for item in queue:
                        adv_details_all.append({
                            "id": item.id,
                            "source_track": "graph_steps",
                            "dropped": False,
                            "reason": f"skipped (one_per_step: step already resolved)",
                            "skipped": True,
                            "parsed": {},
                            "raw_text": "",
                            "thoughts": "",
                            "prompt": "",
                        })

            if verbose and skipped_count:
                print(f"  one_per_step: skipped {skipped_count} candidates for already-resolved steps")

            # Merge graph_step survivors back into flagged
            flagged.extend(graph_step_survivors)
            result.total_cost += adv_cost_total
            debug["phase3_post_adversarial"] = [
                {"id": it.id, "track": it.source_track, "label": it.label}
                for it in flagged
            ]
            debug["phase3_adversarial_details"] = adv_details_all

        elif flagged:
            # ── Standard single-pass adversarial validation ────────────
            if verbose:
                _phase_header(phase_counter, total_phases, "Adversarial validation",
                              f"{len(flagged)} flagged items")
            flagged, adv_cost, adv_details = run_adversarial_validation(
                flagged, question, hint, sentences, judge, use_tqdm=verbose,
                steps_adversarial_prompt=steps_adversarial_prompt,
                adversarial_judge=adversarial_judge,
                dual_adversarial=dual_adversarial,
            )
            result.total_cost += adv_cost
            all_adv_details = list(adv_details)

            if verbose:
                dropped_items = [d for d in adv_details if d["dropped"]]
                kept_items = [d for d in adv_details if not d["dropped"]]
                lines = [f"{len(kept_items)} survived, {len(dropped_items)} dropped"]
                for d in dropped_items:
                    lines.append(f"  DROPPED {d['id']} ({d['source_track']}): {d['reason']}")
                _phase_result(lines, adv_cost, time.time() - t0)

            # Follow-up rounds for judge recommendations (single-pass path)
            if graph_step_flagged and max_recommendations_per_step > 0:
                recommendation_counts: dict[str, int] = {}
                evaluated_pairs: set[tuple[str, str]] = set()
                # Seed with all items already evaluated
                for d in adv_details:
                    if d.get("source_track") == "graph_steps" and d.get("gt_step"):
                        evaluated_pairs.add((d["gt_step"], d["id"]))

                latest_details = adv_details
                for _rec_round in range(max_recommendations_per_step):
                    recommended = _collect_recommendations(
                        latest_details, sentences, evaluated_pairs,
                        recommendation_counts, max_recommendations_per_step,
                    )
                    if not recommended:
                        break

                    if verbose:
                        print(f"  Recommendation follow-up: {len(recommended)} items")

                    rec_surviving, rec_cost, rec_details = run_adversarial_validation(
                        recommended, question, hint, sentences, judge,
                        use_tqdm=verbose,
                        steps_adversarial_prompt=steps_adversarial_prompt,
                        adversarial_judge=adversarial_judge,
                        dual_adversarial=dual_adversarial,
                    )
                    result.total_cost += rec_cost
                    all_adv_details.extend(rec_details)
                    flagged.extend(rec_surviving)
                    latest_details = rec_details  # next round only checks new details

                    if verbose:
                        rec_kept = [d for d in rec_details if not d["dropped"]]
                        rec_dropped = [d for d in rec_details if d["dropped"]]
                        print(f"    {len(rec_kept)} survived, {len(rec_dropped)} dropped")

            debug["phase3_post_adversarial"] = [
                {"id": it.id, "track": it.source_track, "label": it.label}
                for it in flagged
            ]
            debug["phase3_adversarial_details"] = all_adv_details
        elif verbose:
            _phase_header(phase_counter, total_phases, "Adversarial validation", "skipped, 0 flagged")

    # ── Deferred Inert Phase (smart_inert) ──────────────────────────────
    if inert_deferred:
        phase_counter += 1
        has_unfaithful = any(
            it.source_track in ("attribution", "tool_calls", "graph_steps")
            for it in flagged
        )
        n_sents = len(sentences)
        should_run_inert = (n_sents <= 50) and (not has_unfaithful)

        if should_run_inert:
            if verbose:
                _phase_header(phase_counter, total_phases, "Deferred inert detection",
                              f"{n_sents} sentences, no unfaithful — running")
            t0 = time.time()
            inert_template = GRAPH_INERT_PROMPT if is_graph_mode else INERT_PROMPT
            inert_prompts = []
            base_fmt_kwargs = {"question": question, "model_answer": model_answer, "hint": hint}
            for chunk in chunks:
                formatted = _format_sentences(chunk)
                inert_prompts.append(inert_template.format(sentences=formatted, **base_fmt_kwargs))

            inert_responses = judge.run_batch(
                inert_prompts, max_concurrency=max_concurrency, use_tqdm=verbose,
            )
            inert_cost = sum(judge.get_request_cost(r) for r in inert_responses)
            result.total_cost += inert_cost

            raw_inert_all = []
            for resp in inert_responses:
                parsed = _parse_response(resp)
                if not isinstance(parsed, list):
                    parsed = []
                raw_inert_all.extend(parsed)

            for entry in raw_inert_all:
                if isinstance(entry, dict):
                    iid = _to_orig(entry.get("id", 0))
                    step_text = entry.get("step_text")
                    iid = _resolve_step_id(iid, step_text, sentences, mismatch_strategy, match_threshold, auto_remap, auto_remap_threshold)
                    if iid is not None:
                        inert_ids.append(iid)
                else:
                    _handle_malformed_entry(entry, "inert (deferred)", mismatch_strategy)

            result.inert = inert_ids
            result.inert_status = "ran"

            if verbose:
                _phase_result([f"inert: {len(inert_ids)}"], inert_cost, time.time() - t0)
        else:
            skip_reason = (
                f"{n_sents} sentences > 50" if n_sents > 50
                else "unfaithful steps found"
            )
            result.inert_status = f"skipped ({skip_reason})"
            if verbose:
                _phase_header(phase_counter, total_phases, "Deferred inert detection",
                              f"skipped ({skip_reason})")

    # ── Phase 4: Aggregate Final Output ───────────────────────────────────
    phase_counter += 1
    for item in flagged:
        result.texts[item.id] = item.text
        if item.source_track == "attribution":
            result.attribution[item.id] = item.label
        elif item.source_track == "tool_calls":
            result.tool_calls[item.id] = item.label
        elif item.source_track == "faithful":
            sid = int(item.id) if item.id.isdigit() else item.id
            if sid not in result.faithful:
                result.faithful.append(sid)
        elif item.source_track == "graph_steps":
            gt_step = item.label.get("ground_truth_step", "")
            sid = int(item.id) if item.id.isdigit() else item.id
            relevant_text = item.label.get("relevant_text", item.text)
            result.graph_steps.setdefault(gt_step, []).append({
                "sentence_id": sid,
                "relevant_text": relevant_text,
            })

    # Also include correctly-attributed items in the attribution dict
    for sid, data in attribution_dict.items():
        if isinstance(data, dict) and data.get("faithful") and sid not in result.attribution:
            result.attribution[sid] = data

    result.debug = debug
    result.wall_time_s = time.time() - pipeline_start

    if verbose:
        n_unfaithful_attr = sum(1 for d in result.attribution.values() if isinstance(d, dict) and not d.get("faithful", True))
        n_correct_attr = sum(1 for d in result.attribution.values() if isinstance(d, dict) and d.get("faithful"))
        print(f"\n[{phase_counter}/{total_phases}] Done")
        print(f"  Attribution: {n_unfaithful_attr} unfaithful, {n_correct_attr} correct")
        print(f"  Tool calls:  {len(result.tool_calls)}")
        print(f"  Faithful:    {len(result.faithful)}")
        print(f"  Inert:       {len(result.inert)}")
        print(f"  Acks:        {len(result.acks)}")
        if result.graph_steps:
            matched = sum(1 for ids in result.graph_steps.values() if ids)
            total_gt = len(ground_truth_steps) if ground_truth_steps else 0
            print(f"  Graph steps: {matched}/{total_gt} matched")
        print(f"  Total cost:  ${result.total_cost:.4f} | {result.wall_time_s:.1f}s")

    return result


# =============================================================================
# Debugging Helpers
# =============================================================================

def debug_track(result: PipelineResult, track_name: str, chunk_idx: int | None = None):
    """
    Print the judge's reasoning (thoughts) and raw output for a specific Phase 1 track.

    Args:
        result: PipelineResult from run_multipass_pipeline
        track_name: One of "attribution_filter", "tool_calls", "faithful_ack", "inert"
        chunk_idx: Which chunk to show. None shows all chunks.
    """
    raw = result.debug.get("phase1_raw", {})
    if track_name not in raw:
        available = list(raw.keys())
        print(f"Track '{track_name}' not found in debug. Available: {available}")
        return

    entries = raw[track_name]
    parsed = result.debug.get("phase1", {}).get(track_name)
    print(f"=== {track_name} ({len(entries)} chunk(s)) ===")
    print(f"\n--- Merged Parsed Output ---")
    print(json.dumps(parsed, indent=2, default=str))

    chunks_to_show = [entries[chunk_idx]] if chunk_idx is not None else entries
    for entry in chunks_to_show:
        cidx = entry.get("chunk_idx", 0)
        header = f" (chunk {cidx})" if len(entries) > 1 else ""
        print(f"\n--- Raw Response{header} ---")
        print(entry["raw_text"])
        print(f"\n--- Judge Thoughts{header} ---")
        print(entry["thoughts"])


def debug_attribution_classify(result: PipelineResult):
    """Print the judge's reasoning for the Phase 2 attribution classification."""
    raw = result.debug.get("phase2_raw")
    if not raw:
        print("No Phase 2 debug data (attribution classification may have been skipped).")
        return

    parsed = result.debug.get("phase2_attribution_classify", {})
    print(f"=== Attribution Classification ===")
    print(f"\n--- Parsed Output ---")
    print(json.dumps(parsed, indent=2, default=str))
    print(f"\n--- Raw Response ---")
    print(raw["raw_text"])
    print(f"\n--- Judge Thoughts ---")
    print(raw["thoughts"])


def debug_prompt(result: PipelineResult, track_name: str, chunk_idx: int = 0):
    """
    Print the exact prompt that was sent to the judge for a track.
    Useful for checking whether the prompt was well-formed.

    Args:
        result: PipelineResult from run_multipass_pipeline
        track_name: "attribution_filter", "tool_calls", "faithful_ack", "inert",
                     or "attribution_classify" for Phase 2.
        chunk_idx: Which chunk's prompt to show (default 0).
    """
    if track_name == "attribution_classify":
        raw = result.debug.get("phase2_raw")
        if not raw:
            print("No Phase 2 prompt (attribution classification may have been skipped).")
            return
        print(raw["prompt"])
        return

    raw = result.debug.get("phase1_raw", {})
    if track_name not in raw:
        available = list(raw.keys())
        print(f"Track '{track_name}' not found. Available: {available}")
        return
    entries = raw[track_name]
    if chunk_idx >= len(entries):
        print(f"Chunk {chunk_idx} not found. {len(entries)} chunk(s) available.")
        return
    if len(entries) > 1:
        print(f"[chunk {chunk_idx}/{len(entries)}]")
    print(entries[chunk_idx]["prompt"])


def debug_adversarial(result: PipelineResult, sentence_id: int | str):
    """
    Print full adversarial validation details for a specific sentence.
    Shows the prompt sent, parsed result, raw response, and judge thoughts.

    Args:
        result: PipelineResult from run_multipass_pipeline
        sentence_id: The sentence ID to inspect (int or str, e.g. 14 or "14a")
    """
    sid = str(sentence_id)
    details = result.debug.get("phase3_adversarial_details", [])
    match = [d for d in details if d["id"] == sid]
    if not match:
        available = [d["id"] for d in details]
        print(f"No adversarial details for sentence {sid}. Available: {available}")
        return

    d = match[0]
    status = "DROPPED" if d["dropped"] else "KEPT"
    print(f"=== Adversarial Validation: Sentence {sid} ({d['source_track']}) — {status} ===")
    print(f"Reason: {d['reason']}")
    print(f"\n--- Parsed Output ---")
    print(json.dumps(d["parsed"], indent=2, default=str))
    print(f"\n--- Prompt ---")
    print(d["prompt"])
    print(f"\n--- Raw Response ---")
    print(d["raw_text"])
    print(f"\n--- Judge Thoughts ---")
    print(d["thoughts"])



def debug_sentence(result: PipelineResult, sentence_id: int, sentences: list[dict]):
    """
    Show everything the pipeline determined about a specific sentence.
    Useful when you expect a sentence to be flagged but it wasn't, or vice versa.

    Args:
        result: PipelineResult from run_multipass_pipeline
        sentence_id: The sentence ID to inspect
        sentences: The original sentence list passed to the pipeline
    """
    sid = str(sentence_id)
    text = _get_sentence_text(sentences, sentence_id)
    print(f"=== Sentence {sentence_id} ===")
    print(f"Text: {text}\n")

    found_anything = False

    # Check attribution
    if sid in result.attribution:
        found_anything = True
        data = result.attribution[sid]
        print(f"Attribution: {json.dumps(data, indent=2)}")

    # Check tool calls
    if sid in result.tool_calls:
        found_anything = True
        data = result.tool_calls[sid]
        print(f"Tool call: {json.dumps(data, indent=2)}")

    # Check faithful
    if sentence_id in result.faithful or sid in [str(x) for x in result.faithful]:
        found_anything = True
        print("Faithful: YES")

    # Check inert
    if sentence_id in result.inert:
        found_anything = True
        print("Inert: YES")

    # Check acks
    if sentence_id in result.acks:
        found_anything = True
        print("Ack: YES")

    if not found_anything:
        print("Not flagged by any track.")

    # Check if it appeared in intermediate phases but was dropped
    print("\n--- Pipeline trace ---")

    # Phase 1: was it in the attribution filter?
    phase1 = result.debug.get("phase1", {})
    filtered = phase1.get("attribution_filter", [])
    if isinstance(filtered, list) and sentence_id in filtered:
        print(f"Phase 1: passed attribution filter")
    else:
        print(f"Phase 1: NOT in attribution filter")

    # Phase 2: was it classified?
    phase2 = result.debug.get("phase2_attribution_classify", {})
    if sid in phase2:
        print(f"Phase 2: classified as {phase2[sid]}")

    # Phase 3: adversarial validation
    adv_details = result.debug.get("phase3_adversarial_details", [])
    adv_match = [d for d in adv_details if d["id"] == sid]
    if adv_match:
        d = adv_match[0]
        status = "DROPPED" if d["dropped"] else "KEPT"
        print(f"Phase 3 (adversarial): {status} — {d['reason']}")
        if d.get("thoughts"):
            print(f"  Judge thoughts: {d['thoughts'][:500]}")
    else:
        phase3 = result.debug.get("phase3_post_adversarial", [])
        adv_items = [it for it in phase3 if it["id"] == sid]
        if adv_items:
            print(f"Phase 3 (adversarial): survived (no detail captured)")

    # Check for relevant_text extraction
    if sid in result.texts:
        full_text = _get_sentence_text(sentences, sentence_id)
        if result.texts[sid] != full_text:
            print(f"Relevant text extracted: \"{result.texts[sid]}\"")


# =============================================================================
# Convenience Wrappers
# =============================================================================

def run_multipass_from_thinking(
    question: str,
    hint: str,
    thinking: str,
    model_answer: str,
    judge: Judge,
    max_concurrency: int = 10,
    verbose: bool = False,
    chunk_size: int | None = 250,
    mismatch_strategy: str = "skip",
    match_threshold: float = 0.9,
    auto_remap: bool = True,
    auto_remap_threshold: float = 0.98,
    use_alpha_ids: bool = False,
    ground_truth_steps: list[str] | None = None,
    steps_adversarial_prompt: str | None = None,
    one_per_step: bool = True,
    max_recommendations_per_step: int = 3,
    adversarial_judge: Judge | None = None,
    dual_adversarial: bool = False,
    smart_inert: bool = True,
    **kwargs,
) -> PipelineResult:
    """
    Convenience wrapper: raw CoT text -> sentence splitting -> pipeline.
    """
    sents = split_into_sentences(thinking)
    sentences = [{"id": i + 1, "text": s} for i, s in enumerate(sents)]
    return run_multipass_pipeline(
        question, hint, sentences, model_answer, judge,
        max_concurrency=max_concurrency, verbose=verbose,
        chunk_size=chunk_size, mismatch_strategy=mismatch_strategy,
        match_threshold=match_threshold, auto_remap=auto_remap,
        auto_remap_threshold=auto_remap_threshold,
        use_alpha_ids=use_alpha_ids,
        ground_truth_steps=ground_truth_steps,
        steps_adversarial_prompt=steps_adversarial_prompt,
        one_per_step=one_per_step,
        max_recommendations_per_step=max_recommendations_per_step,
        adversarial_judge=adversarial_judge,
        dual_adversarial=dual_adversarial,
        smart_inert=smart_inert,
        **kwargs,
    )
