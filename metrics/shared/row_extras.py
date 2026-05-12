"""Helpers for extracting per-row context off labeled-data CSVs.

Mirrors the JSON-or-newlines parsing in ``orchestrator/executors.py:_build_row_kwargs``
so the same logic backs both the judging dispatch and the metrics dispatch.
"""

from __future__ import annotations

import json
from typing import Any


def parse_ground_truth_steps(raw: Any) -> list[str]:
    """Parse a ``steps`` / ``intermediate_step`` column value into a list of strings.

    Accepts:
      - a JSON list (preferred for complex tasks)
      - a newline-separated string (graph tasks)
      - a single non-empty string (hinted_complex's ``intermediate_step``)
      - an already-iterable of strings
    Returns ``[]`` for missing / empty inputs.
    """
    if raw is None:
        return []
    if isinstance(raw, float):
        # pandas turns missing CSV cells into NaN
        if raw != raw:
            return []
        raw = str(raw)
    if isinstance(raw, str):
        s = raw.strip()
        if not s:
            return []
        if s.startswith("["):
            parsed = json.loads(s)
            if not isinstance(parsed, list):
                raise ValueError(
                    f"Expected JSON list for ground_truth_steps, got {type(parsed).__name__}"
                )
            return [str(x).strip() for x in parsed if str(x).strip()]
        return [line.strip() for line in s.splitlines() if line.strip()]
    if isinstance(raw, (list, tuple)):
        return [str(x).strip() for x in raw if str(x).strip()]
    raise TypeError(f"Unsupported type for ground_truth_steps: {type(raw).__name__}")


def populate_extras_from_row(ctx, row) -> None:
    """Populate ``ctx.extras`` with hint / ground-truth-steps from a CSV row.

    Sets ``prompted_hint`` if non-empty; sets ``ground_truth_steps`` if the row
    has a non-empty ``steps`` (graph/complex) or ``intermediate_step`` (hinted_complex).
    """
    hint_raw = row.get("prompted_hint", "")
    if hint_raw is not None and not (isinstance(hint_raw, float) and hint_raw != hint_raw):
        hint = str(hint_raw)
        if hint:
            ctx.extras["prompted_hint"] = hint

    steps_raw = row.get("steps", "")
    gt_steps = parse_ground_truth_steps(steps_raw)
    if not gt_steps:
        gt_steps = parse_ground_truth_steps(row.get("intermediate_step", ""))
    if gt_steps:
        ctx.extras["ground_truth_steps"] = gt_steps
