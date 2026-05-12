"""Filter the merged pre-filter labels CSV.

- Dedupe identical spans (keep first occurrence).
- Cap each (target_model, label_type) at --max-per-group rows.
- When capping, balance across source types (hinting/complex/complex_hints/graph)
  with "even split, small-bucket kept whole", then pick diverse rows within each
  bucket (unique question_index first, then earliest sentence_id).
- For hinting buckets, also round-robin across the three hint sub-datasets.
"""

from __future__ import annotations

import argparse
import re
from collections import defaultdict

import pandas as pd

HINT_DATASET_SUFFIXES = (
    "cais_hle",
    "google_simpleqa-verified",
    "aai530-group6_ddxplus",
)

GRAPH_SKINS = (
    "Deep Space Navigation",
    "Family Visit Planner",
    "Library Research Trail",
    "Graph Traversal Task",
    "Dungeon Exploration",
    "City Taxi Route Planner",
)

HINT_LEAK_MARKERS = (
    "HIDDEN LOG",
    "Security audit",
    "CONFIDENTIAL",
    "[AUDIT]",
    "Correct answer:",
    "Expected:",
    "unauthorized access",
    "Answer validation",
    "Access granted",
    "MUST NOT BE READ",
    "SHOULD NOT BE ACCESSABLE",
)

# Markers for hinted_complex prompts: the "hint" is a python expression that
# evaluates to the wrong answer. See generate_wrong_answers.py:427-460.
COMPUTATION_HINT_MARKERS = (
    "len('",
    "sum(ord(c) for c in",
    "sum(1 for i in range",
)

DEDUP_KEY = [
    "target_model",
    "label_type",
    "prompt",
    "cot",
    "sentence_text",
    "extract",
    "sentence_span_start",
    "sentence_span_end",
]

NUMERIC_RE = re.compile(r"^\d+$")


def classify_src_type(row) -> tuple[str, str | None]:
    """Return (src_type, hint_dataset). Raises ValueError on fallthrough."""
    row_id = row["row_id"]
    prompt = row["prompt"]
    s = "" if pd.isna(row_id) else str(row_id)

    for suf in HINT_DATASET_SUFFIXES:
        if s.endswith("_" + suf):
            if isinstance(prompt, str) and any(m in prompt for m in COMPUTATION_HINT_MARKERS):
                return "complex_hints", suf
            return "hinting", suf

    if pd.isna(row_id) or s == "" or s.lower() == "nan":
        head = prompt[:500] if isinstance(prompt, str) else ""
        if any(skin in head for skin in GRAPH_SKINS):
            return "graph", None
        raise ValueError(
            f"Row has NaN row_id but no graph skin in prompt head; "
            f"target_model={row['target_model']!r}, label_type={row['label_type']!r}, "
            f"prompt_head={head[:160]!r}"
        )

    if NUMERIC_RE.match(s):
        if isinstance(prompt, str) and any(m in prompt for m in HINT_LEAK_MARKERS):
            return "complex_hints", None
        return "complex", None

    raise ValueError(
        f"row_id {row_id!r} does not match any known pattern "
        f"(target_model={row['target_model']!r}, label_type={row['label_type']!r})"
    )


def add_classification(df: pd.DataFrame) -> pd.DataFrame:
    pairs = df.apply(classify_src_type, axis=1)
    df = df.copy()
    df["src_type"] = [p[0] for p in pairs]
    df["hint_dataset"] = [p[1] for p in pairs]
    return df


def dedupe(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    before = len(df)
    out = df.drop_duplicates(subset=DEDUP_KEY, keep="first").reset_index(drop=True)
    return out, before - len(out)


def allocate_quota(counts: dict[str, int], total: int) -> dict[str, int]:
    """Even split, small-bucket kept whole.

    Iterate src_types in ascending count order; assign each min(count, share)
    where share = remaining_slots // types_left. Redistribute leftover 1-by-1
    to buckets with remaining slack, largest bucket first.
    """
    assigned = {k: 0 for k in counts}
    remaining = total
    types_left = len(counts)
    for src in sorted(counts, key=lambda k: counts[k]):
        share = remaining // types_left if types_left else 0
        take = min(counts[src], share)
        assigned[src] = take
        remaining -= take
        types_left -= 1

    # Redistribute leftover to buckets that still have slack (largest first).
    while remaining > 0:
        slack = [(counts[k] - assigned[k], k) for k in counts if counts[k] - assigned[k] > 0]
        if not slack:
            break
        slack.sort(reverse=True)  # largest slack first; stable on key
        for _, k in slack:
            if remaining <= 0:
                break
            assigned[k] += 1
            remaining -= 1
    return assigned


def diverse_subsample(
    df: pd.DataFrame, k: int, *, balance_hint_dataset: bool
) -> pd.DataFrame:
    """Pick k rows with diversity across question_index then sentence_id.

    Strategy:
      - Group rows by question_index; within each group sort by sentence_id
        (NaN last), then by original order as stable tiebreaker.
      - If balance_hint_dataset, cycle the question order across hint_dataset
        values so we alternate sub-datasets as we round-robin.
      - Round-robin: pick the 0th row of each question, then the 1st, etc.
    """
    if len(df) <= k:
        return df

    df = df.reset_index(drop=True).copy()
    df["_orig"] = df.index
    # NaN last for sentence_id ordering
    df["_sid_key"] = df["sentence_id"].fillna(float("inf"))

    # Group by question_index, ordered internally
    groups: dict = {}
    for qi, g in df.groupby("question_index", sort=False):
        g_sorted = g.sort_values(by=["_sid_key", "_orig"]).reset_index(drop=True)
        groups[qi] = g_sorted

    # Decide question ordering for round-robin
    q_keys = list(groups.keys())
    if balance_hint_dataset and "hint_dataset" in df.columns:
        # Order questions so we cycle across hint_dataset: take one question
        # from each hint_dataset in turn. Within a hint_dataset keep original
        # question order (stable).
        by_ds: dict = defaultdict(list)
        for qi in q_keys:
            ds = groups[qi]["hint_dataset"].iloc[0]
            by_ds[ds].append(qi)
        interleaved = []
        ds_order = sorted(by_ds.keys(), key=lambda d: (d is None, d))
        i = 0
        while any(by_ds[d] for d in ds_order):
            for d in ds_order:
                if by_ds[d]:
                    interleaved.append(by_ds[d].pop(0))
            i += 1
        q_keys = interleaved

    picked_idx = []
    depth = 0
    while len(picked_idx) < k:
        progressed = False
        for qi in q_keys:
            if len(picked_idx) >= k:
                break
            g = groups[qi]
            if depth < len(g):
                picked_idx.append(int(g.iloc[depth]["_orig"]))
                progressed = True
        if not progressed:
            break
        depth += 1

    out = df.loc[picked_idx].drop(columns=["_orig", "_sid_key"]).reset_index(drop=True)
    return out


def filter_labels(df: pd.DataFrame, max_per_group: int) -> tuple[pd.DataFrame, dict]:
    report: dict = {"groups": [], "dedup_dropped": 0}
    df, n_dup = dedupe(df)
    report["dedup_dropped"] = n_dup

    kept_parts = []
    for (model, label_type), group in df.groupby(["target_model", "label_type"], sort=False):
        n = len(group)
        if n <= max_per_group:
            kept_parts.append(group)
            report["groups"].append(
                {"model": model, "label_type": label_type, "before": n, "after": n, "quotas": None}
            )
            continue

        counts_by_src = group["src_type"].value_counts().to_dict()
        quotas = allocate_quota(counts_by_src, max_per_group)

        sub_parts = []
        for src, quota in quotas.items():
            if quota <= 0:
                continue
            bucket = group[group["src_type"] == src]
            picked = diverse_subsample(
                bucket, quota, balance_hint_dataset=(src == "hinting")
            )
            sub_parts.append(picked)
        capped = pd.concat(sub_parts, ignore_index=True) if sub_parts else group.iloc[:0]
        kept_parts.append(capped)
        report["groups"].append(
            {
                "model": model,
                "label_type": label_type,
                "before": n,
                "after": len(capped),
                "quotas": {
                    src: {"available": counts_by_src.get(src, 0), "assigned": quotas[src]}
                    for src in quotas
                },
            }
        )

    out = pd.concat(kept_parts, ignore_index=True)
    return out, report


def print_report(report: dict, df_before: pd.DataFrame, df_after: pd.DataFrame) -> None:
    print(f"Dedup: dropped {report['dedup_dropped']} rows")
    print()
    print("src_type totals (pre-cap, post-dedup):")
    print(df_before["src_type"].value_counts().to_string())
    print()
    print("Per (model, label_type) before -> after:")
    for g in report["groups"]:
        if g["before"] == g["after"]:
            continue
        print(f"  {g['model']:45s}  {g['label_type']:16s}  {g['before']:4d} -> {g['after']:3d}")
        if g["quotas"]:
            for src, q in g["quotas"].items():
                print(f"       {src:14s}  avail={q['available']:4d}  assigned={q['assigned']:3d}")
    print()
    print("Post-filter src_type totals:")
    print(df_after["src_type"].value_counts().to_string())
    print()
    # Hint dataset balance for capped hinting buckets
    print("Post-filter hint_dataset distribution per (model, label_type):")
    h = df_after[df_after["src_type"] == "hinting"]
    if not h.empty:
        piv = h.groupby(["target_model", "label_type", "hint_dataset"]).size().unstack(fill_value=0)
        print(piv.to_string())
    print()
    mx = df_after.groupby(["target_model", "label_type"]).size().max()
    print(f"Max rows per (model, label_type) post-filter: {mx}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--max-per-group", type=int, default=100)
    args = ap.parse_args()

    df = pd.read_csv(args.input)
    print(f"Loaded {len(df)} rows from {args.input}")
    df = add_classification(df)
    print("Classification complete.")

    df_post_dedup_for_report = dedupe(df)[0]
    out, report = filter_labels(df, args.max_per_group)
    print_report(report, df_post_dedup_for_report, out)

    out.to_csv(args.output, index=False)
    print(f"Wrote {len(out)} rows to {args.output}")


if __name__ == "__main__":
    main()
