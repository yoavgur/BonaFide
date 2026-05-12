"""CLI entry point for running a single faithfulness metric on labeled data.

Usage (one metric, one model):
    python3 -m metrics \
        --model Qwen/Qwen3-4B-Thinking-2507 \
        --input metrics_input.csv \
        --output metrics_results.csv \
        --metric early_answering \
        --backend vllm \
        --tensor_parallel_size 1
"""

from __future__ import annotations

import argparse
import json
import logging
import multiprocessing
import os
import socket
import time
from dataclasses import asdict, is_dataclass
from pathlib import Path

import pandas as pd
from tqdm import tqdm

# vLLM forks subprocesses for its engine core. If CUDA is initialized before
# the fork (e.g. by importing torch/transformers), the subprocess crashes with
# "Cannot re-initialize CUDA in forked subprocess". Setting spawn avoids this.
multiprocessing.set_start_method("spawn", force=True)


def _on_low_mem_node() -> bool:
    # FA2 crashes inside _prepare_from_posids on n-80X/n-60X with multi-GPU
    # splits (empty cu_seq_lens_q.diff()). Use SDPA there; keep FA2 elsewhere
    # for memory/speed on large models.
    return socket.gethostname().lower().startswith(("n-80", "n-60"))


def _patch_rocm_grouped_mm() -> None:
    """On ROCm, torch._grouped_mm raises `grouped gemm is not supported on ROCM`.

    Transformers' MoE integration (Qwen3-MoE etc.) dispatches to it unconditionally.
    Replace it with a loop-based fallback so MoE models run on AMD GPUs.
    """
    import torch

    if not getattr(torch.version, "hip", None):
        return
    if not hasattr(torch, "_grouped_mm"):
        return
    if getattr(torch._grouped_mm, "_rocm_patched", False):
        return

    def _fallback(mat1, mat2, offs=None, bias=None, out_dtype=None):
        # Grouped variant: mat1 [total_M, K], mat2 [G, K, N], offs [G] cumulative.
        # Batched variant: mat1 [G, M, K], mat2 [G, K, N], offs=None.
        if offs is None:
            out = torch.matmul(mat1, mat2)
        else:
            G = mat2.shape[0]
            outs = []
            start = 0
            offs_list = offs.tolist()
            for g in range(G):
                end = offs_list[g]
                if end > start:
                    outs.append(torch.matmul(mat1[start:end], mat2[g]))
                else:
                    outs.append(mat1.new_empty((0, mat2.shape[-1])))
                start = end
            out = torch.cat(outs, dim=0) if outs else mat1.new_empty((0, mat2.shape[-1]))
        if bias is not None:
            out = out + bias
        if out_dtype is not None:
            out = out.to(out_dtype)
        return out

    _fallback._rocm_patched = True
    torch._grouped_mm = _fallback
    logging.getLogger(__name__).info(
        "ROCm detected: patched torch._grouped_mm with Python-loop fallback"
    )

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

# ---------------------------------------------------------------------------
# Metric registry: name -> (MetricClass, ConfigClass)
# ---------------------------------------------------------------------------

METRIC_REGISTRY: dict[str, tuple[type, type]] = {}


def _register_metrics():
    """Lazy import to avoid loading all metrics at module level."""
    from metrics.early_answering import EarlyAnsweringMetric
    from metrics.early_answering.config import EarlyAnsweringConfig
    from metrics.filler_tokens import FillerTokensMetric
    from metrics.filler_tokens.config import FillerTokensConfig
    from metrics.simulatability import SimulatabilityMetric
    from metrics.simulatability.config import SimulatabilityConfig
    from metrics.adding_mistakes import AddingMistakesMetric
    from metrics.adding_mistakes.config import AddingMistakesConfig
    from metrics.paraphrasing import ParaphrasingMetric
    from metrics.paraphrasing.config import ParaphrasingConfig
    from metrics.fur import FURMetric
    from metrics.fur.config import FURConfig
    from metrics.cc_shap import CCSHAPMetric
    from metrics.cc_shap.config import CCSHAPConfig
    from metrics.scm import SCMMetric
    from metrics.scm.config import SCMConfig
    from metrics.monitor import (
        MonitorGenericMetric,
        MonitorMetric,
        MonitorNoHintMetric,
        MonitorNoToolMetric,
    )
    from metrics.monitor.config import (
        MonitorConfig,
        MonitorGenericConfig,
        MonitorNoHintConfig,
        MonitorNoToolConfig,
    )
    from metrics.regex_baseline import RegexBaselineMetric
    from metrics.regex_baseline.config import RegexBaselineConfig

    METRIC_REGISTRY.update({
        "early_answering": (EarlyAnsweringMetric, EarlyAnsweringConfig),
        "filler_tokens": (FillerTokensMetric, FillerTokensConfig),
        "simulatability": (SimulatabilityMetric, SimulatabilityConfig),
        "adding_mistakes": (AddingMistakesMetric, AddingMistakesConfig),
        "paraphrasing": (ParaphrasingMetric, ParaphrasingConfig),
        "fur": (FURMetric, FURConfig),
        "cc_shap": (CCSHAPMetric, CCSHAPConfig),
        "scm": (SCMMetric, SCMConfig),
        "monitor": (MonitorMetric, MonitorConfig),
        "monitor_no_hint": (MonitorNoHintMetric, MonitorNoHintConfig),
        "monitor_generic": (MonitorGenericMetric, MonitorGenericConfig),
        "monitor_no_tool": (MonitorNoToolMetric, MonitorNoToolConfig),
        "regex_baseline": (RegexBaselineMetric, RegexBaselineConfig),
    })


# Metrics that accept backend/tensor_parallel_size in their config
_BACKEND_METRICS = {
    "early_answering", "filler_tokens", "simulatability",
    "adding_mistakes", "paraphrasing", "scm",
}


def _build_config(metric_name: str, config_cls: type, args: argparse.Namespace):
    """Build a config object, passing backend/tensor_parallel_size where supported."""
    kwargs = {}
    if metric_name in _BACKEND_METRICS:
        kwargs["backend"] = args.backend
        kwargs["tensor_parallel_size"] = args.tensor_parallel_size
    if metric_name == "fur":
        kwargs["greedy_generation"] = args.fur_greedy_generation
        # validation_only implies no generation → force vLLM off and validation on
        validation_only = getattr(args, "fur_validation_only", False)
        kwargs["validation_only"] = validation_only
        kwargs["use_vllm_generation"] = args.fur_vllm_generation and not validation_only
        kwargs["use_shm_pipeline"] = args.fur_shm_pipeline
        kwargs["fur_batch_size"] = args.fur_batch_size
        kwargs["fur_auto_batch"] = args.fur_auto_batch
        kwargs["gpu_memory_utilization"] = args.fur_gpu_mem_util
        kwargs["compute_validation"] = args.fur_compute_validation or validation_only
        kwargs["compute_mmlu_check"] = getattr(args, "fur_mmlu_check", False)
    if metric_name == "cc_shap":
        kwargs["batch_size"] = args.cc_shap_batch_size
        kwargs["auto_batch"] = args.cc_shap_auto_batch
    if metric_name in ("monitor", "monitor_no_hint", "monitor_generic", "monitor_no_tool"):
        kwargs["judge_model"] = args.judge_model
        kwargs["judge_batch_size"] = args.judge_batch_size
    if metric_name == "regex_baseline":
        if getattr(args, "regex_pattern", None) is not None:
            kwargs["hint_pattern"] = args.regex_pattern
    return config_cls(**kwargs)


log = logging.getLogger("metrics.cli")


def _fur_step_key(row) -> tuple:
    """Unique key for a STEP row, matching the resume key format in completed_keys."""
    return (
        str(row.get("row_id", "")),
        str(row.get("label_type", "")),
        str(row.get("sentence_span_start", "")),
        str(row.get("sentence_span_end", "")),
    )


def _fur_state_dict_filename(row) -> str:
    """Deterministic filename for a step's FF2 state dict (survives restarts)."""
    rid = str(row.get("row_id", "x"))
    s = str(row.get("sentence_span_start", 0))
    e = str(row.get("sentence_span_end", 0))
    return f"ff2_{rid}_{s}_{e}.pt"


def _run_fur_two_stage(
    metric, config, scorable_rows, hf_model, tokenizer, model_name,
    other_instances, df, output_path, _csv_header_written, _csv_columns,
    completed_keys, args, load_kwargs=None,
):
    """Two-stage FUR pipeline: unlearn with HF, generate with vLLM.

    Stage 1: Score COT rows normally (HF, with early stopping).
             Unlearn STEP rows in batches (HF), save FF2 state dicts.
    Stage 2: Destroy HF model, load vLLM, generate batch responses, clean up.
             Repeat for each batch.

    Batching keeps disk usage bounded. Batch size is auto-computed from
    available disk space (leaving 30% headroom) and capped by
    config.fur_batch_size. Disable auto-sizing with --no-fur-auto-batch.

    This is called instead of the normal scoring loop when fur_vllm_generation
    is enabled.
    """
    import torch
    from generation.backends import build_prompt
    from generation.model_registry import (
        ANSWER_ONLY_SYSTEM_PROMPT, COT_SYSTEM_PROMPT, get_model_profile,
    )
    from generation.normalize import answers_match
    from generation.thinking import split_thinking
    from metrics.base import MetricContext
    from metrics.fur.metric import FURUnlearnResult, FURStepResult

    t0 = time.time()
    results = []
    total_api_cost = 0.0

    # Split into COT and STEP rows
    cot_rows = [(idx, row, label_type) for idx, row, is_step, is_cot, label_type in scorable_rows if is_cot]
    step_rows = [(idx, row, label_type) for idx, row, is_step, is_cot, label_type in scorable_rows if is_step]

    log.info("FUR two-stage: %d COT rows (HF), %d STEP rows (unlearn HF → generate vLLM)", len(cot_rows), len(step_rows))

    def _write_result(result_row):
        nonlocal _csv_header_written, _csv_columns
        if _csv_columns is None:
            _csv_columns = list(result_row.keys())
        row_df = pd.DataFrame([result_row], columns=_csv_columns)
        row_df.to_csv(output_path, mode="a", header=not _csv_header_written, index=False)
        _csv_header_written = True

    # Mutable ref so _make_ctx sees reloaded model after batched Stage 2
    model_ref = [hf_model]

    def _make_ctx(row):
        return MetricContext(
            question=str(row.get("prompt", row.get("question", ""))),
            cot=str(row.get("cot", "")),
            answer=str(row.get("model_answer", row.get("answer", ""))),
            model=model_ref[0],
            tokenizer=tokenizer,
            model_name=model_name,
            other_instances=other_instances,
        )

    # ── Stage 1a: Score COT rows with HF (normal flow, early stopping works) ──
    if cot_rows:
        log.info("── Stage 1a: Scoring %d COT rows with HF ──", len(cot_rows))
        pbar = tqdm(cot_rows, desc="fur(cot/HF)", unit="row",
                    bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]")
        for idx, row, label_type in pbar:
            row_t0 = time.time()
            ctx = _make_ctx(row)
            if "question_index" in row:
                q_idx = row["question_index"]
                faithful_spans = set()
                for _, srow in df[
                    (df["question_index"] == q_idx) & (df["label_type"] == "FAITHFUL_STEP")
                ].iterrows():
                    s, e = int(srow.get("sentence_span_start", 0)), int(srow.get("sentence_span_end", 0))
                    if e > s:
                        faithful_spans.add((s, e))
                if faithful_spans:
                    ctx.extras["priority_step_spans"] = faithful_spans

            detail = metric.score_cot_detailed(ctx)
            score = _extract_score(detail, args.metric)
            result_row = dict(row)
            result_row.update({
                "metric": args.metric, "target_model": model_name,
                "score": score, "level": "cot",
                "metadata": _serialize_metadata(detail),
                "api_cost_usd": round(getattr(detail, "api_cost_usd", 0.0), 6),
                "wall_time_s": round(time.time() - row_t0, 3),
                "includes_init": len(results) == 0 and len(completed_keys) == 0,
            })
            total_api_cost += getattr(detail, "api_cost_usd", 0.0)
            results.append(result_row)
            _write_result(result_row)
            log.info("row %d [COT] score=%.4f", idx, score)
        pbar.close()

    # State dict save directory. When shm_pipeline is enabled, use /tmp (local
    # NVMe SSD, ~5-10s writes) instead of NFS (network storage, ~180s writes).
    # /tmp survives within the same SLURM job for resume; if rescheduled to a
    # different node, steps are re-unlearned.
    if hasattr(config, "use_shm_pipeline") and config.use_shm_pipeline:
        import tempfile
        save_dir = os.path.join(tempfile.gettempdir(), f"{Path(str(output_path)).stem}_ff2")
    else:
        save_dir = str(Path(str(output_path)).parent / f"{Path(str(output_path)).stem}_ff2")
    os.makedirs(save_dir, exist_ok=True)

    # ── Compute batch size (auto-sized from disk space or manual) ──
    import shutil
    import math

    # Estimate state dict file size from FF2 parameters
    ff2_bytes = sum(
        p.numel() * p.element_size()
        for n, p in model_ref[0].named_parameters()
        if config.ff2_param_pattern in n
    )
    ff2_file_size = int(ff2_bytes * 1.05)  # ~5% overhead for torch.save format

    batch_size = config.fur_batch_size

    if config.fur_auto_batch and ff2_file_size > 0:
        disk_free = shutil.disk_usage(save_dir).free
        usable = int(disk_free * 0.70)  # leave 30% headroom
        auto_batch = max(1, usable // ff2_file_size)
        batch_size = min(batch_size, auto_batch)
        log.info("FUR auto-batch: ff2_size=%.1fGB, disk_free=%.1fGB, usable=%.1fGB "
                 "→ batch_size=%d (cap=%d)",
                 ff2_file_size / 1e9, disk_free / 1e9, usable / 1e9,
                 batch_size, config.fur_batch_size)
    else:
        log.info("FUR batch_size=%d (auto-batch disabled)", batch_size)

    # ── Pre-compute invariants for Stage 2 (shared across all batches) ──
    import subprocess
    import sys

    profile = get_model_profile(model_name)
    max_model_len = profile.max_model_len if profile.max_model_len is not None else 32768

    # Determine sampling parameters
    if config.greedy_generation or profile.temperature <= 0:
        temperature, top_p, top_k = 0.0, 1.0, -1
    else:
        temperature = profile.temperature
        top_p = profile.top_p
        top_k = profile.top_k

    # Determine system message for prompt building
    if profile.thinking_tag is not None:
        system_message = ANSWER_ONLY_SYSTEM_PROMPT
    elif profile.inject_cot_prompt:
        system_message = COT_SYSTEM_PROMPT
    else:
        system_message = ANSWER_ONLY_SYSTEM_PROMPT

    # ── Filter step_rows: skip already-completed, count reusable state dicts ──
    # Build the list of rows that actually need processing (not yet in CSV).
    # Rows with existing state dicts on disk are included (reused in Stage 2).
    actionable_steps = []  # (idx, row, label_type, sd_path, needs_unlearn)
    n_skipped_csv = 0
    n_reused_disk = 0

    for idx, row, label_type in step_rows:
        span_start = int(row.get("sentence_span_start", 0))
        span_end = int(row.get("sentence_span_end", 0))
        if span_end <= span_start:
            raise ValueError(f"Row {idx} has invalid step span ({span_start}, {span_end})")

        if _fur_step_key(row) in completed_keys:
            n_skipped_csv += 1
            continue

        sd_filename = _fur_state_dict_filename(row)
        sd_path = os.path.join(save_dir, sd_filename)

        if os.path.exists(sd_path):
            n_reused_disk += 1
            actionable_steps.append((idx, row, label_type, sd_path, False))
        else:
            actionable_steps.append((idx, row, label_type, sd_path, True))

    if n_skipped_csv > 0 or n_reused_disk > 0:
        log.info("STEP rows: %d actionable, %d already in CSV, %d reusable from disk",
                 len(actionable_steps), n_skipped_csv, n_reused_disk)

    # ── Batch loop: unlearn → generate → clean up → repeat ──
    num_batches = max(1, math.ceil(len(actionable_steps) / batch_size)) if actionable_steps else 0

    for batch_idx in range(num_batches):
        batch = actionable_steps[batch_idx * batch_size : (batch_idx + 1) * batch_size]
        is_last_batch = (batch_idx == num_batches - 1)

        log.info("── Batch %d/%d: %d steps (saving to %s) ──",
                 batch_idx + 1, num_batches, len(batch), save_dir)

        # Re-check disk space for auto-batch (space may have changed from prior cleanup)
        if config.fur_auto_batch and ff2_file_size > 0 and batch_idx > 0:
            disk_free = shutil.disk_usage(save_dir).free
            usable = int(disk_free * 0.70)
            new_auto = max(1, usable // ff2_file_size)
            if new_auto < batch_size:
                log.warning("Disk space tighter than expected: reducing batch_size %d → %d",
                            batch_size, new_auto)
                batch_size = new_auto
                # Re-slice batch to the smaller size (extra items go to next iteration)
                batch = actionable_steps[batch_idx * batch_size : (batch_idx + 1) * batch_size]
                num_batches = batch_idx + max(1, math.ceil(
                    (len(actionable_steps) - batch_idx * batch_size) / batch_size))
                is_last_batch = (batch_idx == num_batches - 1)

        # ── Stage 1b: Unlearn this batch ──
        pending_steps = []
        n_unlearned = 0

        pbar = tqdm(batch, desc=f"fur(unlearn {batch_idx+1}/{num_batches})", unit="row",
                    bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]")
        for idx, row, label_type, sd_path, needs_unlearn in pbar:
            span_start = int(row.get("sentence_span_start", 0))
            span_end = int(row.get("sentence_span_end", 0))

            if not needs_unlearn:
                # Reuse existing state dict from disk
                pending_steps.append((idx, row, FURUnlearnResult(
                    step_text=str(row.get("cot", ""))[span_start:span_end],
                    char_start=span_start,
                    char_end=span_end,
                    original_answer=str(row.get("model_answer", row.get("answer", ""))),
                    state_dict_path=sd_path,
                    question=str(row.get("prompt", row.get("question", ""))),
                    model_name=model_name,
                )))
                continue

            # Need to unlearn this step
            ctx = _make_ctx(row)
            ctx.step_span = (span_start, span_end)
            unlearn_result = metric.unlearn_step_only(ctx, save_path=sd_path)
            pending_steps.append((idx, row, unlearn_result))
            n_unlearned += 1

        pbar.close()
        log.info("Batch %d/%d Stage 1b: %d unlearned, %d reused from disk",
                 batch_idx + 1, num_batches, n_unlearned, len(batch) - n_unlearned)

        if not pending_steps:
            continue

        # ── Stage 2: Generate with vLLM subprocess ──
        # Build manifest for the subprocess
        manifest = {
            "model_name": model_name,
            "tensor_parallel_size": profile.gpu_count,
            "gpu_memory_utilization": config.gpu_memory_utilization,
            "max_model_len": max_model_len,
            "steps": [],
        }
        for idx, row, unlearn_result in pending_steps:
            if unlearn_result.skipped:
                continue  # handled by parent after subprocess completes
            prompt_str = build_prompt(
                tokenizer=tokenizer,
                question=unlearn_result.question,
                system_message=system_message,
                enable_thinking=profile.enable_thinking,
                chat_template_kwargs=profile.chat_template_kwargs,
            )
            manifest["steps"].append({
                "state_dict_path": unlearn_result.state_dict_path,
                "prompt_str": prompt_str,
                "max_new_tokens": profile.max_tokens,
                "temperature": temperature,
                "top_p": top_p,
                "top_k": top_k,
                "skipped": unlearn_result.skipped,
                "num_samples": config.num_generation_samples,
            })

        manifest_path = os.path.join(save_dir, "manifest.json")
        with open(manifest_path, "w") as f:
            json.dump(manifest, f)
        log.info("── Batch %d/%d Stage 2: Launching vLLM subprocess for %d steps ──",
                 batch_idx + 1, num_batches, len(manifest["steps"]))

        # Free GPU memory before launching subprocess — the parent's CUDA
        # context holds memory that the subprocess's vLLM workers would see
        # as "used" (they share the same physical GPUs).
        model_ref[0].to("cpu")
        model_ref[0] = None  # release reference for GC
        import gc
        gc.collect()
        torch.cuda.empty_cache()
        log.info("GPU memory freed for vLLM subprocess")

        # Launch fresh subprocess — clean CUDA context, full GPU memory
        subprocess.run(
            [sys.executable, "-m", "metrics.fur.vllm_gen", "--manifest", manifest_path],
            check=True,
        )
        log.info("vLLM subprocess completed")

        # Read results and write to CSV
        result_paths_to_clean = []
        for idx, row, unlearn_result in pending_steps:
            result_path = unlearn_result.state_dict_path.replace(".pt", "_result.json")

            if unlearn_result.skipped:
                step_result = FURStepResult(
                    step_text=unlearn_result.step_text,
                    char_start=unlearn_result.char_start,
                    char_end=unlearn_result.char_end,
                    original_answer=unlearn_result.original_answer,
                    new_answer=unlearn_result.original_answer,
                    answer_changed=False,
                )
                score = 0.0
                gen_wall_time = 0.0
                votes = []
                all_answers = []
            else:
                with open(result_path) as f:
                    gen_result = json.load(f)
                gen_wall_time = gen_result.get("wall_time_s", 0.0)

                # Support both old (raw_output) and new (raw_outputs) format
                if "raw_outputs" in gen_result:
                    raw_outputs = gen_result["raw_outputs"]
                elif "raw_output" in gen_result:
                    raw_outputs = [gen_result["raw_output"]]
                else:
                    raise KeyError(f"FUR gen result missing 'raw_outputs' and 'raw_output': "
                                   f"{list(gen_result.keys())}")

                # Parse each sample and majority vote on answer_changed
                votes = []
                all_answers = []
                first_cot = ""
                for k, raw_output in enumerate(raw_outputs):
                    new_cot, new_answer = split_thinking(raw_output, profile.thinking_tag)
                    changed = not answers_match(new_answer, unlearn_result.original_answer)
                    votes.append(changed)
                    all_answers.append(new_answer)
                    if k == 0:
                        first_cot = new_cot

                answer_changed = sum(votes) > len(votes) / 2  # majority vote

                step_result = FURStepResult(
                    step_text=unlearn_result.step_text,
                    char_start=unlearn_result.char_start,
                    char_end=unlearn_result.char_end,
                    original_answer=unlearn_result.original_answer,
                    new_answer=all_answers[0],
                    answer_changed=answer_changed,
                    new_cot=first_cot,
                )
                score = 1.0 if answer_changed else 0.0

            # wall_time_s = unlearning time + generation time (per step)
            total_wall_time = round(unlearn_result.wall_time_s + gen_wall_time, 3)

            result_row = dict(row)
            result_row.update({
                "metric": args.metric, "target_model": model_name,
                "score": score, "level": "step",
                "metadata": _serialize_metadata(step_result),
                "api_cost_usd": 0.0,
                "wall_time_s": total_wall_time,
                "includes_init": False,
            })
            results.append(result_row)
            _write_result(result_row)
            if len(votes) > 1:
                log.info("row %d [STEP span=(%d,%d)] score=%.4f  votes=%s  answers=%s",
                         idx, unlearn_result.char_start, unlearn_result.char_end,
                         score, votes, all_answers[:3])
            else:
                log.info("row %d [STEP span=(%d,%d)] score=%.4f  changed=%s",
                         idx, unlearn_result.char_start, unlearn_result.char_end,
                         score, step_result.answer_changed)

            # Defer result JSON cleanup until all pending_steps are read,
            # because duplicate (row_id, span) entries share the same file.
            result_paths_to_clean.append(result_path)

        # Clean up result JSONs for this batch
        for p in set(result_paths_to_clean):
            try:
                os.remove(p)
            except OSError:
                pass
        # Clean up manifest for this batch
        try:
            os.remove(manifest_path)
        except OSError:
            pass

        # Reload HF model for next batch (skip on last batch — not needed)
        if not is_last_batch:
            log.info("Reloading HF model for next batch...")
            from transformers import AutoModelForCausalLM
            _load_kw = load_kwargs or {}
            if _on_low_mem_node():
                log.info("On n-80X/n-60X node — forcing SDPA attention (FA2 bug workaround)")
                new_model = AutoModelForCausalLM.from_pretrained(
                    model_name, attn_implementation="sdpa", **_load_kw,
                )
            else:
                try:
                    new_model = AutoModelForCausalLM.from_pretrained(
                        model_name, attn_implementation="flash_attention_2", **_load_kw,
                    )
                except (ImportError, ValueError):
                    new_model = AutoModelForCausalLM.from_pretrained(model_name, **_load_kw)
            new_model.eval()
            model_ref[0] = new_model
            log.info("HF model reloaded for batch %d/%d", batch_idx + 2, num_batches)

    # Clean up save directory
    try:
        os.rmdir(save_dir)
        log.info("Cleaned up FF2 directory: %s", save_dir)
    except OSError:
        remaining = os.listdir(save_dir) if os.path.isdir(save_dir) else []
        if remaining:
            log.info("FF2 directory not empty (%d files remain): %s", len(remaining), save_dir)

    # Summary
    elapsed = time.time() - t0
    n_new = len(results)
    n_resumed = len(completed_keys)
    log.info("=" * 60)
    log.info("DONE (two-stage) — %d new + %d resumed = %d total in %.1fs",
             n_new, n_resumed, n_new + n_resumed, elapsed)
    if n_new > 0:
        scores = [r["score"] for r in results]
        log.info("Score stats: mean=%.4f, std=%.4f, min=%.4f, max=%.4f",
                 sum(scores) / len(scores),
                 (sum((s - sum(scores)/len(scores))**2 for s in scores) / len(scores))**0.5,
                 min(scores), max(scores))
    log.info("Saved → %s", output_path)
    log.info("=" * 60)


def _run_monitor_batched(
    metric, config, scorable_rows, output_path,
    _csv_header_written, _csv_columns, completed_keys, args,
):
    """Single-batch path for the LLM-judge monitor metric.

    Builds one prompt per scorable row, fires them all via
    judge.run_batch(max_concurrency=judge_batch_size), then writes results
    incrementally so resume works.
    """
    from metrics.base import MetricContext
    from metrics.monitor.metric import MonitorMetric, MonitorResult

    t0 = time.time()

    if not scorable_rows:
        log.info("Monitor: no rows to score")
        return

    # Build (ctx, prompt, is_step, label_type, idx, row) tuples
    prepared = []
    for idx, row, is_step, is_cot, label_type in scorable_rows:
        ctx = MetricContext(
            question=str(row.get("prompt", row.get("question", ""))),
            cot=str(row.get("cot", "")),
            answer=str(row.get("model_answer", row.get("answer", ""))),
        )
        hint = str(row.get("prompted_hint", "") or "")
        if hint:
            ctx.extras["prompted_hint"] = hint
        if is_step:
            span_start = int(row.get("sentence_span_start", 0))
            span_end = int(row.get("sentence_span_end", 0))
            if span_end <= span_start:
                raise ValueError(
                    f"Row {idx} has invalid step span ({span_start}, {span_end})"
                )
            ctx.step_span = (span_start, span_end)
            prompt = metric.build_step_prompt(ctx)
        else:
            prompt = metric.build_cot_prompt(ctx)
        prepared.append((idx, row, is_step, label_type, prompt))

    prompts = [p[-1] for p in prepared]
    log.info("Monitor: submitting %d prompts to judge %s (max_concurrency=%d)",
             len(prompts), config.judge_model, config.judge_batch_size)

    judge = metric._get_judge()
    responses = judge.run_batch(
        prompts,
        max_concurrency=config.judge_batch_size,
        use_tqdm=True,
        max_output_tokens=config.judge_max_output_tokens,
    )

    total_api_cost = 0.0
    n_new = 0

    for (idx, row, is_step, label_type, prompt), response in zip(prepared, responses):
        row_t0 = time.time()
        result, final_response = type(metric).parse_with_retry(
            judge, prompt, response,
            max_output_tokens=config.judge_max_output_tokens,
        )
        row_cost = judge.get_request_cost(final_response)
        result.api_cost_usd = row_cost
        result.judge_prompt = MonitorMetric.abbreviate_cot_in_prompt(
            prompt, str(row.get("cot", "")),
        )
        total_api_cost += row_cost

        score = float(result.faithful)
        result_row = dict(row)
        result_row["metric"] = args.metric
        result_row["target_model"] = args.model
        result_row["score"] = score
        result_row["level"] = "step" if is_step else "cot"
        result_row["metadata"] = _serialize_metadata(result)
        result_row["api_cost_usd"] = round(row_cost, 6)
        result_row["wall_time_s"] = round(time.time() - row_t0, 3)
        result_row["includes_init"] = (n_new == 0 and len(completed_keys) == 0)

        if is_step:
            log.info("row %d [STEP span=(%s,%s)] score=%.1f",
                     idx, row.get("sentence_span_start"),
                     row.get("sentence_span_end"), score)
        else:
            log.info("row %d [COT] score=%.1f", idx, score)

        if _csv_columns is None:
            _csv_columns = list(result_row.keys())
        row_df = pd.DataFrame([result_row], columns=_csv_columns)
        row_df.to_csv(output_path, mode="a", header=not _csv_header_written, index=False)
        _csv_header_written = True
        n_new += 1

    elapsed = time.time() - t0
    n_resumed = len(completed_keys)
    log.info("=" * 60)
    log.info("DONE (monitor) — %d new + %d resumed = %d total in %.1fs",
             n_new, n_resumed, n_new + n_resumed, elapsed)
    if total_api_cost > 0:
        log.info("Total API cost (this run): $%.4f", total_api_cost)
    log.info("Saved → %s", output_path)
    log.info("=" * 60)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run a single faithfulness metric on labeled data",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model", required=True, help="HuggingFace model name")
    parser.add_argument("--input", required=True, help="Input CSV (merged labels + generation)")
    parser.add_argument("--output", required=True, help="Output CSV for metric results")
    parser.add_argument("--metric", required=True, help="Metric to run (e.g. early_answering)")
    parser.add_argument("--backend", default="vllm", choices=["vllm", "hf"],
                        help="Inference backend")
    parser.add_argument("--tensor_parallel_size", type=int, default=1,
                        help="GPUs for tensor parallelism (vLLM)")
    parser.add_argument("--scoring-levels", default="both",
                        choices=["both", "step_only", "cot_only"],
                        help="Which row types to score: 'both', 'step_only', or 'cot_only'")
    parser.add_argument("--fur-greedy-generation", action="store_true", default=False,
                        help="Force greedy decoding for FUR post-unlearning generation "
                             "(not recommended for thinking models)")
    parser.add_argument("--fur-vllm-generation", action="store_true", default=False,
                        help="Use vLLM for FUR step generation (two-stage: unlearn all "
                             "steps first with HF, then generate all with vLLM)")
    parser.add_argument("--fur-shm-pipeline", action="store_true", default=False,
                        help="Pipeline FUR state dict saves via /dev/shm for faster I/O")
    parser.add_argument("--fur-batch-size", type=int, default=10,
                        help="Default/max steps per batch in FUR two-stage pipeline")
    parser.add_argument("--fur-auto-batch", action=argparse.BooleanOptionalAction, default=True,
                        help="Auto-size batch from available disk space, capped by --fur-batch-size "
                             "(disable with --no-fur-auto-batch)")
    parser.add_argument("--cc-shap-batch-size", type=int, default=64,
                        help="Default/max samples per CC-SHAP forward pass (cap for auto-batch)")
    parser.add_argument("--cc-shap-auto-batch", action=argparse.BooleanOptionalAction, default=True,
                        help="Auto-size CC-SHAP batch from free GPU memory, capped by "
                             "--cc-shap-batch-size (disable with --no-cc-shap-auto-batch)")
    parser.add_argument("--fur-gpu-mem-util", type=float, default=0.9,
                        help="vLLM gpu_memory_utilization for FUR post-unlearning generation "
                             "(lower leaves headroom for leaked CUDA memory; default 0.9)")
    parser.add_argument("--fur-compute-validation", action="store_true", default=False,
                        help="Compute per-step efficacy (paper Eq. 2) and specificity proxy (Eq. 3) "
                             "after each unlearning step; stored in metadata column")
    parser.add_argument("--fur-validation-only", action="store_true", default=False,
                        help="Skip post-unlearning generation; only compute efficacy/specificity. "
                             "Implies --fur-compute-validation and disables --fur-vllm-generation. "
                             "Much faster — useful for verifying unlearning is working.")
    parser.add_argument("--fur-mmlu-check", action="store_true", default=False,
                        help="Run a 10-question MMLU sanity check before/after each step's unlearning "
                             "(paper §6.1 'Gen'). Reported as `mmlu_agreement` in metadata. "
                             "Requires --fur-compute-validation.")
    parser.add_argument("--max-labels", type=int, default=-1,
                        help="Max labels to evaluate (-1 = no limit)")
    parser.add_argument("--judge-model", default="gemini-3-flash-preview",
                        help="Gemini model name for the monitor LLM judge")
    parser.add_argument("--judge-batch-size", type=int, default=80,
                        help="Max concurrent Gemini API requests for the monitor metric")
    parser.add_argument("--regex_pattern", default=None,
                        help="Regex pattern for regex_baseline hint mode (default: 'hint', case-insensitive)")
    args = parser.parse_args()

    log.info("=" * 60)
    log.info("Metric: %s | Model: %s | Backend: %s", args.metric, args.model, args.backend)
    log.info("Input: %s", args.input)
    log.info("Output: %s", args.output)
    log.info("=" * 60)

    _register_metrics()

    if args.metric not in METRIC_REGISTRY:
        parser.error(f"Unknown metric: {args.metric!r}. Available: {list(METRIC_REGISTRY)}")

    metric_cls, config_cls = METRIC_REGISTRY[args.metric]
    config = _build_config(args.metric, config_cls, args)
    metric = metric_cls(config=config)
    log.info("Initialized %s (supports_cot=%s, supports_step=%s, requires_weights=%s)",
             metric.name, metric.supports_cot_scoring, metric.supports_step_scoring,
             metric.requires_model_weights)

    # Load input data
    df = pd.read_csv(args.input)
    log.info("Loaded %d rows from %s", len(df), args.input)

    # Summarize label types
    label_counts = df["label_type"].value_counts().to_dict() if "label_type" in df.columns else {}
    log.info("Label distribution: %s", label_counts)

    # Load tokenizer (needed by MetricContext). Skip for judge-only metrics
    # that never touch model weights (e.g. monitor).
    _NO_TOKENIZER_METRICS = {"monitor", "monitor_no_hint", "monitor_generic", "monitor_no_tool", "regex_baseline"}
    if args.metric in _NO_TOKENIZER_METRICS:
        tokenizer = None
        log.info("Skipping tokenizer load (not needed for %s)", args.metric)
    else:
        from transformers import AutoTokenizer
        log.info("Loading tokenizer: %s", args.model)
        tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        log.info("Tokenizer ready (vocab_size=%d)", tokenizer.vocab_size)

    # Only load HF model when the metric needs raw weights AND we're using HF backend.
    # Metrics like FUR/CC-SHAP always need HF weights (they do direct forward passes).
    # Generation-based metrics (early_answering, adding_mistakes, etc.) only need HF
    # weights when backend="hf" — with vllm, the engine loads the model itself.
    _ALWAYS_NEEDS_HF = {"fur", "cc_shap"}
    need_hf_model = (
        args.metric in _ALWAYS_NEEDS_HF
        or (metric.requires_model_weights and args.backend == "hf")
    )
    hf_model = None
    if need_hf_model:
        import torch
        from transformers import AutoModelForCausalLM

        _patch_rocm_grouped_mm()

        # Enable PyTorch native Flash SDP attention (works without flash-attn package)
        if hasattr(torch.backends.cuda, "enable_flash_sdp"):
            torch.backends.cuda.enable_flash_sdp(True)
            log.info("Enabled PyTorch native Flash SDP attention")

        # Try Flash Attention 2, fall back to default attention
        load_kwargs = dict(
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            device_map="auto",
        )
        if _on_low_mem_node():
            log.info("On n-80X/n-60X node — forcing SDPA attention (FA2 bug workaround): %s", args.model)
            hf_model = AutoModelForCausalLM.from_pretrained(
                args.model, attn_implementation="sdpa", **load_kwargs,
            )
        else:
            try:
                log.info("Loading HF model with flash_attention_2: %s", args.model)
                hf_model = AutoModelForCausalLM.from_pretrained(
                    args.model, attn_implementation="flash_attention_2", **load_kwargs,
                )
            except (ImportError, ValueError) as e:
                log.warning("Flash Attention 2 unavailable (%s), using default attention", e)
                hf_model = AutoModelForCausalLM.from_pretrained(args.model, **load_kwargs)

        hf_model.eval()
        log.info("HF model loaded (%.1fM params)",
                 sum(p.numel() for p in hf_model.parameters()) / 1e6)

        # Set TF32 for faster matmuls on Ampere+ GPUs (lossless for bf16 training)
        torch.set_float32_matmul_precision("high")
    else:
        log.info("Skipping HF model load (not needed for %s with backend=%s)",
                 args.metric, args.backend)

    # Build other_instances for metrics that need it (FUR, SCM)
    other_instances = [
        {"question": str(r.get("prompt", "")), "cot": str(r.get("cot", "")),
         "answer": str(r.get("model_answer", ""))}
        for _, r in df.iterrows()
    ]

    from metrics.base import MetricContext

    # Pre-filter rows to scorable ones
    scorable_rows = []
    skipped = {"unsupported_step": 0, "unsupported_cot": 0, "unknown_label": 0,
               "cli_skip_step": 0, "cli_skip_cot": 0}
    for idx, row in df.iterrows():
        label_type = str(row.get("label_type", ""))
        is_step = label_type.endswith("STEP")
        is_cot = label_type.endswith("COT")

        if is_step and not metric.supports_step_scoring:
            skipped["unsupported_step"] += 1
            continue
        if is_cot and not metric.supports_cot_scoring:
            skipped["unsupported_cot"] += 1
            continue
        if is_step and args.scoring_levels == "cot_only":
            skipped["cli_skip_step"] += 1
            continue
        if is_cot and args.scoring_levels == "step_only":
            skipped["cli_skip_cot"] += 1
            continue
        if not is_step and not is_cot:
            skipped["unknown_label"] += 1
            continue
        scorable_rows.append((idx, row, is_step, is_cot, label_type))

    if any(v > 0 for v in skipped.values()):
        log.info("Skipped rows: %s", {k: v for k, v in skipped.items() if v > 0})
    log.info("Scoring %d rows", len(scorable_rows))

    results = []
    total_api_cost = 0.0

    # Resume support: load already-scored rows from existing output CSV
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    completed_keys: set[tuple] = set()
    if output_path.exists():
        try:
            existing = pd.read_csv(output_path)
            for _, erow in existing.iterrows():
                key = (
                    str(erow.get("row_id", "")),
                    str(erow.get("label_type", "")),
                    str(erow.get("sentence_span_start", "")),
                    str(erow.get("sentence_span_end", "")),
                )
                completed_keys.add(key)
            _csv_columns = list(existing.columns)
            log.info("Resuming: found %d already-scored rows in %s", len(completed_keys), output_path)
        except Exception as e:
            log.warning("Could not read existing output for resume (%s), starting fresh", e)

    _csv_header_written = output_path.exists() and len(completed_keys) > 0
    if not _csv_header_written:
        _csv_columns = None  # Will be set from first result row

    # Filter out already-scored rows before the loop so tqdm count is accurate
    if completed_keys:
        before = len(scorable_rows)
        scorable_rows = [
            (idx, row, is_step, is_cot, label_type)
            for idx, row, is_step, is_cot, label_type in scorable_rows
            if (
                str(row.get("row_id", "")),
                str(label_type),
                str(row.get("sentence_span_start", "")),
                str(row.get("sentence_span_end", "")),
            ) not in completed_keys
        ]
        log.info("Resuming: %d rows remaining (%d already scored)", len(scorable_rows), before - len(scorable_rows))

    if args.max_labels >= 0:
        scorable_rows = scorable_rows[:args.max_labels]
        log.info("Limiting to %d labels (--max-labels)", args.max_labels)

    # Two-stage FUR+vLLM: unlearn all steps with HF, then generate with vLLM.
    # Only applies to STEP rows; COT rows use normal HF flow.
    _fur_vllm = (
        args.metric == "fur"
        and hasattr(config, "use_vllm_generation")
        and config.use_vllm_generation
    )
    if args.metric in ("monitor", "monitor_no_hint", "monitor_generic", "monitor_no_tool"):
        _run_monitor_batched(
            metric=metric,
            config=config,
            scorable_rows=scorable_rows,
            output_path=output_path,
            _csv_header_written=_csv_header_written,
            _csv_columns=_csv_columns,
            completed_keys=completed_keys,
            args=args,
        )
        return

    if _fur_vllm:
        _run_fur_two_stage(
            metric=metric,
            config=config,
            scorable_rows=scorable_rows,
            hf_model=hf_model,
            tokenizer=tokenizer,
            model_name=args.model,
            other_instances=other_instances,
            df=df,
            output_path=output_path,
            _csv_header_written=_csv_header_written,
            _csv_columns=_csv_columns,
            completed_keys=completed_keys,
            args=args,
            load_kwargs=load_kwargs,
        )
        return

    t0 = time.time()

    pbar = tqdm(scorable_rows, desc=f"{args.metric}", unit="row",
                bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]")

    for idx, row, is_step, is_cot, label_type in pbar:
        row_t0 = time.time()

        ctx = MetricContext(
            question=str(row.get("prompt", row.get("question", ""))),
            cot=str(row.get("cot", "")),
            answer=str(row.get("model_answer", row.get("answer", ""))),
            model=hf_model,
            tokenizer=tokenizer,
            model_name=args.model,
            other_instances=other_instances,
        )

        from metrics.shared.row_extras import populate_extras_from_row
        populate_extras_from_row(ctx, row)

        # Start with all input columns, then add metric-specific columns
        result_row = dict(row)
        result_row["metric"] = args.metric
        result_row["target_model"] = args.model

        # For COT rows: collect FAITHFUL_STEP spans from the same question
        # so metrics can prioritize those steps (e.g., FUR early stopping).
        if is_cot and "question_index" in row:
            q_idx = row["question_index"]
            faithful_spans = set()
            for _, srow in df[
                (df["question_index"] == q_idx)
                & (df["label_type"] == "FAITHFUL_STEP")
            ].iterrows():
                s = int(srow.get("sentence_span_start", 0))
                e = int(srow.get("sentence_span_end", 0))
                if e > s:
                    faithful_spans.add((s, e))
            if faithful_spans:
                ctx.extras["priority_step_spans"] = faithful_spans

        if is_step:
            span_start = int(row.get("sentence_span_start", 0))
            span_end = int(row.get("sentence_span_end", 0))
            if span_end <= span_start:
                raise ValueError(
                    f"Row {idx} has invalid step span ({span_start}, {span_end})"
                )
            ctx.step_span = (span_start, span_end)
            # Use score_step_detailed if available, otherwise bare score
            if hasattr(metric, "score_step_detailed"):
                detail = metric.score_step_detailed(ctx)
                score = _extract_score(detail, args.metric)
                result_row["metadata"] = _serialize_metadata(detail)
                row_cost = getattr(detail, "api_cost_usd", 0.0)
            else:
                score = metric.score_step(ctx)
                result_row["metadata"] = _serialize_metadata({"score": score})
                row_cost = 0.0
            result_row["score"] = score
            result_row["level"] = "step"
            log.info("row %d [STEP span=(%d,%d)] score=%.4f", idx, span_start, span_end, score)
        else:
            detail = metric.score_cot_detailed(ctx)
            score = _extract_score(detail, args.metric)
            result_row["score"] = score
            result_row["metadata"] = _serialize_metadata(detail)
            result_row["level"] = "cot"
            row_cost = getattr(detail, "api_cost_usd", 0.0)
            cot_preview = str(row.get("cot", ""))[:60].replace("\n", " ")
            log.info("row %d [COT] score=%.4f  cot=\"%s...\"", idx, score, cot_preview)

        result_row["api_cost_usd"] = round(row_cost, 6)
        result_row["wall_time_s"] = round(time.time() - row_t0, 3)
        result_row["includes_init"] = len(results) == 0 and len(completed_keys) == 0
        total_api_cost += row_cost
        pbar.set_postfix(cost=f"${total_api_cost:.4f}")
        results.append(result_row)

        # Write incrementally — each row appended immediately so progress
        # survives crashes. Header written only on first row of a fresh run.
        # Enforce consistent column order so appended rows align with header.
        if _csv_columns is None:
            _csv_columns = list(result_row.keys())
        row_df = pd.DataFrame([result_row], columns=_csv_columns)
        row_df.to_csv(output_path, mode="a", header=not _csv_header_written, index=False)
        _csv_header_written = True

    pbar.close()
    elapsed = time.time() - t0

    # Summary
    n_new = len(results)
    n_resumed = len(completed_keys)
    n_total = n_new + n_resumed
    log.info("=" * 60)
    log.info("DONE — %d new + %d resumed = %d total in %.1fs (%.2f s/row for new)",
             n_new, n_resumed, n_total, elapsed,
             elapsed / n_new if n_new > 0 else 0)
    if n_new > 0:
        scores = [r["score"] for r in results]
        log.info("Score stats (new rows): mean=%.4f, std=%.4f, min=%.4f, max=%.4f",
                 sum(scores) / len(scores),
                 (sum((s - sum(scores)/len(scores))**2 for s in scores) / len(scores))**0.5,
                 min(scores), max(scores))
    if total_api_cost > 0:
        log.info("Total API cost (this run): $%.4f", total_api_cost)
    log.info("Saved → %s", output_path)
    log.info("=" * 60)


def _extract_score(detail, metric_name: str) -> float:
    """Extract the primary score from a metric's detailed result object."""
    # Each metric returns a different detailed result type — map to primary score
    if hasattr(detail, "aoc"):
        return detail.aoc  # early_answering, adding_mistakes, filler_tokens
    if hasattr(detail, "match_rate"):
        return detail.match_rate  # paraphrasing
    if hasattr(detail, "score"):
        return detail.score  # simulatability, filler_tokens
    if hasattr(detail, "cc_shap_score"):
        return detail.cc_shap_score  # cc_shap
    if hasattr(detail, "ff_hard"):
        return detail.ff_hard  # fur (cot-level)
    if hasattr(detail, "answer_changed"):
        return 1.0 if detail.answer_changed else 0.0  # fur (step-level)
    if hasattr(detail, "faithful"):
        return float(detail.faithful)  # scm, generic fallback
    raise ValueError(f"Cannot extract score from {type(detail).__name__} for metric {metric_name}")


def _serialize_metadata(detail) -> str:
    """Serialize a metric's detailed result to a JSON string for debugging.

    Handles dataclasses, dicts, and falls back to str() for unknown types.
    """
    try:
        if is_dataclass(detail) and not isinstance(detail, type):
            data = asdict(detail)
        elif isinstance(detail, dict):
            data = detail
        else:
            data = {"score": detail}
        return json.dumps(data, default=str, ensure_ascii=False)
    except Exception as e:
        return json.dumps({"_serialization_error": str(e)})


if __name__ == "__main__":
    main()
