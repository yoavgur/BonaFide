"""Core orchestration: load data, run inference, save results."""

from __future__ import annotations

import csv
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from generation.backends import GenerationResult, VLLMBackend, HFBackend, build_prompt
from generation.config import GenerationConfig
from generation.model_registry import ANSWER_ONLY_SYSTEM_PROMPT, COT_SYSTEM_PROMPT, ModelProfile, get_model_profile
from generation.normalize import extract_answer_from_raw
from generation.thinking import split_thinking


def _build_no_cot_suffix(answer_key: str, thinking_tag: str | None, prompt_so_far: str) -> tuple[str, str]:
    """Build the suffix to append to the chat-template prompt for no-CoT mode.

    Returns (suffix, answer_prefix). answer_prefix is the JSON-start string
    that must be prepended to the raw model output before answer parsing.
    """
    answer_prefix = f'{{"{answer_key}": "'
    if thinking_tag:
        open_tag = f"<{thinking_tag}>"
        close_tag = f"</{thinking_tag}>"
        if prompt_so_far.rstrip().endswith(open_tag):
            # Chat template already opened the thinking block.
            suffix = f"\n.\n{close_tag}\n{answer_prefix}"
        else:
            suffix = f"{open_tag}\n.\n{close_tag}\n{answer_prefix}"
    else:
        suffix = answer_prefix
    return suffix, answer_prefix


def _apply_profile_defaults(config: GenerationConfig, profile: ModelProfile, cli_overrides: set[str]) -> None:
    """Apply model profile defaults for fields not explicitly set via CLI."""
    field_map = {
        "temperature": "temperature",
        "top_p": "top_p",
        "top_k": "top_k",
        "max_tokens": "max_tokens",
        "thinking_tag": "thinking_tag",
        "enable_thinking": "enable_thinking",
        "cot_pattern": "cot_pattern",
        "inject_cot_prompt": "inject_cot_prompt",
        "dtype": "dtype",
        "trust_remote_code": "trust_remote_code",
        "max_model_len": "max_model_len",
        "chat_template_kwargs": "chat_template_kwargs",
    }
    for cli_name, field_name in field_map.items():
        if cli_name not in cli_overrides:
            setattr(config, field_name, getattr(profile, field_name))


def _load_input(config: GenerationConfig) -> pd.DataFrame:
    """Load input CSV, validate columns, auto-generate ID if missing."""
    path = Path(config.input_path)
    if path.suffix == ".parquet":
        df = pd.read_parquet(path)
    else:
        df = pd.read_csv(path)

    if config.prompt_column not in df.columns:
        raise ValueError(
            f"Input file missing required column '{config.prompt_column}'. "
            f"Available columns: {list(df.columns)}"
        )

    # Auto-generate ID column if missing
    if config.id_column not in df.columns:
        df[config.id_column] = range(len(df))
        print(f"Auto-generated '{config.id_column}' column (0..{len(df) - 1})")

    config.num_rows = len(df)
    return df


def _derive_output_path(config: GenerationConfig, df: pd.DataFrame) -> str:
    """Derive output path as ./faithbench/generation/{model}_{dataset}.csv"""
    model_short = config.model.split("/")[-1]

    if config.dataset_column not in df.columns:
        raise ValueError(
            f"Input file missing required column '{config.dataset_column}' for output path derivation. "
            f"Available columns: {list(df.columns)}. "
            f"Set --dataset_column or provide --output explicitly."
        )

    dataset = str(df[config.dataset_column].iloc[0]).replace("/", "_").replace("\\", "_")
    return str(Path("faithbench") / "generation" / f"{model_short}_{dataset}.csv")


def _load_completed_ids(output_path: Path, id_column: str) -> set:
    """Load IDs of already-completed rows from existing results."""
    if not output_path.exists():
        return set()
    existing = pd.read_csv(output_path)
    if id_column in existing.columns:
        return set(existing[id_column].tolist())
    return set()


def _save_results(
    results_rows: list[dict],
    output_path: Path,
    append: bool,
) -> None:
    """Save results to CSV, appending if resuming."""
    if not results_rows:
        return

    fieldnames = list(results_rows[0].keys())

    if append and output_path.exists():
        with open(output_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writerows(results_rows)
    else:
        with open(output_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(results_rows)


def run(config: GenerationConfig, cli_overrides: set[str]) -> None:
    """Main generation pipeline."""
    # 1. Apply model profile defaults and validate
    profile = get_model_profile(config.model)
    _apply_profile_defaults(config, profile, cli_overrides)

    # Validate enable_thinking consistency with model type
    if "enable_thinking" in cli_overrides and config.enable_thinking != profile.enable_thinking:
        raise ValueError(
            f"--enable_thinking conflicts with model '{config.model}'. "
            f"This model has enable_thinking={profile.enable_thinking} in its profile. "
            f"Use a {'Thinking' if config.enable_thinking else 'Instruct'} model variant instead."
        )

    print(f"Model profile: {config.model}")
    print(f"  thinking_tag={config.thinking_tag}, enable_thinking={config.enable_thinking}")
    print(f"  temperature={config.temperature}, top_p={config.top_p}, top_k={config.top_k}")
    print(f"  max_tokens={config.max_tokens}, dtype={config.dtype}")

    # 2. Load input data (needed early for output path derivation)
    df = _load_input(config)
    print(f"Loaded {len(df)} rows from {config.input_path}")

    # 3. Derive output path if not explicitly set
    if not config.output_path:
        config.output_path = _derive_output_path(config, df)
        print(f"Auto-derived output path: {config.output_path}")

    output_path = Path(config.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # 4. Check for existing results before doing expensive setup
    if not config.resume and output_path.exists():
        raise FileExistsError(
            f"Output file already exists: {output_path}\n"
            f"Use --resume to continue from where you left off, "
            f"or choose a different --output path."
        )

    # 5. Populate environment info and save config
    config.generate_run_id()
    config.populate_environment()
    config.populate_backend_version()
    config.populate_input_hash()
    config.populate_model_revision()

    # 6. Resume: filter out completed rows (only if --resume is set)
    if config.resume:
        completed_ids = _load_completed_ids(output_path, config.id_column)
        if completed_ids:
            before = len(df)
            df = df[~df[config.id_column].isin(completed_ids)].reset_index(drop=True)
            print(f"Resuming: skipping {before - len(df)} already-completed rows, {len(df)} remaining")

        if len(df) == 0:
            print("All rows already completed. Nothing to do.")
            return

    # 7. Initialize backend
    print(f"Initializing {config.backend} backend...")
    t0 = time.time()
    if config.backend == "vllm":
        backend = VLLMBackend(config)
        tokenizer = backend.tokenizer
    else:
        backend = HFBackend(config)
        tokenizer = backend.tokenizer
    print(f"Backend initialized in {time.time() - t0:.1f}s")

    # 8. Build prompts
    has_system_col = config.system_message_column in df.columns
    has_answer_key_col = "answer_key" in df.columns
    if config.no_cot and not has_answer_key_col:
        raise ValueError(
            "no_cot=True requires an 'answer_key' column in the input CSV "
            "(values: 'final_answer' or 'final_node'). Re-run task generation "
            "with the no_cot_benchmark flow."
        )
    prompts = []
    row_indices = []
    answer_prefixes: list[str] = []  # Per-row JSON prefix; empty string when no_cot=False
    for row_num, (i, row) in enumerate(df.iterrows()):
        system_msg = None
        if has_system_col and pd.notna(row[config.system_message_column]):
            system_msg = str(row[config.system_message_column])
        # Append CoT/answer instruction to system message if enabled.
        # Skip in no_cot mode — the whole point is to deny the model any
        # reasoning, so a "think step by step" preamble would be inert
        # noise in the prompt.
        if config.inject_cot_prompt and not config.no_cot:
            # Thinking models: just need clean answer after reasoning
            # Non-thinking models: need step-by-step + final answer on last line
            cot_instruction = ANSWER_ONLY_SYSTEM_PROMPT if config.thinking_tag else COT_SYSTEM_PROMPT
            if system_msg:
                system_msg = f"{system_msg}\n\n{cot_instruction}"
            else:
                system_msg = cot_instruction
        prompt = build_prompt(
            tokenizer=tokenizer,
            question=row[config.prompt_column],
            system_message=system_msg,
            enable_thinking=config.enable_thinking,
            chat_template_kwargs=config.chat_template_kwargs,
        )
        if config.no_cot:
            answer_key = str(row["answer_key"])
            suffix, answer_prefix = _build_no_cot_suffix(
                answer_key, config.thinking_tag, prompt
            )
            prompt = prompt + suffix
            answer_prefixes.append(answer_prefix)
        else:
            answer_prefixes.append("")
        prompts.append(prompt)
        row_indices.append(row_num)

    # 9. Generate in chunks with incremental saves
    batch_size = config.batch_size if config.batch_size > 0 else len(prompts)
    num_batches = (len(prompts) + batch_size - 1) // batch_size
    if num_batches > 1:
        print(f"Generating {len(prompts)} responses in {num_batches} batches of {batch_size}...")
    else:
        print(f"Generating {len(prompts)} responses...")
    df_rows = list(df.iterrows())  # Materialize once for indexing into batches

    # Build reproducibility info (shared across all rows in this run)
    repro_base = {
        "run_id": config.run_id,
        "model": config.model,
        "model_revision": config.model_revision,
        "backend": config.backend,
        "backend_version": config.backend_version,
        "temperature": config.temperature,
        "top_p": config.top_p,
        "top_k": config.top_k,
        "max_tokens": config.max_tokens,
        "base_seed": config.seed,
    }

    total_tokens = 0
    total_rows = 0
    append = config.resume and output_path.exists()
    t0 = time.time()

    for batch_idx in range(num_batches):
        start = batch_idx * batch_size
        end = min(start + batch_size, len(prompts))
        batch_prompts = prompts[start:end]
        batch_indices = row_indices[start:end]
        batch_df_rows = df_rows[start:end]

        if num_batches > 1:
            print(f"  Batch {batch_idx + 1}/{num_batches} ({len(batch_prompts)} prompts)...")

        gen_results: list[GenerationResult] = backend.generate_all(batch_prompts, batch_indices)
        batch_prefixes = answer_prefixes[start:end]

        # Process batch results
        results_rows = []
        for gen_result, (_, row), answer_prefix in zip(gen_results, batch_df_rows, batch_prefixes):
            if config.no_cot:
                # No CoT was generated: the whole raw_output is the answer
                # continuation. Prepend the prefix so the JSON parser sees a
                # complete object.
                cot = ""
                full_for_parsing = answer_prefix + gen_result.raw_output
                answer = extract_answer_from_raw(full_for_parsing) or ""
            else:
                cot, answer = split_thinking(
                    gen_result.raw_output, config.thinking_tag,
                    config.cot_pattern,
                )

            # Start with all original input columns
            result_row = {}
            for col in df.columns:
                result_row[col] = row[col]

            # Add generation output columns
            result_row["cot"] = cot
            result_row["model_answer"] = answer
            result_row["model_raw_response"] = gen_result.raw_output
            result_row["target_model"] = config.model
            if config.no_cot:
                result_row["answer_prefix"] = answer_prefix

            # Bundle per-row reproducibility info as JSON
            repro = {
                **repro_base,
                "seed_used": gen_result.seed_used,
                "finish_reason": gen_result.finish_reason,
                "num_tokens": gen_result.num_tokens,
                "raw_output": gen_result.raw_output,
            }
            result_row["reproducibility_info"] = json.dumps(repro)

            results_rows.append(result_row)
            total_tokens += gen_result.num_tokens

        # Save this batch incrementally
        _save_results(results_rows, output_path, append=append)
        append = True  # All subsequent batches append
        total_rows += len(results_rows)

    gen_time = time.time() - t0

    # 10. Summary
    print(f"\nGeneration complete:")
    print(f"  Rows: {total_rows}")
    print(f"  Total tokens: {total_tokens:,}")
    print(f"  Time: {gen_time:.1f}s ({total_tokens / max(gen_time, 0.01):.0f} tok/s)")
    print(f"  Results: {output_path}")
