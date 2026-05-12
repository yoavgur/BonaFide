"""CLI entry point: python -m generation"""

from __future__ import annotations

import argparse
from pathlib import Path

from generation.config import GenerationConfig
from generation.generate import run


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Reproducible LLM generation pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Required
    parser.add_argument("--model", required=True, help="HuggingFace model name or path")
    parser.add_argument("--input", required=True, help="Path to input CSV/parquet with questions")

    # Output (optional — auto-derived as ./faithbench/generation/{model}_{dataset}.csv)
    parser.add_argument("--output", default=None,
                        help="Output CSV path. If not set, auto-derived as "
                             "./faithbench/generation/{model}_{dataset}.csv")

    # Backend
    parser.add_argument("--backend", default="vllm", choices=["vllm", "hf"], help="Inference backend")
    parser.add_argument("--tensor_parallel_size", type=int, default=None, help="GPUs for tensor parallelism (vLLM)")
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.9, help="vLLM GPU memory fraction")
    parser.add_argument("--max_model_len", type=int, default=None, help="Override max model context length")

    # Sampling (defaults are None = use model profile)
    parser.add_argument("--seed", type=int, default=42, help="Base seed for reproducibility")
    parser.add_argument("--temperature", type=float, default=None, help="Sampling temperature")
    parser.add_argument("--top_p", type=float, default=None, help="Nucleus sampling p")
    parser.add_argument("--top_k", type=int, default=None, help="Top-k sampling (-1 = disabled)")
    parser.add_argument("--max_tokens", type=int, default=None, help="Max new tokens to generate")

    # Model loading
    parser.add_argument("--dtype", default=None, choices=["bfloat16", "float16", "float32"], help="Model dtype")
    parser.add_argument("--trust_remote_code", action="store_true", default=None, help="Trust remote code")

    # CoT / thinking
    parser.add_argument("--thinking_tag", default=None, help="Tag for reasoning blocks (e.g., 'think'). Use 'none' to disable.")
    parser.add_argument("--enable_thinking", action="store_true", default=None, help="Enable thinking in chat template (Qwen3)")
    parser.add_argument("--no_inject_cot_prompt", action="store_true",
                        help="Disable injecting 'think step by step' CoT instruction into system message")

    # Data columns
    parser.add_argument("--prompt_column", default="prompt", help="Column name for prompts")
    parser.add_argument("--system_message_column", default="system_message", help="Column name for per-row system messages")
    parser.add_argument("--id_column", default="id", help="Column name for row IDs")
    parser.add_argument("--dataset_column", default="dataset", help="Column name for dataset name (used in output path)")

    # Execution
    parser.add_argument("--batch_size", type=int, default=0,
                        help="Process prompts in chunks of this size with incremental saves. "
                             "0 = send all at once (vLLM handles internal batching). "
                             "Useful for large datasets to get incremental saves and limit Python-side memory.")

    # Resume
    parser.add_argument("--resume", action="store_true", help="Skip already-completed rows")

    # No-CoT mode
    parser.add_argument("--no_cot", action="store_true",
                        help="Suppress chain-of-thought: close <think> with '.' "
                             "and prefill the JSON answer start. Requires an "
                             "'answer_key' column in the input CSV "
                             "('final_answer' or 'final_node').")

    args = parser.parse_args()

    # Track which args were explicitly set via CLI (vs left as None/default)
    cli_overrides: set[str] = set()
    for key, value in vars(args).items():
        if value is not None and key not in ("model", "input", "output", "backend", "resume",
                                              "no_inject_cot_prompt",
                                              "prompt_column", "system_message_column",
                                              "id_column", "dataset_column",
                                              "seed", "gpu_memory_utilization"):
            cli_overrides.add(key)
    if args.no_inject_cot_prompt:
        cli_overrides.add("inject_cot_prompt")

    # Handle --thinking_tag none
    thinking_tag = args.thinking_tag
    if thinking_tag == "none":
        thinking_tag = None

    # Build config
    config = GenerationConfig(
        model=args.model,
        input_path=str(Path(args.input).resolve()),
        backend=args.backend,
        seed=args.seed,
        gpu_memory_utilization=args.gpu_memory_utilization,
        batch_size=args.batch_size,
        resume=args.resume,
        prompt_column=args.prompt_column,
        system_message_column=args.system_message_column,
        id_column=args.id_column,
        dataset_column=args.dataset_column,
    )

    # Set output path if explicitly provided
    if args.output is not None:
        config.output_path = str(Path(args.output).resolve())

    # Apply CLI overrides (non-None values)
    if args.temperature is not None:
        config.temperature = args.temperature
    if args.top_p is not None:
        config.top_p = args.top_p
    if args.top_k is not None:
        config.top_k = args.top_k
    if args.max_tokens is not None:
        config.max_tokens = args.max_tokens
    if args.dtype is not None:
        config.dtype = args.dtype
    if args.trust_remote_code is not None:
        config.trust_remote_code = args.trust_remote_code
    if args.thinking_tag is not None:
        config.thinking_tag = thinking_tag
    if args.enable_thinking is not None:
        config.enable_thinking = args.enable_thinking
    if args.tensor_parallel_size is not None:
        config.tensor_parallel_size = args.tensor_parallel_size
    if args.max_model_len is not None:
        config.max_model_len = args.max_model_len
    if args.no_inject_cot_prompt:
        config.inject_cot_prompt = False
    if args.no_cot:
        config.no_cot = True

    run(config, cli_overrides)


if __name__ == "__main__":
    main()
