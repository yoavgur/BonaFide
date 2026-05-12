"""Backend classes for vLLM and HuggingFace generation."""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import torch

if TYPE_CHECKING:
    from generation.config import GenerationConfig


@dataclass
class GenerationResult:
    """Result for a single generated row."""

    row_index: int
    raw_output: str
    finish_reason: str  # "stop", "length", etc.
    num_tokens: int
    seed_used: int


def _row_seed(base_seed: int, row_index: int) -> int:
    """Deterministic per-row seed."""
    return base_seed + row_index


def _set_all_seeds(seed: int) -> None:
    """Reset all RNG states for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_prompt(
    tokenizer,
    question: str,
    system_message: str | None,
    enable_thinking: bool,
    chat_template_kwargs: dict,
) -> str:
    """Build a formatted prompt string using the tokenizer's chat template."""
    messages = []
    if system_message:
        messages.append({"role": "system", "content": system_message})
    messages.append({"role": "user", "content": question})

    template_kwargs = {
        "add_generation_prompt": True,
        "tokenize": False,
    }
    if enable_thinking:
        template_kwargs["enable_thinking"] = True
    template_kwargs.update(chat_template_kwargs)

    return tokenizer.apply_chat_template(messages, **template_kwargs)


class VLLMBackend:
    """vLLM-based generation with per-request deterministic seeds."""

    def __init__(self, config: GenerationConfig) -> None:
        from vllm import LLM

        dtype_map = {"bfloat16": "bfloat16", "float16": "float16", "float32": "float32"}
        dtype = dtype_map.get(config.dtype, config.dtype)

        llm_kwargs = dict(
            model=config.model,
            tensor_parallel_size=config.tensor_parallel_size,
            dtype=dtype,
            gpu_memory_utilization=config.gpu_memory_utilization,
            trust_remote_code=config.trust_remote_code,
        )
        if config.max_model_len is not None:
            llm_kwargs["max_model_len"] = config.max_model_len

        self.llm = LLM(**llm_kwargs)
        self.tokenizer = self.llm.get_tokenizer()
        self.config = config

    def generate_all(
        self, prompts: list[str], row_indices: list[int]
    ) -> list[GenerationResult]:
        """Generate all prompts in one vLLM call with per-request seeds."""
        from vllm import SamplingParams

        # Build per-request sampling params with unique seeds
        sampling_params_list = [
            SamplingParams(
                seed=_row_seed(self.config.seed, idx),
                temperature=self.config.temperature,
                top_p=self.config.top_p,
                top_k=self.config.top_k,
                max_tokens=self.config.max_tokens,
            )
            for idx in row_indices
        ]

        outputs = self.llm.generate(prompts, sampling_params=sampling_params_list)

        results = []
        for output, idx in zip(outputs, row_indices):
            seed = _row_seed(self.config.seed, idx)
            if not output.outputs:
                raise RuntimeError(
                    f"vLLM returned no outputs for row index {idx} (seed={seed})"
                )

            completion = output.outputs[0]
            results.append(
                GenerationResult(
                    row_index=idx,
                    raw_output=completion.text,
                    finish_reason=completion.finish_reason or "unknown",
                    num_tokens=len(completion.token_ids),
                    seed_used=seed,
                )
            )

        return results


class HFBackend:
    """HuggingFace transformers generation with per-row seed resets."""

    def __init__(self, config: GenerationConfig) -> None:
        from transformers import AutoModelForCausalLM, AutoTokenizer

        dtype_map = {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
        }
        torch_dtype = dtype_map.get(config.dtype, torch.bfloat16)

        self.tokenizer = AutoTokenizer.from_pretrained(
            config.model, trust_remote_code=config.trust_remote_code
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            config.model,
            torch_dtype=torch_dtype,
            device_map="auto",
            trust_remote_code=config.trust_remote_code,
        )
        self.config = config

        # Enforce deterministic CUDA operations
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    def generate_all(
        self, prompts: list[str], row_indices: list[int]
    ) -> list[GenerationResult]:
        """Generate prompts one-at-a-time with per-row seed reset."""
        results = []
        for prompt, idx in zip(prompts, row_indices):
            seed = _row_seed(self.config.seed, idx)
            _set_all_seeds(seed)

            inputs = self.tokenizer(prompt, return_tensors="pt").to(
                self.model.device
            )
            input_len = inputs["input_ids"].shape[-1]

            with torch.no_grad():
                output_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=self.config.max_tokens,
                    temperature=self.config.temperature,
                    top_p=self.config.top_p,
                    top_k=self.config.top_k if self.config.top_k > 0 else None,
                    do_sample=True,
                )

            new_ids = output_ids[0][input_len:]
            raw_output = self.tokenizer.decode(
                new_ids, skip_special_tokens=True
            )
            finish_reason = (
                "length" if len(new_ids) >= self.config.max_tokens else "stop"
            )

            results.append(
                GenerationResult(
                    row_index=idx,
                    raw_output=raw_output,
                    finish_reason=finish_reason,
                    num_tokens=len(new_ids),
                    seed_used=seed,
                )
            )

        return results
