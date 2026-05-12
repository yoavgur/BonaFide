"""Shared assistant-prefill generation for faithfulness metrics.

Builds prompts with assistant-prefilled CoTs and generates answers using
either vLLM (batched) or HuggingFace (sequential). Used by SCM,
Simulatability, and Early Answering metrics.
"""

from __future__ import annotations

import logging
from typing import Any, Protocol, runtime_checkable

import torch

from generation.backends import build_prompt
from generation.model_registry import (
    ANSWER_ONLY_SYSTEM_PROMPT,
    COT_SYSTEM_PROMPT,
    ModelProfile,
    get_model_profile,
)
from generation.normalize import extract_answer_from_raw
from generation.thinking import split_thinking

logger = logging.getLogger(__name__)


@runtime_checkable
class PrefillGeneratorConfig(Protocol):
    """Minimal config interface for PrefillGenerator."""

    backend: str
    generation_temperature: float
    tensor_parallel_size: int


class PrefillGenerator:
    """Generates answers from prompts with injected (prefilled) CoTs.

    The assistant-prefill approach:
    1. Build a chat-template prompt for the question.
    2. Append the injected CoT as if the assistant already produced it:
       - Thinking models (e.g. Qwen3): <think>{cot}</think>
       - Non-thinking models: plain text CoT
    3. Let the model generate only the answer continuation.
    """

    def __init__(
        self,
        *,
        tokenizer: Any,
        model_name: str,
        config: PrefillGeneratorConfig,
        hf_model: Any | None = None,
    ) -> None:
        self.tokenizer = tokenizer
        self.model_name = model_name
        self.config = config
        self.profile: ModelProfile = get_model_profile(model_name)

        logger.info(
            "PrefillGenerator: using model profile max_tokens=%d for %s",
            self.profile.max_tokens, model_name,
        )

        # Ensure pad_token is set (needed for batched HF generation)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        if config.backend == "vllm":
            try:
                from vllm import LLM
            except ImportError:
                raise ImportError(
                    "vllm is required when backend='vllm'. "
                    "Install with: pip install vllm"
                )
            self._vllm_llm = LLM(
                model=model_name,
                tensor_parallel_size=config.tensor_parallel_size,
                trust_remote_code=self.profile.trust_remote_code,
                dtype=self.profile.dtype,
                max_model_len=self.profile.max_model_len,
            )
            self._backend = "vllm"
        elif config.backend == "hf":
            if hf_model is None:
                raise ValueError(
                    "HF backend requires a model. "
                    "Set hf_model to a loaded HuggingFace model, or use backend='vllm'."
                )
            self._hf_model = hf_model
            self._backend = "hf"
        else:
            raise ValueError(f"Unknown backend: {config.backend!r}")

    def _system_message(self) -> str | None:
        """Build the system message, matching the original generation pipeline."""
        if self.profile.inject_cot_prompt:
            if self.profile.thinking_tag:
                return ANSWER_ONLY_SYSTEM_PROMPT
            else:
                return COT_SYSTEM_PROMPT
        return None

    def _build_prefilled_prompt(
        self,
        question: str,
        cot: str,
        *,
        close_thinking: bool = True,
        answer_prefix: str = "",
    ) -> str:
        """Build a prompt with the CoT injected as assistant prefill.

        Args:
            question: The question text.
            cot: The CoT text to inject.
            close_thinking: If True (default), closes the thinking block
                so the model generates only the answer. If False, leaves
                the thinking block open so the model continues reasoning
                (used by Adding Mistakes to let the model continue CoT
                after an injected mistake).
            answer_prefix: Optional text to append after the thinking block
                closes (e.g. '{"final_answer":' to force the model to
                complete the JSON answer immediately without extra
                reasoning). Only used when close_thinking=True.

        For thinking models with close_thinking=True:
            appends <think>{cot}</think>{answer_prefix}
        For thinking models with close_thinking=False:
            appends <think>{cot} (no closing tag). answer_prefix is ignored.

        For non-thinking models: appends the CoT as plain text, followed
        by answer_prefix if provided.

        Raises:
            ValueError: If cot is empty. An empty prefill causes thinking
                models to restart reasoning from scratch, defeating the
                purpose of prefill-based evaluation.
        """
        if not cot or not cot.strip():
            raise ValueError(
                "PrefillGenerator requires non-empty CoT. An empty prefill "
                "causes thinking models to restart reasoning from scratch, "
                "producing meaningless results."
            )

        prompt = build_prompt(
            self.tokenizer,
            question,
            self._system_message(),
            self.profile.enable_thinking,
            self.profile.chat_template_kwargs,
        )

        tag = self.profile.thinking_tag
        if tag:
            open_tag = f"<{tag}>"
            close_tag = f"</{tag}>"

            # Handle double-tag issue: chat template may already add <think>
            # at the end of the prompt (as start of assistant's response).
            stripped = prompt.rstrip()
            if stripped.endswith(open_tag):
                # Template already opened the thinking block — just append
                # the CoT content.
                if close_thinking:
                    prefill = f"\n{cot}\n{close_tag}\n{answer_prefix}"
                else:
                    prefill = f"\n{cot}\n"
            else:
                # Template didn't open thinking block — add open tag.
                if close_thinking:
                    prefill = f"{open_tag}\n{cot}\n{close_tag}\n{answer_prefix}"
                else:
                    prefill = f"{open_tag}\n{cot}\n"
        else:
            # Non-thinking model: inject CoT as plain text continuation.
            prefill = f"{cot}\n{answer_prefix}"

        return prompt + prefill

    def _run_generation(
        self,
        questions: list[str],
        cots: list[str],
        *,
        close_thinking: bool = True,
        answer_prefix: str = "",
    ) -> tuple[list[str], list[str], list[str]]:
        """Core generation logic: build prompts, generate, parse.

        Args:
            questions: List of question texts.
            cots: List of CoT texts to inject.
            close_thinking: If True, close the thinking block (model
                generates only answer). If False, leave it open (model
                continues CoT reasoning).
            answer_prefix: Optional text to append after the thinking
                block closes, forcing the model to continue from there
                (e.g. '{"final_answer":').

        Returns:
            (prompts, raw_outputs, parsed_answers) — all parallel lists.
        """
        if len(questions) != len(cots):
            raise ValueError(
                f"questions and cots must have same length, "
                f"got {len(questions)} and {len(cots)}"
            )

        prompts = [
            self._build_prefilled_prompt(
                q, c,
                close_thinking=close_thinking,
                answer_prefix=answer_prefix,
            )
            for q, c in zip(questions, cots)
        ]

        if self._backend == "vllm":
            raw_outputs = self._generate_vllm(prompts)
        else:
            raw_outputs = self._generate_hf(prompts)

        # Parse answers from raw outputs.
        # When answer_prefix is set, the raw output is a continuation of that
        # prefix (e.g. prefix='{"final_answer":' and raw=' "B"}').  Prepend
        # the prefix so the parser sees the full JSON.
        answers = []
        for i, raw in enumerate(raw_outputs):
            text_for_parsing = answer_prefix + raw if answer_prefix else raw
            answer = self._parse_answer(text_for_parsing)
            if not answer:
                logger.warning(
                    "Could not parse answer from generation output for prompt %d. "
                    "Raw output: %r",
                    i, raw[:200],
                )
            answers.append(answer or "")
        return prompts, raw_outputs, answers

    def generate_answers(
        self,
        questions: list[str],
        cots: list[str],
        *,
        close_thinking: bool = True,
        answer_prefix: str = "",
    ) -> list[str]:
        """Generate answers for question+CoT pairs via assistant prefill.

        Args:
            questions: List of question texts.
            cots: List of CoT texts to inject (one per question).
            close_thinking: If True, close the thinking block (model
                generates only answer). If False, leave it open (model
                continues CoT reasoning then answers).
            answer_prefix: Optional text to append after the thinking
                block closes (e.g. '{"final_answer":').

        Returns:
            List of parsed answer strings.

        Raises:
            RuntimeError: If generation fails for any prompt.
        """
        _, _, answers = self._run_generation(
            questions, cots,
            close_thinking=close_thinking,
            answer_prefix=answer_prefix,
        )
        return answers

    def generate_answers_debug(
        self,
        questions: list[str],
        cots: list[str],
        *,
        close_thinking: bool = True,
        answer_prefix: str = "",
    ) -> tuple[list[str], list[str], list[str]]:
        """Like generate_answers, but also returns prompts and raw outputs.

        Returns:
            (prompts, raw_outputs, parsed_answers) — all parallel lists.
            - prompts: The full prefilled prompt strings sent to the model.
            - raw_outputs: The raw model output text (before parsing).
            - parsed_answers: The extracted answer strings.
        """
        return self._run_generation(
            questions, cots,
            close_thinking=close_thinking,
            answer_prefix=answer_prefix,
        )

    def _generate_vllm(self, prompts: list[str]) -> list[str]:
        """Batch generation via vLLM."""
        from vllm import SamplingParams

        params = SamplingParams(
            temperature=self.config.generation_temperature,
            max_tokens=self.profile.max_tokens,
        )

        outputs = self._vllm_llm.generate(prompts, sampling_params=params)

        raw_outputs = []
        for output in outputs:
            if not output.outputs:
                raise RuntimeError(
                    f"vLLM returned no outputs for prompt: {output.prompt[:100]}..."
                )
            raw_outputs.append(output.outputs[0].text)
        return raw_outputs

    def _generate_hf(self, prompts: list[str]) -> list[str]:
        """Sequential generation via HuggingFace model.generate()."""
        model = self._hf_model
        tokenizer = self.tokenizer
        device = next(model.parameters()).device

        raw_outputs = []
        for prompt in prompts:
            inputs = tokenizer(prompt, return_tensors="pt").to(device)
            input_len = inputs["input_ids"].shape[-1]

            with torch.no_grad():
                gen_kwargs: dict[str, Any] = dict(
                    max_new_tokens=self.profile.max_tokens,
                    do_sample=self.config.generation_temperature > 0,
                )
                if self.config.generation_temperature > 0:
                    gen_kwargs["temperature"] = self.config.generation_temperature

                output_ids = model.generate(**inputs, **gen_kwargs)

            new_ids = output_ids[0][input_len:]
            raw = tokenizer.decode(new_ids, skip_special_tokens=True)
            raw_outputs.append(raw)

        return raw_outputs

    def _parse_answer(self, raw_output: str) -> str | None:
        """Parse the answer from raw generation output.

        Tries in order:
        1. split_thinking (handles <think> tag models and JSON extraction)
        2. extract_answer_from_raw (brace-matched JSON + last-line fallback)
        """
        # For thinking models, there might be additional thinking output
        # before the answer. Use split_thinking to handle this.
        _, answer = split_thinking(
            raw_output,
            self.profile.thinking_tag,
            self.profile.cot_pattern,
        )
        if answer:
            return answer

        # Fallback: try raw extraction
        return extract_answer_from_raw(raw_output)
