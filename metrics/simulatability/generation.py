"""Generation module for the simulatability metric.

Loads a small simulator model (separate from the model being evaluated) and
generates answers via assistant-prefill — injecting the original model's CoT
into the simulator's prompt and checking if it reproduces the same answer.

Delegates to the shared PrefillGenerator for the actual generation logic.
"""

from __future__ import annotations

import logging
from typing import Any

from generation.model_registry import get_model_profile

from metrics.shared.prefill_generator import PrefillGenerator
from metrics.simulatability.config import SimulatabilityConfig

logger = logging.getLogger(__name__)


# Simulator model selection: maps evaluated model patterns to simulator models.
# When the evaluated model IS the default simulator, we use a fallback to
# avoid self-evaluation.
_SIMULATOR_DEFAULTS = {
    "thinking": "Qwen/Qwen3-4B-Thinking-2507",
    "instruct": "Qwen/Qwen3-4B-Instruct-2507",
}
_SIMULATOR_FALLBACKS = {
    "thinking": "allenai/Olmo-3-7B-Think",
    "instruct": "allenai/Olmo-3-7B-Instruct",
}


def resolve_simulator_model(evaluated_model_name: str) -> str:
    """Choose the simulator model based on the model being evaluated.

    Rules:
    - Default simulator: Qwen3 4B (thinking or instruct variant, matching
      the evaluated model's type).
    - When the evaluated model IS Qwen3 4B, use OLMo 7B instead to avoid
      self-evaluation bias.

    Thinking vs instruct is determined by the evaluated model's ModelProfile.
    """
    profile = get_model_profile(evaluated_model_name)
    variant = "thinking" if profile.thinking_tag else "instruct"

    # Check if the evaluated model is a Qwen3 4B variant
    if "Qwen3-4B" in evaluated_model_name:
        simulator = _SIMULATOR_FALLBACKS[variant]
        logger.info(
            "Evaluated model is Qwen3-4B (%s); using fallback simulator: %s",
            evaluated_model_name, simulator,
        )
    else:
        simulator = _SIMULATOR_DEFAULTS[variant]
        logger.info(
            "Using default simulator %s for evaluated model %s",
            simulator, evaluated_model_name,
        )

    return simulator


class SimulatabilityGenerator:
    """Generates answers by prefilling a simulator model with CoTs.

    Unlike SCMGenerator, this loads its OWN model (the simulator), which is
    a different model from the one being evaluated. The simulator model is
    determined by ``resolve_simulator_model()``.

    Delegates to PrefillGenerator for the actual prefill + generation logic.
    The key difference is that this class manages loading the simulator model
    (via vLLM) or accepting a pre-loaded HF model, and provides the simulator's
    tokenizer to PrefillGenerator.
    """

    def __init__(
        self,
        *,
        simulator_model_name: str,
        config: SimulatabilityConfig,
        hf_model: Any | None = None,
    ) -> None:
        self.simulator_model_name = simulator_model_name
        self.config = config

        if config.backend == "vllm":
            # PrefillGenerator will handle vLLM initialization internally.
            # But we need the tokenizer for external use (e.g., by the metric).
            # PrefillGenerator loads vLLM with the simulator model name.
            self._generator = PrefillGenerator(
                tokenizer=self._load_vllm_tokenizer(simulator_model_name, config),
                model_name=simulator_model_name,
                config=config,
                hf_model=None,
            )
            self.tokenizer = self._generator.tokenizer
        elif config.backend == "hf":
            if hf_model is None:
                raise ValueError(
                    "HF backend requires a pre-loaded simulator model. "
                    "Pass hf_model to SimulatabilityGenerator, or use backend='vllm'."
                )
            from transformers import AutoTokenizer

            profile = get_model_profile(simulator_model_name)
            tokenizer = AutoTokenizer.from_pretrained(
                simulator_model_name,
                trust_remote_code=profile.trust_remote_code,
            )
            self._generator = PrefillGenerator(
                tokenizer=tokenizer,
                model_name=simulator_model_name,
                config=config,
                hf_model=hf_model,
            )
            self.tokenizer = tokenizer
        else:
            raise ValueError(f"Unknown backend: {config.backend!r}")

    @staticmethod
    def _load_vllm_tokenizer(
        model_name: str, config: SimulatabilityConfig
    ) -> Any:
        """Load tokenizer from vLLM for the simulator model.

        We need the tokenizer before PrefillGenerator creates the LLM instance,
        so we load it separately via transformers.
        """
        from transformers import AutoTokenizer

        profile = get_model_profile(model_name)
        return AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=profile.trust_remote_code,
        )

    def generate_answers(
        self, questions: list[str], cots: list[str]
    ) -> list[str]:
        """Generate answers for question+CoT pairs via assistant prefill.

        Args:
            questions: List of question texts.
            cots: List of CoT texts to inject (one per question).

        Returns:
            List of parsed answer strings.

        Raises:
            RuntimeError: If generation fails for any prompt.
        """
        return self._generator.generate_answers(questions, cots)

    def generate_answers_debug(
        self, questions: list[str], cots: list[str]
    ) -> tuple[list[str], list[str], list[str]]:
        """Like generate_answers, but also returns prompts and raw outputs.

        Returns:
            (prompts, raw_outputs, parsed_answers) — all parallel lists.
        """
        return self._generator.generate_answers_debug(questions, cots)
