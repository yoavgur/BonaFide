"""Generation module for the SCM metric.

Builds prompts with assistant-prefilled CoTs and generates answers using
either vLLM (default, batched) or HuggingFace (sequential).

Delegates to the shared PrefillGenerator for the actual generation logic.
"""

from __future__ import annotations

import logging
from typing import Any

from metrics.scm.config import SCMConfig
from metrics.shared.prefill_generator import PrefillGenerator

logger = logging.getLogger(__name__)


class SCMGenerator:
    """Generates answers from prompts with injected (prefilled) CoTs.

    Thin wrapper around PrefillGenerator, configured with SCM-specific
    settings (generation temperature, max tokens, etc.).
    """

    def __init__(
        self,
        *,
        tokenizer: Any,
        model_name: str,
        config: SCMConfig,
        hf_model: Any | None = None,
    ) -> None:
        self._generator = PrefillGenerator(
            tokenizer=tokenizer,
            model_name=model_name,
            config=config,
            hf_model=hf_model,
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
