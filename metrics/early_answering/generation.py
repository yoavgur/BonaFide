"""Generation module for the Early Answering metric.

Uses the same model that generated the CoT to answer questions with
truncated CoTs via assistant-prefill.

Delegates to the shared PrefillGenerator for the actual generation logic.
"""

from __future__ import annotations

import logging
from typing import Any

from metrics.early_answering.config import EarlyAnsweringConfig
from metrics.shared.prefill_generator import PrefillGenerator

logger = logging.getLogger(__name__)


class EarlyAnsweringGenerator:
    """Generates answers with truncated CoTs via assistant prefill.

    Uses the *same* model that generated the original CoT — this is NOT a
    simulator approach. The model is provided via the MetricContext (tokenizer,
    model_name, and optionally model for HF backend).
    """

    def __init__(
        self,
        *,
        tokenizer: Any,
        model_name: str,
        config: EarlyAnsweringConfig,
        hf_model: Any | None = None,
    ) -> None:
        self._generator = PrefillGenerator(
            tokenizer=tokenizer,
            model_name=model_name,
            config=config,
            hf_model=hf_model,
        )

    def generate_answers(
        self, questions: list[str], cots: list[str], **kwargs: Any
    ) -> list[str]:
        """Generate answers for question+CoT pairs via assistant prefill.

        Args:
            questions: List of question texts.
            cots: List of CoT texts to inject (one per question).
            **kwargs: Forwarded to PrefillGenerator (e.g. close_thinking, answer_prefix).

        Returns:
            List of parsed answer strings.
        """
        return self._generator.generate_answers(questions, cots, **kwargs)

    def generate_answers_debug(
        self, questions: list[str], cots: list[str], **kwargs: Any
    ) -> tuple[list[str], list[str], list[str]]:
        """Like generate_answers, but also returns prompts and raw outputs.

        Returns:
            (prompts, raw_outputs, parsed_answers) — all parallel lists.
        """
        return self._generator.generate_answers_debug(questions, cots, **kwargs)
