"""Generation module for the Adding Mistakes metric.

Uses the same model that generated the CoT to continue reasoning after
a mistake is injected, via open-ended assistant-prefill (thinking block
left open).

Delegates to the shared PrefillGenerator with close_thinking=False.
"""

from __future__ import annotations

import logging
from typing import Any

from metrics.adding_mistakes.config import AddingMistakesConfig
from metrics.shared.prefill_generator import PrefillGenerator

logger = logging.getLogger(__name__)


class AddingMistakesGenerator:
    """Generates continuations after injecting a mistake into the CoT.

    Uses the *same* model that generated the original CoT. The thinking
    block is left OPEN so the model continues reasoning after the
    corrupted prefix, then produces an answer.
    """

    def __init__(
        self,
        *,
        tokenizer: Any,
        model_name: str,
        config: AddingMistakesConfig,
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
        """Generate answers for question+corrupted-CoT pairs.

        The thinking block is left open so the model continues reasoning
        after the corrupted prefix.

        Args:
            questions: List of question texts.
            cots: List of corrupted CoT prefixes (one per question).

        Returns:
            List of parsed answer strings.
        """
        return self._generator.generate_answers(
            questions, cots, close_thinking=False,
        )

    def generate_answers_debug(
        self, questions: list[str], cots: list[str]
    ) -> tuple[list[str], list[str], list[str]]:
        """Like generate_answers, but also returns prompts and raw outputs.

        Returns:
            (prompts, raw_outputs, parsed_answers) — all parallel lists.
        """
        return self._generator.generate_answers_debug(
            questions, cots, close_thinking=False,
        )
