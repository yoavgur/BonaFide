"""SimulatabilityMetric: measures CoT faithfulness via simulator reproduction.

Tests whether a small simulator model, given the original model's question and
CoT via assistant-prefill, reproduces the same answer. Based on the RLC
(rationale-label consistency) framework from the FRAME paper.

Reference: Chan et al., "FRAME: Evaluating Rationale-Label Consistency
Metrics for Free-Text Rationales" (arXiv 2207.00779).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from generation.normalize import answers_match
from metrics.base import FaithfulnessMetric, MetricContext
from metrics.simulatability.config import SimulatabilityConfig
from metrics.simulatability.generation import (
    SimulatabilityGenerator,
    resolve_simulator_model,
)

logger = logging.getLogger(__name__)


@dataclass
class SimulatabilityResult:
    """Detailed result for a single instance's simulatability evaluation."""

    matches: bool  # Whether simulator's answer matches the original
    score: float  # 1.0 if matches, 0.0 otherwise
    simulator_answer: str  # What the simulator predicted
    target_answer: str  # The original model's answer (ctx.answer)
    simulator_model: str  # Which simulator model was used
    prompt: str = ""  # Full prefilled prompt sent to simulator
    raw_output: str = ""  # Raw simulator output before answer parsing
    api_cost_usd: float = 0.0


class SimulatabilityMetric(FaithfulnessMetric):
    """Simulatability faithfulness metric.

    Measures whether a small simulator model can reproduce the original model's
    answer when given the same question and CoT via assistant-prefill.

    The simulator model is chosen automatically:
    - Default: Qwen3 4B (thinking or instruct, matching the evaluated model)
    - Fallback: OLMo 7B when the evaluated model IS Qwen3 4B

    Does NOT require the original model's weights — loads its own simulator
    model internally via vLLM or HuggingFace.

    Requires:
    - model_name in MetricContext (to determine simulator model)
    - Non-empty question, cot, answer
    """

    def __init__(self, config: SimulatabilityConfig | None = None) -> None:
        self._config = config or SimulatabilityConfig()
        self._generator: SimulatabilityGenerator | None = None
        self._generator_model: str | None = None

    @property
    def name(self) -> str:
        return "simulatability"

    @property
    def supports_cot_scoring(self) -> bool:
        return True

    @property
    def supports_step_scoring(self) -> bool:
        return False

    @property
    def requires_model_weights(self) -> bool:
        return False

    def _validate_ctx(self, ctx: MetricContext) -> None:
        """Validate that the context has everything we need."""
        if ctx.model_name is None:
            raise ValueError(
                "SimulatabilityMetric requires ctx.model_name "
                "(used to determine the simulator model)"
            )
        if not ctx.question or not ctx.question.strip():
            raise ValueError("SimulatabilityMetric requires non-empty ctx.question")
        if not ctx.cot or not ctx.cot.strip():
            raise ValueError("SimulatabilityMetric requires non-empty ctx.cot")
        if not ctx.answer or not ctx.answer.strip():
            raise ValueError("SimulatabilityMetric requires non-empty ctx.answer")

    def _get_generator(self, ctx: MetricContext) -> SimulatabilityGenerator:
        """Get or create the SimulatabilityGenerator.

        Cached for reuse across calls. If the evaluated model changes
        (different ctx.model_name), a new generator is created.
        """
        simulator_model = resolve_simulator_model(ctx.model_name)

        if self._generator is not None and self._generator_model == simulator_model:
            return self._generator

        logger.info("Initializing simulator generator with model: %s", simulator_model)
        self._generator = SimulatabilityGenerator(
            simulator_model_name=simulator_model,
            config=self._config,
        )
        self._generator_model = simulator_model
        return self._generator

    def score_cot(self, ctx: MetricContext) -> float:
        """Score entire CoT faithfulness: 1.0 if simulator reproduces the answer, 0.0 otherwise."""
        result = self.score_cot_detailed(ctx)
        return result.score

    def score_step(self, ctx: MetricContext) -> float:
        raise NotImplementedError(
            "Simulatability does not support step-level scoring. "
            "The method operates on the entire CoT as a unit."
        )

    def score_cot_detailed(self, ctx: MetricContext) -> SimulatabilityResult:
        """Run full simulatability evaluation with detailed results.

        Prefills the simulator with the question + CoT and checks whether
        the simulator's generated answer matches the original model's answer.
        """
        self._validate_ctx(ctx)
        generator = self._get_generator(ctx)

        simulator_model = self._generator_model

        logger.info(
            "Simulatability: generating answer via %s with prefilled CoT",
            simulator_model,
        )
        prompts, raw_outputs, answers = generator.generate_answers_debug(
            [ctx.question], [ctx.cot]
        )
        simulator_answer = answers[0]

        matches = answers_match(simulator_answer, ctx.answer)

        logger.info(
            "Simulatability result: simulator=%r, target=%r, matches=%s",
            simulator_answer[:100], ctx.answer[:100], matches,
        )

        return SimulatabilityResult(
            matches=matches,
            score=1.0 if matches else 0.0,
            simulator_answer=simulator_answer,
            target_answer=ctx.answer,
            simulator_model=simulator_model,
            prompt=prompts[0],
            raw_output=raw_outputs[0],
        )
