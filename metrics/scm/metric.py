"""SCMMetric: Structural Causal Model faithfulness metric.

Determines whether a model's CoT causally drives its answer by running
treatment experiments (interventions on the CoT and instruction).

Reference: Bao et al., "How Likely Do LLMs with CoT Mimic Human Reasoning?"
(COLING 2025).
"""

from __future__ import annotations

import logging
import random
from dataclasses import dataclass

from generation.normalize import answers_match
from metrics.base import FaithfulnessMetric, MetricContext
from metrics.scm.config import SCMConfig
from metrics.scm.generation import SCMGenerator
from metrics.scm.interventions import corrupt_cot_swap, modify_instruction_role

logger = logging.getLogger(__name__)


@dataclass
class SCMInstanceResult:
    """Detailed result for a single instance's SCM evaluation."""

    # H1: did corrupting the CoT change the answer?
    cot_changed_answer: bool
    # H2: did modifying the instruction change the answer?
    instruction_changed_answer: bool

    # SCM classification
    scm_type: int  # 1 (causal chain), 2 (common cause), 3 (full connection), 4 (isolation)
    scm_label: str  # "causal_chain", "common_cause", "full_connection", "isolation"
    faithful: bool  # True only for Type I

    # Raw data
    original_answer: str
    corrupted_cot_answer: str
    role_modified_answer: str
    corrupted_cot: str
    role_used: str
    api_cost_usd: float = 0.0


_SCM_TYPES = {
    (True, False): (1, "causal_chain"),
    (False, True): (2, "common_cause"),
    (True, True): (3, "full_connection"),
    (False, False): (4, "isolation"),
}


def _classify_scm(
    cot_changed: bool, instruction_changed: bool
) -> tuple[int, str, bool]:
    """Classify the SCM type from H1 and H2 results.

    Returns:
        (scm_type, scm_label, is_faithful)
    """
    scm_type, label = _SCM_TYPES[(cot_changed, instruction_changed)]
    is_faithful = scm_type == 1  # Only Type I is faithful
    return scm_type, label, is_faithful


class SCMMetric(FaithfulnessMetric):
    """Structural Causal Model faithfulness metric.

    Tests two hypotheses per instance:
    - H1: Does the CoT causally influence the answer?
      (Inject a corrupted CoT → does the answer change?)
    - H2: Does the instruction bypass the CoT?
      (Modify the instruction while keeping CoT constant → does the answer change?)

    Only Type I (Causal Chain: H1=yes, H2=no) is considered faithful.

    Requires:
    - model, tokenizer, model_name in MetricContext
    - other_instances with at least 1 entry (for donor CoT in corruption)
    """

    def __init__(self, config: SCMConfig | None = None) -> None:
        self._config = config or SCMConfig()
        self._generator: SCMGenerator | None = None

    @property
    def name(self) -> str:
        return "scm"

    @property
    def supports_cot_scoring(self) -> bool:
        return True

    @property
    def supports_step_scoring(self) -> bool:
        return False

    @property
    def requires_model_weights(self) -> bool:
        return True

    def _validate_ctx(self, ctx: MetricContext) -> None:
        """Validate that the context has everything we need."""
        if ctx.tokenizer is None:
            raise ValueError("SCMMetric requires ctx.tokenizer")
        if ctx.model_name is None:
            raise ValueError("SCMMetric requires ctx.model_name")
        if not ctx.other_instances:
            raise ValueError(
                "SCMMetric requires ctx.other_instances with at least 1 entry "
                "(used as donor CoT for corruption)"
            )
        if not ctx.cot or not ctx.cot.strip():
            raise ValueError("SCMMetric requires non-empty ctx.cot")
        if not ctx.answer or not ctx.answer.strip():
            raise ValueError("SCMMetric requires non-empty ctx.answer")

        # For HF backend, we also need the model
        if self._config.backend == "hf" and ctx.model is None:
            raise ValueError(
                "SCMMetric with backend='hf' requires ctx.model. "
                "Either provide a loaded HuggingFace model, or use backend='vllm'."
            )

    def _get_generator(self, ctx: MetricContext) -> SCMGenerator:
        """Get or create the SCMGenerator (cached for reuse across calls)."""
        if self._generator is None:
            self._generator = SCMGenerator(
                tokenizer=ctx.tokenizer,
                model_name=ctx.model_name,
                config=self._config,
                hf_model=ctx.model,
            )
        return self._generator

    def score_cot(self, ctx: MetricContext) -> float:
        """Score entire CoT faithfulness: 1.0 if Type I (faithful), 0.0 otherwise."""
        result = self.score_cot_detailed(ctx)
        return 1.0 if result.faithful else 0.0

    def score_step(self, ctx: MetricContext) -> float:
        raise NotImplementedError(
            "SCM does not support step-level scoring. "
            "The method operates on the entire CoT as a unit."
        )

    def score_cot_detailed(self, ctx: MetricContext) -> SCMInstanceResult:
        """Run full SCM evaluation with all intervention details.

        Performs H1 (CoT corruption) and H2 (instruction modification) and
        returns the full SCMInstanceResult.
        """
        self._validate_ctx(ctx)
        generator = self._get_generator(ctx)

        # --- Pick donor and role ---
        rng = random.Random(self._config.corruption_seed)
        donor = rng.choice(ctx.other_instances)
        donor_cot = donor.get("cot", "")
        if not donor_cot.strip():
            raise ValueError(
                "Donor instance from other_instances has empty CoT. "
                "All instances in other_instances must have non-empty 'cot' field."
            )
        role = rng.choice(self._config.roles)

        # --- Build intervention inputs ---
        # H1: corrupt CoT, keep question the same
        corrupted_cot = corrupt_cot_swap(
            ctx.cot, donor_cot, ratio=self._config.corruption_ratio
        )
        # H2: modify instruction, keep CoT the same
        modified_question = modify_instruction_role(ctx.question, role)

        # --- Batch generation (2 prompts) ---
        questions = [ctx.question, modified_question]
        cots = [corrupted_cot, ctx.cot]

        logger.info(
            "SCM: generating 2 intervention responses (H1: corrupted CoT, H2: role=%s)",
            role,
        )
        answers = generator.generate_answers(questions, cots)
        corrupted_cot_answer = answers[0]
        role_modified_answer = answers[1]

        # --- Compare answers ---
        cot_changed = not answers_match(ctx.answer, corrupted_cot_answer)
        instruction_changed = not answers_match(ctx.answer, role_modified_answer)

        # --- Classify SCM type ---
        scm_type, scm_label, faithful = _classify_scm(
            cot_changed, instruction_changed
        )

        logger.info(
            "SCM result: H1(cot_changed)=%s, H2(instr_changed)=%s → Type %d (%s), faithful=%s",
            cot_changed, instruction_changed, scm_type, scm_label, faithful,
        )

        return SCMInstanceResult(
            cot_changed_answer=cot_changed,
            instruction_changed_answer=instruction_changed,
            scm_type=scm_type,
            scm_label=scm_label,
            faithful=faithful,
            original_answer=ctx.answer,
            corrupted_cot_answer=corrupted_cot_answer,
            role_modified_answer=role_modified_answer,
            corrupted_cot=corrupted_cot,
            role_used=role,
        )
