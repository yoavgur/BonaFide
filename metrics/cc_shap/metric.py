"""CCSHAPMetric: CC-SHAP self-consistency measure for CoT faithfulness.

Orchestrates the full CC-SHAP pipeline using the vendored SHAP fork's Partition
explainer: build prompts -> compute SHAP values for prediction and explanation
-> compare contribution profiles -> score.

Uses string-level masking (Text masker with "..." collapse) and teacher-forcing
(log-odds) rather than token-level masking with Kernel SHAP.

Reference:
    Parcalabescu & Frank (2024). "On Measuring Faithfulness or Self-consistency
    of Natural Language Explanations." ACL 2024.
    https://github.com/Heidelberg-NLP/CC-SHAP
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from metrics.base import FaithfulnessMetric, MetricContext
from metrics.cc_shap.config import CCSHAPConfig
from metrics.cc_shap.contributions import normalize_contributions
from metrics.cc_shap.divergence import compute_all_divergences, compute_divergence
from metrics.cc_shap.model_wrapper import QuestionTeacherForcing, FixedTargetGenerator

logger = logging.getLogger(__name__)


@dataclass
class CCSHAPDetailedResult:
    """Detailed results from a CC-SHAP evaluation."""

    cc_shap_score: float  # 1 - primary divergence, in [-1, 1]
    divergence_method: str
    divergence_value: float  # Raw divergence before inversion
    all_divergences: dict[str, float] = field(default_factory=dict)
    c_prediction: NDArray | None = None  # Contribution vector for prediction
    c_explanation: NDArray | None = None  # Contribution vector for explanation
    n_input_tokens: int = 0
    n_prediction_tokens: int = 0
    n_explanation_tokens: int = 0
    n_coalitions_used: int = 0
    api_cost_usd: float = 0.0


class CCSHAPMetric(FaithfulnessMetric):
    """CC-SHAP self-consistency measure.

    Compares how the model's input tokens contribute to answer prediction
    vs. explanation (CoT) generation using Shapley values. High agreement
    indicates the explanation is consistent with the prediction process.

    Supports both CoT-level and step-level scoring. Requires model weights
    for forward passes.
    """

    def __init__(self, config: CCSHAPConfig | None = None):
        self.config = config or CCSHAPConfig()
        # Cache prediction contributions to avoid recomputation across step-level
        # calls for the same question+answer. Key: (question, answer), Value: c_pred.
        self._pred_cache: dict[tuple[str, str], NDArray] = {}
        self._pred_cache_max_size: int = 32

    @property
    def name(self) -> str:
        return "cc_shap"

    @property
    def supports_cot_scoring(self) -> bool:
        return True

    @property
    def supports_step_scoring(self) -> bool:
        return True

    @property
    def requires_model_weights(self) -> bool:
        return True

    def _validate_ctx(self, ctx: MetricContext) -> None:
        if ctx.model is None:
            raise ValueError("CC-SHAP requires model weights (ctx.model)")
        if ctx.tokenizer is None:
            raise ValueError("CC-SHAP requires a tokenizer (ctx.tokenizer)")
        if not ctx.question:
            raise ValueError("CC-SHAP requires a non-empty question (ctx.question)")
        if not ctx.cot:
            raise ValueError("CC-SHAP requires a non-empty CoT (ctx.cot)")
        if not ctx.answer:
            raise ValueError("CC-SHAP requires a non-empty answer (ctx.answer)")

    def _ensure_pad_token(self, tokenizer) -> None:
        if tokenizer.pad_token is None:
            logger.info(
                f"Tokenizer has no pad_token. Setting pad_token = eos_token "
                f"('{tokenizer.eos_token}', id={tokenizer.eos_token_id})"
            )
            tokenizer.pad_token = tokenizer.eos_token

    def _make_prompt_builder(self, tokenizer, model_name):
        """Create a callable that builds full chat-template prompt from question text."""
        from generation.backends import build_prompt
        from generation.model_registry import (
            get_model_profile,
            ANSWER_ONLY_SYSTEM_PROMPT,
            COT_SYSTEM_PROMPT,
        )

        profile = None
        if model_name:
            try:
                profile = get_model_profile(model_name)
            except ValueError:
                raise ValueError(f"No model profile for '{model_name}'")

        # Determine system message and thinking settings
        if profile is not None:
            if profile.thinking_tag is not None:
                system_message = ANSWER_ONLY_SYSTEM_PROMPT
            elif profile.inject_cot_prompt:
                system_message = COT_SYSTEM_PROMPT
            else:
                system_message = ANSWER_ONLY_SYSTEM_PROMPT
            enable_thinking = profile.enable_thinking
            chat_template_kwargs = profile.chat_template_kwargs
        else:
            system_message = ANSWER_ONLY_SYSTEM_PROMPT
            enable_thinking = False
            chat_template_kwargs = {}

        def builder(question_text: str) -> str:
            return build_prompt(
                tokenizer=tokenizer,
                question=question_text,
                system_message=system_message,
                enable_thinking=enable_thinking,
                chat_template_kwargs=chat_template_kwargs,
            )

        return builder

    def _compute_shap_contributions(
        self,
        model,
        tokenizer,
        question: str,
        model_name: str | None,
        target_text: str,
        branch_name: str = "unknown",
    ) -> NDArray[np.float64]:
        """Compute SHAP contribution vector using the paper's Partition explainer.

        Uses the vendored CC-SHAP SHAP library fork with:
        - Text masker (string-level "..." collapse)
        - TeacherForcing model (log-odds, not log-probs)
        - Partition explainer (hierarchical Owen values)
        """
        from metrics.cc_shap.shap_fork.maskers import Text as TextMasker, OutputComposite
        from metrics.cc_shap.shap_fork.explainers import Partition

        prompt_builder = self._make_prompt_builder(tokenizer, model_name)

        # Create our custom model wrapper
        model_wrapper = QuestionTeacherForcing(
            model=model,
            tokenizer=tokenizer,
            prompt_builder=prompt_builder,
            batch_size=self.config.batch_size,
            auto_batch=self.config.auto_batch,
        )

        # Create Text masker operating on question text
        masker = TextMasker(tokenizer, mask_token="...", collapse_mask_token=True)

        # Fixed target generator (skip generation, we already have the target)
        target_gen = FixedTargetGenerator(target_text)

        # OutputComposite wraps masker + target generator
        composite_masker = OutputComposite(masker, target_gen)

        logger.info(
            f"  [{branch_name}] Running Partition explainer: "
            f"max_evals={self.config.max_evals}, "
            f"target_len={len(tokenizer(target_text, add_special_tokens=False).input_ids)} tokens"
        )

        # Create and run Partition explainer
        explainer = Partition(model_wrapper, composite_masker)
        shap_values = explainer([question], max_evals=self.config.max_evals, silent=True)

        # shap_values.values shape: (1, n_input_tokens, n_output_tokens)
        phi = shap_values.values[0]  # (n_input_tokens, n_output_tokens)

        logger.info(
            f"  [{branch_name}] SHAP values computed: "
            f"phi shape={phi.shape}, range=[{phi.min():.4f}, {phi.max():.4f}]"
        )

        # Normalize using Eq 3-4 (contribution ratios + averaging)
        contributions = normalize_contributions(phi)
        logger.info(
            f"  [{branch_name}] Contributions normalized: "
            f"range=[{contributions.min():.4f}, {contributions.max():.4f}]"
        )

        return contributions

    def _get_prediction_contributions(
        self, model, tokenizer, question: str, model_name: str | None, answer: str,
    ) -> NDArray[np.float64]:
        """Get prediction contributions, using cache if available.

        For step-level scoring, the prediction branch (question → answer) is
        identical across all steps of the same question. This cache avoids
        recomputing it for every step.
        """
        cache_key = (question, answer)
        if cache_key in self._pred_cache:
            logger.info("  [prediction] cache HIT (skipping %d evals)", self.config.max_evals)
            return self._pred_cache[cache_key]

        logger.info("  [prediction] cache MISS — computing (%d evals, batch_size=%d)",
                     self.config.max_evals, self.config.batch_size)
        c_pred = self._compute_shap_contributions(
            model, tokenizer, question, model_name,
            target_text=answer,
            branch_name="prediction",
        )

        # Evict oldest if cache is full
        if len(self._pred_cache) >= self._pred_cache_max_size:
            oldest_key = next(iter(self._pred_cache))
            del self._pred_cache[oldest_key]

        self._pred_cache[cache_key] = c_pred
        return c_pred

    def score_cot(self, ctx: MetricContext) -> float:
        """CC-SHAP score for full CoT faithfulness.

        Returns a value in [0, 1] where higher = more faithful (self-consistent).
        """
        result = self.score_cot_detailed(ctx)
        return max(0.0, result.cc_shap_score)

    def score_step(self, ctx: MetricContext) -> float:
        """CC-SHAP score for a single CoT step.

        Restricts explanation SHAP to output tokens within ctx.step_span.
        Returns the raw score in [-1, 1] (higher = more faithful); negatives
        indicate divergence > 1 between prediction and explanation contributions.
        """
        self._validate_ctx(ctx)
        if ctx.step_span is None:
            raise ValueError("score_step requires ctx.step_span to be set")

        char_start, char_end = ctx.step_span
        if char_start < 0 or char_end > len(ctx.cot) or char_start >= char_end:
            raise ValueError(
                f"Invalid step_span ({char_start}, {char_end}) for CoT of length {len(ctx.cot)}"
            )

        self._ensure_pad_token(ctx.tokenizer)

        step_text = ctx.cot[char_start:char_end]
        step_tokens = ctx.tokenizer(step_text, add_special_tokens=False).input_ids
        if len(step_tokens) == 0:
            raise ValueError(f"Step text tokenizes to empty: {step_text!r}")

        logger.info(
            f"Step span ({char_start}, {char_end}) → {len(step_tokens)} tokens: "
            f"'{step_text[:80]}...'"
        )

        # Prediction branch: target = answer (cached across steps of same question)
        c_pred = self._get_prediction_contributions(
            ctx.model, ctx.tokenizer, ctx.question, ctx.model_name, ctx.answer,
        )

        # Explanation branch: target = step text only
        c_expl = self._compute_shap_contributions(
            ctx.model, ctx.tokenizer, ctx.question, ctx.model_name,
            target_text=step_text,
            branch_name="step_explanation",
        )

        div = compute_divergence(c_pred, c_expl, method=self.config.divergence)
        score = 1.0 - div
        logger.info(f"Step CC-SHAP: divergence({self.config.divergence})={div:.4f}, score={score:.4f}")
        return score



    def score_cot_detailed(self, ctx: MetricContext) -> CCSHAPDetailedResult:
        """Full CC-SHAP analysis with all intermediate results."""
        self._validate_ctx(ctx)
        self._ensure_pad_token(ctx.tokenizer)

        # Tokenize for logging
        answer_tokens = ctx.tokenizer(ctx.answer, add_special_tokens=False).input_ids
        cot_tokens = ctx.tokenizer(ctx.cot, add_special_tokens=False).input_ids
        question_tokens = ctx.tokenizer(ctx.question, add_special_tokens=False).input_ids

        if len(answer_tokens) == 0:
            raise ValueError(f"Answer tokenizes to empty: {ctx.answer!r}")
        if len(cot_tokens) == 0:
            raise ValueError(f"CoT tokenizes to empty: {ctx.cot!r}")

        logger.info(
            f"CC-SHAP: question={len(question_tokens)} tokens, "
            f"prediction={len(answer_tokens)} tokens, "
            f"explanation={len(cot_tokens)} tokens"
        )

        # Compute contribution profiles
        c_pred = self._get_prediction_contributions(
            ctx.model, ctx.tokenizer, ctx.question, ctx.model_name, ctx.answer,
        )
        c_expl = self._compute_shap_contributions(
            ctx.model, ctx.tokenizer, ctx.question, ctx.model_name,
            target_text=ctx.cot,
            branch_name="explanation",
        )

        # Compute all divergences
        all_divs = compute_all_divergences(c_pred, c_expl)
        primary_div = all_divs[self.config.divergence]
        cc_shap_score = 1.0 - primary_div

        logger.info(
            f"CC-SHAP results: score={cc_shap_score:.4f} "
            f"(1 - {self.config.divergence}={primary_div:.4f}). "
            f"All divergences: { {k: round(v, 4) for k, v in all_divs.items()} }"
        )

        return CCSHAPDetailedResult(
            cc_shap_score=cc_shap_score,
            divergence_method=self.config.divergence,
            divergence_value=primary_div,
            all_divergences=all_divs,
            c_prediction=c_pred,
            c_explanation=c_expl,
            n_input_tokens=len(question_tokens),
            n_prediction_tokens=len(answer_tokens),
            n_explanation_tokens=len(cot_tokens),
            n_coalitions_used=self.config.max_evals,
        )
