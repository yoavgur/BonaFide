"""FURMetric: Faithfulness by Unlearning Reasoning steps.

Orchestrates the full FUR pipeline: segment CoT → build forget/retain sets →
unlearn each step → generate fresh response → compare answers.
"""

from __future__ import annotations

import gc
import logging
import os
from dataclasses import dataclass, field

import torch

from generation.backends import build_prompt
from generation.normalize import answers_match
from generation.model_registry import (
    ANSWER_ONLY_SYSTEM_PROMPT,
    COT_SYSTEM_PROMPT,
    get_model_profile,
)
from generation.thinking import split_thinking
from metrics.base import FaithfulnessMetric, MetricContext
from metrics.fur.config import FURConfig
from metrics.fur.dataset import (
    build_forget_pairs,
    build_paraphrase_forget_pairs,
    build_retain_pairs,
    encode_step_for_unlearning,
    left_pad_sequence,
    IGNORE_IDX,
)
from metrics.fur.segmentation import StepInfo, segment_and_filter
from metrics.fur.unlearn import (
    UnlearnValidation,
    freeze_non_ff2,
    get_ff2_state_dict,
    restore_ff2_state_dict,
    unlearn_step,
)

logger = logging.getLogger(__name__)


@dataclass
class FURStepResult:
    """Result for a single step's unlearning."""

    step_text: str
    char_start: int
    char_end: int
    original_answer: str
    new_answer: str
    answer_changed: bool
    new_cot: str = ""
    # Paper Eq. 2: probability reduction of forget step. ~1.0 = fully unlearned, ~0 = no effect.
    efficacy: float | None = None
    # Paper Eq. 3 proxy: probability ratio on retain set. ~1.0 = retain preserved, smaller = damaged.
    specificity: float | None = None
    # Optional MMLU sanity check (paper §6.1 "Gen"). ~1.0 = general capabilities preserved.
    mmlu_agreement: float | None = None
    # Words being targeted by the unlearning (the step's content words, by POS).
    content_words: list[str] | None = None
    # Per-target-token oracle/post probabilities and top-K predictions, for debugging.
    step_debug: dict | None = None


@dataclass
class FURDetailedResult:
    """Detailed results from score_cot_detailed()."""

    ff_hard: float  # 1.0 if any step flipped the answer
    step_results: list[FURStepResult] = field(default_factory=list)
    skipped_steps: list[str] = field(default_factory=list)  # Steps with < min_content_tokens
    api_cost_usd: float = 0.0
    mean_efficacy: float | None = None
    mean_specificity: float | None = None


@dataclass
class FURUnlearnResult:
    """Result from unlearn_step_only() — unlearning done, generation deferred.

    Used in the two-stage vLLM pipeline: stage 1 produces these, stage 2
    loads the state dict and generates with vLLM.
    """

    step_text: str
    char_start: int
    char_end: int
    original_answer: str
    state_dict_path: str  # Path to saved modified FF2 state dict
    question: str  # For prompt construction in stage 2
    model_name: str
    skipped: bool = False  # True if step had too few content tokens
    wall_time_s: float = 0.0  # Total unlearning time (data prep + unlearn + save)
    efficacy: float | None = None
    specificity: float | None = None


def _generate_response(
    model,
    tokenizer,
    question: str,
    model_name: str | None = None,
    greedy: bool = False,
) -> tuple[str, str]:
    """Generate a fresh response (CoT + answer) from the model.

    Uses the existing generation/ pipeline for prompt building and answer parsing.

    Args:
        model: HF model (potentially with modified weights).
        tokenizer: HF tokenizer.
        question: The question text (user message content).
        model_name: HF model identifier for registry lookup.
        greedy: Force greedy decoding. When False, uses the model profile's
            recommended sampling parameters (safer for thinking models).

    Returns:
        (cot, answer): Parsed CoT and final answer strings.
    """
    # Get model profile for prompt construction
    if model_name:
        try:
            profile = get_model_profile(model_name)
        except ValueError:
            profile = None
    else:
        profile = None

    # Determine system message and thinking settings
    if profile is not None:
        if profile.thinking_tag is not None:
            system_message = ANSWER_ONLY_SYSTEM_PROMPT
        elif profile.inject_cot_prompt:
            system_message = COT_SYSTEM_PROMPT
        else:
            system_message = ANSWER_ONLY_SYSTEM_PROMPT
        enable_thinking = profile.enable_thinking
        thinking_tag = profile.thinking_tag
        chat_template_kwargs = profile.chat_template_kwargs
    else:
        # Fallback defaults
        system_message = ANSWER_ONLY_SYSTEM_PROMPT
        enable_thinking = False
        thinking_tag = None
        chat_template_kwargs = {}

    # Build prompt
    prompt_str = build_prompt(
        tokenizer=tokenizer,
        question=question,
        system_message=system_message,
        enable_thinking=enable_thinking,
        chat_template_kwargs=chat_template_kwargs,
    )

    # Use the model profile's max_tokens if available
    if profile is not None:
        max_new_tokens = profile.max_tokens
    else:
        raise ValueError(
            f"No model profile found for '{model_name}'. Cannot determine max_new_tokens. "
            f"Register the model in model_registry.py or pass a known model_name."
        )

    # Tokenize
    inputs = tokenizer(prompt_str, return_tensors="pt").to(model.device)
    input_len = inputs["input_ids"].shape[-1]

    # Build generation kwargs: greedy or profile-recommended sampling
    gen_kwargs = dict(max_new_tokens=max_new_tokens)
    if greedy or profile is None or profile.temperature <= 0:
        gen_kwargs["temperature"] = 0.0
        gen_kwargs["do_sample"] = False
        logger.debug(f"  Generating (greedy): input_len={input_len}, max_new_tokens={max_new_tokens}")
    else:
        gen_kwargs["temperature"] = profile.temperature
        gen_kwargs["top_p"] = profile.top_p
        if profile.top_k > 0:
            gen_kwargs["top_k"] = profile.top_k
        gen_kwargs["do_sample"] = True
        logger.debug(
            f"  Generating (sampled, temp={profile.temperature}): "
            f"input_len={input_len}, max_new_tokens={max_new_tokens}"
        )

    with torch.no_grad():
        output_ids = model.generate(**inputs, **gen_kwargs)

    new_ids = output_ids[0][input_len:]
    num_generated = len(new_ids)
    hit_limit = num_generated >= max_new_tokens
    raw_output = tokenizer.decode(new_ids, skip_special_tokens=True)

    logger.debug(f"  Generated {num_generated} tokens, hit_limit={hit_limit}")

    # Parse CoT and answer
    cot, answer = split_thinking(raw_output, thinking_tag)

    # If generation hits the token limit without producing an answer, treat it as
    # an answer change: unlearning disrupted the model enough that it can no longer
    # terminate its reasoning within the budget. Use a sentinel so answers_match
    # returns False against any real original answer.
    if hit_limit and not answer.strip():
        logger.warning(
            "Generation hit max_new_tokens=%d with no answer (generated %d tokens, "
            "%d chars). Treating as answer_changed=True.",
            max_new_tokens, num_generated, len(raw_output),
        )
        answer = "<TRUNCATED_NO_ANSWER>"

    return cot, answer


def _prepare_device(model) -> torch.device:
    """Get the device of the model's first parameter."""
    return next(model.parameters()).device


class FURMetric(FaithfulnessMetric):
    """Faithfulness by Unlearning Reasoning steps.

    Measures parametric faithfulness by unlearning individual CoT steps and
    checking if the model's answer changes when generating a fresh response.

    Both CoT-level and step-level scores are binary (0.0 or 1.0).
    """

    def __init__(self, config: FURConfig | None = None):
        self.config = config or FURConfig()
        self._cached_retain_tensors = None  # Cached across steps (same for all steps in a run)
        self._cached_spec_tensors = None    # Held-out specificity set; disjoint from retain
        self._cached_mmlu_data = None       # MMLU sanity-check tensors (prompts + choice_token_ids)
        self._spec_used_idxs: set[int] = set()  # Instance indices reserved for retain (excluded from spec)

    @property
    def name(self) -> str:
        return "fur"

    @property
    def supports_cot_scoring(self) -> bool:
        return True

    @property
    def supports_step_scoring(self) -> bool:
        return True

    @property
    def requires_model_weights(self) -> bool:
        return True

    def _resolve_learning_rate(self, model_name: str | None) -> None:
        """Set config.learning_rate from model_lr_overrides if a match is found.

        Uses longest-key-match to avoid e.g. "Qwen3-4B" shadowing "Qwen3-4B-Instruct".
        """
        if model_name is None:
            return
        name_lower = model_name.lower()
        best_key = None
        best_lr = None
        for key, lr in self.config.model_lr_overrides.items():
            if key.lower() in name_lower:
                if best_key is None or len(key) > len(best_key):
                    best_key = key
                    best_lr = lr
        if best_key is not None:
            logger.info(f"Using model-specific LR: {best_lr} (matched '{best_key}' in '{model_name}')")
            self.config.learning_rate = best_lr

    def _validate_ctx(self, ctx: MetricContext) -> None:
        """Check that required fields are present."""
        if ctx.model is None:
            raise ValueError("FUR requires model weights (ctx.model)")
        if ctx.tokenizer is None:
            raise ValueError("FUR requires a tokenizer (ctx.tokenizer)")
        if ctx.other_instances is None or len(ctx.other_instances) == 0:
            raise ValueError("FUR requires other_instances for the retain set")

    def _ensure_pad_token(self, tokenizer) -> None:
        """Ensure tokenizer has a pad token set."""
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

    @staticmethod
    def _reorder_steps_by_priority(
        steps: list[StepInfo],
        priority_spans: set[tuple[int, int]] | None,
    ) -> list[StepInfo]:
        """Reorder steps so priority spans come first, rest in original order.

        Used for early stopping: try steps most likely to be faithful first.
        """
        if not priority_spans:
            return steps
        priority = []
        rest = []
        for s in steps:
            if (s.char_start, s.char_end) in priority_spans:
                priority.append(s)
            else:
                rest.append(s)
        if priority:
            logger.info(
                "Step ordering: %d priority (FAITHFUL_STEP) + %d remaining",
                len(priority), len(rest),
            )
        return priority + rest

    def _get_paraphrases(self, step_texts: list[str]) -> tuple[dict[str, list[str]], float]:
        """Generate paraphrases for step texts using a Judge, if configured.

        Returns (mapping, api_cost_usd). Empty lists and 0.0 when num_paraphrases == 0.
        """
        if self.config.num_paraphrases <= 0:
            return {s: [] for s in step_texts}, 0.0

        from isolate_steps import Judge
        from metrics.fur.paraphrase import generate_paraphrases

        judge = Judge(model_name=self.config.paraphrase_model)
        result = generate_paraphrases(
            step_texts=step_texts,
            judge=judge,
            num_paraphrases=self.config.num_paraphrases,
        )
        return result, judge.total_cost

    def _split_retain_spec_indices(self, ctx: MetricContext) -> tuple[list[int], list[int]]:
        """Split other_instances indices into disjoint (retain, specificity) subsets.

        Deterministic via fixed seed so retain and spec are stable across steps.
        """
        import random as _r
        n = len(ctx.other_instances)
        rng = _r.Random(0xFE2E0E)
        idxs = list(range(n))
        rng.shuffle(idxs)
        n_retain = min(self.config.retain_sample_count, n)
        retain_idxs = idxs[:n_retain]
        n_spec = min(self.config.specificity_set_size, n - n_retain)
        spec_idxs = idxs[n_retain:n_retain + n_spec]
        return retain_idxs, spec_idxs

    def _build_retain_tensors(
        self,
        ctx: MetricContext,
        device: torch.device,
    ) -> list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        """Build retain tensors, caching across steps within the same run.

        The retain set depends only on other_instances + tokenizer + config,
        which are identical for all steps. Built once, reused for every step.

        Returns list of (input_ids, labels, attention_mask) tuples on device.
        """
        if self._cached_retain_tensors is not None:
            return self._cached_retain_tensors
        import time as _time
        _t0 = _time.time()

        retain_idxs, _ = self._split_retain_spec_indices(ctx)
        self._spec_used_idxs = set(retain_idxs)
        retain_only_instances = [ctx.other_instances[i] for i in retain_idxs]

        retain_pairs = build_retain_pairs(
            tokenizer=ctx.tokenizer,
            other_instances=retain_only_instances,
            min_content_tokens=self.config.min_content_tokens,
            n=self.config.retain_sample_count,
            max_seq_len=self.config.max_seq_len,
        )

        def _to_device(ids, labels, mask):
            return (
                ids.unsqueeze(0).to(device),
                labels.unsqueeze(0).to(device),
                mask.unsqueeze(0).to(device),
            )

        retain_tensor_list = []
        for r_ids, r_labels, r_mask, _ in retain_pairs:
            retain_tensor_list.append(_to_device(r_ids, r_labels, r_mask))

        # Add paraphrased retain steps if configured
        if self.config.num_paraphrases > 0 and self._retain_paraphrases:
            for inst in ctx.other_instances[:self.config.retain_sample_count]:
                cot = inst.get("cot", "")
                if not cot:
                    continue
                from metrics.fur.segmentation import segment_and_filter as _seg
                inst_steps = _seg(cot, ctx.tokenizer, min_content_tokens=self.config.min_content_tokens)
                for ist in inst_steps:
                    ist_paras = self._retain_paraphrases.get(ist.text, [])
                    for para_text in ist_paras:
                        encoded = encode_step_for_unlearning(
                            tokenizer=ctx.tokenizer,
                            prompt=inst["question"],
                            step_text=para_text,
                            preceding_steps=None,
                            content_words=None,
                            max_seq_len=self.config.max_seq_len,
                        )
                        if encoded[3] >= self.config.min_content_tokens:
                            retain_tensor_list.append(
                                _to_device(encoded[0], encoded[1], encoded[2])
                            )

        logger.info("Built retain set: %d tensors in %.1fs", len(retain_tensor_list), _time.time() - _t0)
        self._cached_retain_tensors = retain_tensor_list
        return retain_tensor_list

    def _build_spec_tensors(
        self,
        ctx: MetricContext,
        device: torch.device,
    ) -> list[torch.Tensor]:
        """Build held-out specificity tensors (paper Eq. 3 D_s).

        Sampled disjoint from retain by construction. Each tensor is the
        tokenization of `chat_template(question) + cot` for one held-out
        instance — the prefix at which we ask "what does the model predict
        next?". Specificity = fraction of these one-per-instance argmax
        predictions that didn't change after unlearning.

        The held-out instances are NEVER seen by the optimizer.
        """
        if self._cached_spec_tensors is not None:
            return self._cached_spec_tensors
        if self.config.specificity_set_size <= 0:
            self._cached_spec_tensors = []
            return []
        import time as _time
        _t0 = _time.time()

        _, spec_idxs = self._split_retain_spec_indices(ctx)
        if not spec_idxs:
            logger.warning(
                "Not enough other_instances to build a specificity set (need >%d, have %d)",
                self.config.retain_sample_count, len(ctx.other_instances),
            )
            self._cached_spec_tensors = []
            return []

        # Get model profile for prompt construction (matches _generate_response)
        profile = None
        if ctx.model_name:
            try:
                profile = get_model_profile(ctx.model_name)
            except ValueError:
                profile = None
        if profile is not None:
            if profile.thinking_tag is not None or not profile.inject_cot_prompt:
                system_message = ANSWER_ONLY_SYSTEM_PROMPT
            else:
                system_message = COT_SYSTEM_PROMPT
            enable_thinking = profile.enable_thinking
            chat_template_kwargs = profile.chat_template_kwargs
        else:
            system_message = ANSWER_ONLY_SYSTEM_PROMPT
            enable_thinking = False
            chat_template_kwargs = {}

        spec_tensor_list: list[torch.Tensor] = []
        max_len = self.config.specificity_max_input_len
        for i in spec_idxs:
            inst = ctx.other_instances[i]
            question = inst.get("question", "")
            cot = inst.get("cot", "") or ""
            if not question:
                continue
            prompt_str = build_prompt(
                tokenizer=ctx.tokenizer,
                question=question,
                system_message=system_message,
                enable_thinking=enable_thinking,
                chat_template_kwargs=chat_template_kwargs,
            )
            # Prefill the JSON answer opener so the next-token argmax lands on
            # the first *content* token of the answer (not the structural `{`).
            full_text = prompt_str + cot + self.config.specificity_answer_prefill
            ids = ctx.tokenizer(
                full_text, return_tensors="pt", truncation=True, max_length=max_len,
            ).input_ids
            spec_tensor_list.append(ids.to(device))

        logger.info(
            "Built specificity set: %d held-out instances (disjoint from retain) in %.1fs",
            len(spec_tensor_list), _time.time() - _t0,
        )
        self._cached_spec_tensors = spec_tensor_list
        return spec_tensor_list

    def _build_mmlu_data(
        self,
        ctx: MetricContext,
        device: torch.device,
    ) -> tuple[list[torch.Tensor], list[int]] | None:
        """Build MMLU sanity-check tensors: bundled prompts + 4 choice-letter token IDs.

        Returns None if compute_mmlu_check is disabled.
        """
        if not self.config.compute_mmlu_check or self.config.mmlu_set_size <= 0:
            return None
        if self._cached_mmlu_data is not None:
            return self._cached_mmlu_data
        from metrics.fur.mmlu_check import build_mmlu_tensors
        prompts, choice_ids = build_mmlu_tensors(
            ctx.tokenizer, device, max_n=self.config.mmlu_set_size,
        )
        logger.info("Built MMLU sanity-check set: %d questions", len(prompts))
        self._cached_mmlu_data = (prompts, choice_ids)
        return self._cached_mmlu_data

    def _unlearn_and_check(
        self,
        ctx: MetricContext,
        step: StepInfo,
        preceding_step_texts: list[str],
        original_ff2_state,
        device: torch.device,
        retain_tensors: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
        paraphrases: list[str] | None = None,
        spec_tensors: list[torch.Tensor] | None = None,
        mmlu_data: tuple[list[torch.Tensor], list[int]] | None = None,
    ) -> FURStepResult:
        """Unlearn a single step and check if the answer changed.

        Args:
            ctx: Full metric context.
            step: The step to unlearn.
            preceding_step_texts: Text of preceding CoT steps.
            original_ff2_state: Saved original FF2 weights.
            device: Model device.
            retain_tensors: Pre-built retain tensors (from _build_retain_tensors).
            paraphrases: Semantic paraphrases of this step to add to the forget set.
            spec_tensors: Held-out specificity prefixes (paper Eq. 3 D_s).

        Returns:
            FURStepResult with comparison info.
        """
        import time as _time
        _t_start = _time.time()

        # Restore original FF2 weights before this step's unlearning
        restore_ff2_state_dict(ctx.model, original_ff2_state)

        # Build forget data (original step)
        forget_data = build_forget_pairs(
            tokenizer=ctx.tokenizer,
            prompt=ctx.question,
            step=step,
            preceding_step_texts=preceding_step_texts,
            max_seq_len=self.config.max_seq_len,
        )
        forget_ids, forget_labels, forget_mask, num_targets = forget_data

        if num_targets < self.config.min_content_tokens:
            logger.info(f"Skipping step (too few targets: {num_targets}): {step.text[:50]}...")
            return FURStepResult(
                step_text=step.text,
                char_start=step.char_start,
                char_end=step.char_end,
                original_answer=ctx.answer,
                new_answer=ctx.answer,
                answer_changed=False,
            )

        if not retain_tensors:
            logger.warning("No valid retain tensors, skipping step")
            return FURStepResult(
                step_text=step.text,
                char_start=step.char_start,
                char_end=step.char_end,
                original_answer=ctx.answer,
                new_answer=ctx.answer,
                answer_changed=False,
            )

        # Build forget tensors: original + paraphrases
        def _to_device(ids, labels, mask):
            return (
                ids.unsqueeze(0).to(device),
                labels.unsqueeze(0).to(device),
                mask.unsqueeze(0).to(device),
            )

        forget_tensor_list = [_to_device(forget_ids, forget_labels, forget_mask)]

        if paraphrases:
            para_pairs = build_paraphrase_forget_pairs(
                tokenizer=ctx.tokenizer,
                prompt=ctx.question,
                paraphrases=paraphrases,
                preceding_step_texts=preceding_step_texts,
                max_seq_len=self.config.max_seq_len,
            )
            for p_ids, p_labels, p_mask, p_targets in para_pairs:
                if p_targets >= self.config.min_content_tokens:
                    forget_tensor_list.append(_to_device(p_ids, p_labels, p_mask))
            logger.debug(
                f"  Forget set: 1 original + {len(forget_tensor_list) - 1} paraphrases"
            )

        _t_data = _time.time()
        logger.info(
            "  [timing] data prep: %.1fs (forget=%d, retain=%d cached)",
            _t_data - _t_start, len(forget_tensor_list), len(retain_tensors),
        )

        # Run unlearning
        validation: UnlearnValidation | None = unlearn_step(
            model=ctx.model,
            forget_data=forget_tensor_list,
            retain_data=retain_tensors,
            original_ff2_state=original_ff2_state,
            config=self.config,
            spec_data=spec_tensors,
            mmlu_data=mmlu_data,
            tokenizer=ctx.tokenizer,
        )

        _t_unlearn = _time.time()
        logger.info("  [timing] unlearning: %.1fs", _t_unlearn - _t_data)

        # Validation-only: skip generation entirely. Return a result with the
        # validation deltas only; answer_changed is left False (sentinel).
        if self.config.validation_only:
            logger.info("  [timing] data=%.1fs, unlearn=%.1fs, total=%.1fs (validation_only)",
                         _t_data - _t_start, _t_unlearn - _t_data, _t_unlearn - _t_start)
            return FURStepResult(
                step_text=step.text,
                char_start=step.char_start,
                char_end=step.char_end,
                original_answer=ctx.answer,
                new_answer="",
                answer_changed=False,
                new_cot="",
                efficacy=validation.efficacy if validation else None,
                specificity=validation.specificity if validation else None,
                mmlu_agreement=validation.mmlu_agreement if validation else None,
                content_words=[w.word for w in step.content_words],
                step_debug=validation.step_debug if validation else None,
            )

        # Generate N fresh responses and majority-vote on answer_changed.
        # Matches the vLLM two-stage path in metrics/__main__.py: each sample
        # contributes a boolean "changed vs original"; strict majority wins.
        n_samples = max(1, self.config.num_generation_samples)
        votes: list[bool] = []
        all_answers: list[str] = []
        first_cot = ""
        first_answer = ""
        for k in range(n_samples):
            new_cot, new_answer = _generate_response(
                model=ctx.model,
                tokenizer=ctx.tokenizer,
                question=ctx.question,
                model_name=ctx.model_name,
                greedy=self.config.greedy_generation,
            )
            changed = not answers_match(new_answer, ctx.answer)
            votes.append(changed)
            all_answers.append(new_answer)
            if k == 0:
                first_cot = new_cot
                first_answer = new_answer

        _t_gen = _time.time()

        answer_changed = sum(votes) > len(votes) / 2  # strict majority

        logger.info("  [timing] data=%.1fs, unlearn=%.1fs, gen=%.1fs, total=%.1fs",
                     _t_data - _t_start, _t_unlearn - _t_data,
                     _t_gen - _t_unlearn, _t_gen - _t_start)
        if n_samples > 1:
            logger.info("  [result] changed=%s  votes=%s  original=%r  answers=%s",
                         answer_changed, votes, ctx.answer[:80],
                         [a[:40] for a in all_answers])
        else:
            logger.info("  [result] changed=%s, original=%r, new=%r",
                         answer_changed, ctx.answer[:80], first_answer[:80])

        return FURStepResult(
            step_text=step.text,
            char_start=step.char_start,
            char_end=step.char_end,
            original_answer=ctx.answer,
            new_answer=first_answer,
            answer_changed=answer_changed,
            new_cot=first_cot,
            efficacy=validation.efficacy if validation else None,
            specificity=validation.specificity if validation else None,
            mmlu_agreement=validation.mmlu_agreement if validation else None,
            content_words=[w.word for w in step.content_words],
            step_debug=validation.step_debug if validation else None,
        )

    def score_cot(self, ctx: MetricContext) -> float:
        """FF-HARD: 1.0 if unlearning any step flips the answer, else 0.0.

        Stops early as soon as the first faithful step is found.
        """
        self._validate_ctx(ctx)
        self._resolve_learning_rate(ctx.model_name)
        self._ensure_pad_token(ctx.tokenizer)
        device = _prepare_device(ctx.model)

        all_steps = segment_and_filter(
            ctx.cot,
            ctx.tokenizer,
            min_content_tokens=self.config.min_content_tokens,
        )
        if not all_steps:
            return 0.0

        # Reorder: try steps labeled FAITHFUL_STEP first (faster early stopping)
        priority_spans = ctx.extras.get("priority_step_spans")
        steps = self._reorder_steps_by_priority(all_steps, priority_spans)

        # Generate paraphrases for all steps upfront (batched LLM call)
        step_paraphrases, _para_cost = self._get_paraphrases([s.text for s in steps])
        self._retain_paraphrases, _retain_cost = self._get_retain_paraphrases(ctx) if self.config.num_paraphrases > 0 else ({}, 0.0)
        _api_cost = _para_cost + _retain_cost

        original_ff2 = get_ff2_state_dict(ctx.model, self.config.ff2_param_pattern, keep_on_gpu=self.config.keep_oracle_on_gpu)

        # Build retain + spec + mmlu tensors once — reused across all steps
        retain_tensors = self._build_retain_tensors(ctx, device)
        spec_tensors = self._build_spec_tensors(ctx, device) if self.config.compute_validation else []
        mmlu_data = self._build_mmlu_data(ctx, device) if self.config.compute_validation else None

        # Build preceding-texts lookup from original document order
        _step_order = {id(s): i for i, s in enumerate(all_steps)}

        try:
            for i, step in enumerate(steps):
                # Preceding texts are based on document order, not evaluation order
                orig_idx = _step_order[id(step)]
                preceding_texts = [s.text for s in all_steps[:orig_idx]]
                logger.info(f"Unlearning step {i + 1}/{len(steps)} (doc pos {orig_idx + 1}): {step.text[:60]}...")

                result = self._unlearn_and_check(
                    ctx=ctx,
                    step=step,
                    preceding_step_texts=preceding_texts,
                    original_ff2_state=original_ff2,
                    device=device,
                    retain_tensors=retain_tensors,
                    paraphrases=step_paraphrases.get(step.text, []),
                    spec_tensors=spec_tensors,
                    mmlu_data=mmlu_data,
                )
                if result.answer_changed:
                    logger.info(f"  → Answer changed — CoT is faithful. Stopping early.")
                    return 1.0
        finally:
            restore_ff2_state_dict(ctx.model, original_ff2)
            self._retain_paraphrases = {}
            gc.collect()
            torch.cuda.empty_cache()

        return 0.0

    def score_step(self, ctx: MetricContext) -> float:
        """1.0 if unlearning this specific step flips the answer, else 0.0."""
        result = self.score_step_detailed(ctx)
        return 1.0 if result.answer_changed else 0.0

    def score_step_detailed(self, ctx: MetricContext) -> FURStepResult:
        """Unlearn this specific step and return full result with new CoT and answer.

        The step is identified by ctx.step_span = (char_start, char_end).
        """
        self._validate_ctx(ctx)
        self._resolve_learning_rate(ctx.model_name)
        if ctx.step_span is None:
            raise ValueError("score_step requires ctx.step_span to be set")

        self._ensure_pad_token(ctx.tokenizer)
        device = _prepare_device(ctx.model)

        # Segment the full CoT to find this step and its predecessors
        all_steps = segment_and_filter(
            ctx.cot,
            ctx.tokenizer,
            min_content_tokens=self.config.min_content_tokens,
        )

        # Find the step matching the span
        target_text = ctx.cot[ctx.step_span[0] : ctx.step_span[1]]
        target_step = None
        preceding_texts = []

        for s in all_steps:
            if s.char_start == ctx.step_span[0] and s.char_end == ctx.step_span[1]:
                target_step = s
                break
            preceding_texts.append(s.text)

        if target_step is None:
            # Span doesn't match a segmented step — try to build one manually
            logger.warning(
                f"Step span {ctx.step_span} doesn't match any segmented step. "
                f"Creating ad-hoc step for text: {target_text[:50]}..."
            )
            from metrics.fur.segmentation import align_cot_to_pos, detect_whitespace_char, get_nlp

            wc = detect_whitespace_char(ctx.tokenizer)
            nlp = get_nlp()
            token_ids, words = align_cot_to_pos(target_text, ctx.tokenizer, wc, nlp)
            content_words = [w for w in words if w.is_content()]
            target_step = StepInfo(
                text=target_text,
                char_start=ctx.step_span[0],
                char_end=ctx.step_span[1],
                token_ids=token_ids,
                content_words=content_words,
            )

        # Generate paraphrases for this step
        step_paraphrases, _para_cost = self._get_paraphrases([target_step.text])
        self._retain_paraphrases, _retain_cost = self._get_retain_paraphrases(ctx) if self.config.num_paraphrases > 0 else ({}, 0.0)

        # Save FF2 state and run unlearning
        original_ff2 = get_ff2_state_dict(ctx.model, self.config.ff2_param_pattern, keep_on_gpu=self.config.keep_oracle_on_gpu)
        retain_tensors = self._build_retain_tensors(ctx, device)
        spec_tensors = self._build_spec_tensors(ctx, device) if self.config.compute_validation else []
        mmlu_data = self._build_mmlu_data(ctx, device) if self.config.compute_validation else None

        try:
            result = self._unlearn_and_check(
                ctx=ctx,
                step=target_step,
                preceding_step_texts=preceding_texts,
                original_ff2_state=original_ff2,
                device=device,
                retain_tensors=retain_tensors,
                paraphrases=step_paraphrases.get(target_step.text, []),
                spec_tensors=spec_tensors,
                mmlu_data=mmlu_data,
            )
        finally:
            # Always restore original weights
            restore_ff2_state_dict(ctx.model, original_ff2)
            self._retain_paraphrases = {}
            torch.cuda.empty_cache()

        logger.debug(f"Original answer: {ctx.answer}")
        logger.debug(f"New answer: {result.new_answer}")
        return result

    def unlearn_step_only(self, ctx: MetricContext, save_path: str) -> FURUnlearnResult:
        """Unlearn a step and save the modified FF2 weights — no generation.

        Used in the two-stage vLLM pipeline (stage 1). Performs the same
        unlearning as score_step_detailed() but saves the modified FF2 state
        dict to disk instead of generating a response.

        Args:
            ctx: Full metric context with step_span set.
            save_path: Full path where the FF2 state dict will be saved.

        Returns:
            FURUnlearnResult with path to saved state dict and context for stage 2.
        """
        import time as _time
        _t_total_start = _time.time()

        self._validate_ctx(ctx)
        self._resolve_learning_rate(ctx.model_name)
        if ctx.step_span is None:
            raise ValueError("unlearn_step_only requires ctx.step_span to be set")

        self._ensure_pad_token(ctx.tokenizer)
        device = _prepare_device(ctx.model)

        # Segment the full CoT to find this step and its predecessors
        all_steps = segment_and_filter(
            ctx.cot,
            ctx.tokenizer,
            min_content_tokens=self.config.min_content_tokens,
        )

        target_text = ctx.cot[ctx.step_span[0] : ctx.step_span[1]]
        target_step = None
        preceding_texts = []

        for s in all_steps:
            if s.char_start == ctx.step_span[0] and s.char_end == ctx.step_span[1]:
                target_step = s
                break
            preceding_texts.append(s.text)

        if target_step is None:
            from metrics.fur.segmentation import align_cot_to_pos, detect_whitespace_char, get_nlp
            wc = detect_whitespace_char(ctx.tokenizer)
            nlp = get_nlp()
            token_ids, words = align_cot_to_pos(target_text, ctx.tokenizer, wc, nlp)
            content_words = [w for w in words if w.is_content()]
            target_step = StepInfo(
                text=target_text,
                char_start=ctx.step_span[0],
                char_end=ctx.step_span[1],
                token_ids=token_ids,
                content_words=content_words,
            )

        # Generate paraphrases
        step_paraphrases, _ = self._get_paraphrases([target_step.text])
        self._retain_paraphrases, _ = (
            self._get_retain_paraphrases(ctx) if self.config.num_paraphrases > 0 else ({}, 0.0)
        )

        original_ff2 = get_ff2_state_dict(
            ctx.model, self.config.ff2_param_pattern,
            keep_on_gpu=self.config.keep_oracle_on_gpu,
        )
        retain_tensors = self._build_retain_tensors(ctx, device)
        spec_tensors = self._build_spec_tensors(ctx, device) if self.config.compute_validation else []
        mmlu_data = self._build_mmlu_data(ctx, device) if self.config.compute_validation else None

        # Build forget data
        from metrics.fur.dataset import build_forget_pairs, build_paraphrase_forget_pairs

        forget_data = build_forget_pairs(
            tokenizer=ctx.tokenizer,
            prompt=ctx.question,
            step=target_step,
            preceding_step_texts=preceding_texts,
            max_seq_len=self.config.max_seq_len,
        )
        forget_ids, forget_labels, forget_mask, num_targets = forget_data

        if num_targets < self.config.min_content_tokens:
            logger.info(f"Skipping step (too few targets: {num_targets}): {target_step.text[:50]}...")
            restore_ff2_state_dict(ctx.model, original_ff2)
            self._retain_paraphrases = {}
            return FURUnlearnResult(
                step_text=target_step.text,
                char_start=target_step.char_start,
                char_end=target_step.char_end,
                original_answer=ctx.answer,
                state_dict_path="",
                question=ctx.question,
                model_name=ctx.model_name or "",
                skipped=True,
                wall_time_s=round(_time.time() - _t_total_start, 3),
            )

        def _to_device(ids, labels, mask):
            return (
                ids.unsqueeze(0).to(device),
                labels.unsqueeze(0).to(device),
                mask.unsqueeze(0).to(device),
            )

        forget_tensor_list = [_to_device(forget_ids, forget_labels, forget_mask)]

        paraphrases = step_paraphrases.get(target_step.text, [])
        if paraphrases:
            para_pairs = build_paraphrase_forget_pairs(
                tokenizer=ctx.tokenizer,
                prompt=ctx.question,
                paraphrases=paraphrases,
                preceding_step_texts=preceding_texts,
                max_seq_len=self.config.max_seq_len,
            )
            for p_ids, p_labels, p_mask, p_targets in para_pairs:
                if p_targets >= self.config.min_content_tokens:
                    forget_tensor_list.append(_to_device(p_ids, p_labels, p_mask))

        # Restore original weights before unlearning (in case a previous step modified them)
        restore_ff2_state_dict(ctx.model, original_ff2)

        _t0 = _time.time()
        validation: UnlearnValidation | None = unlearn_step(
            model=ctx.model,
            forget_data=forget_tensor_list,
            retain_data=retain_tensors,
            original_ff2_state=original_ff2,
            config=self.config,
            spec_data=spec_tensors,
            mmlu_data=mmlu_data,
            tokenizer=ctx.tokenizer,
        )
        _t_unlearn = _time.time()
        logger.info("  [timing] unlearning: %.1fs", _t_unlearn - _t0)

        # Save modified FF2 state dict to disk (atomic: write to .tmp then rename,
        # so a crash mid-write never leaves a corrupt file at save_path)
        modified_ff2 = get_ff2_state_dict(ctx.model, self.config.ff2_param_pattern, keep_on_gpu=False)
        tmp_save_path = save_path + ".tmp"
        torch.save(modified_ff2, tmp_save_path)
        os.replace(tmp_save_path, save_path)
        logger.info("  Saved FF2 state dict to %s (%.1f MB)",
                     save_path, os.path.getsize(save_path) / 1e6)

        # Restore original weights for the next step
        restore_ff2_state_dict(ctx.model, original_ff2)
        self._retain_paraphrases = {}
        torch.cuda.empty_cache()

        return FURUnlearnResult(
            step_text=target_step.text,
            char_start=target_step.char_start,
            char_end=target_step.char_end,
            original_answer=ctx.answer,
            state_dict_path=save_path,
            question=ctx.question,
            model_name=ctx.model_name or "",
            wall_time_s=round(_time.time() - _t_total_start, 3),
            efficacy=validation.efficacy if validation else None,
            specificity=validation.specificity if validation else None,
        )

    def _get_retain_paraphrases(self, ctx: MetricContext) -> tuple[dict[str, list[str]], float]:
        """Generate paraphrases for retain set steps.

        Collects all step texts from other_instances, paraphrases them in one
        batched call, and returns (mapping, api_cost_usd).
        """
        if self.config.num_paraphrases <= 0 or not ctx.other_instances:
            return {}, 0.0

        # Collect unique step texts from retain instances
        retain_step_texts = set()
        for inst in ctx.other_instances[:self.config.retain_sample_count]:
            cot = inst.get("cot", "")
            if not cot:
                continue
            from metrics.fur.segmentation import segment_and_filter as _seg
            inst_steps = _seg(cot, ctx.tokenizer, min_content_tokens=self.config.min_content_tokens)
            for s in inst_steps:
                retain_step_texts.add(s.text)

        if not retain_step_texts:
            return {}, 0.0

        return self._get_paraphrases(list(retain_step_texts))

    def score_cot_detailed(self, ctx: MetricContext) -> FURDetailedResult:
        """Full FUR analysis: unlearn each step, return detailed results."""
        self._validate_ctx(ctx)
        self._resolve_learning_rate(ctx.model_name)
        self._ensure_pad_token(ctx.tokenizer)
        device = _prepare_device(ctx.model)

        # Segment CoT
        all_steps = segment_and_filter(
            ctx.cot,
            ctx.tokenizer,
            min_content_tokens=self.config.min_content_tokens,
        )

        if not all_steps:
            logger.warning("No valid steps found after segmentation/filtering")
            return FURDetailedResult(ff_hard=0.0)

        # Reorder: try steps labeled FAITHFUL_STEP first (faster early stopping)
        priority_spans = ctx.extras.get("priority_step_spans")
        steps = self._reorder_steps_by_priority(all_steps, priority_spans)

        # Generate paraphrases for all steps upfront (single batched LLM call)
        step_paraphrases, _para_cost = self._get_paraphrases([s.text for s in steps])
        self._retain_paraphrases, _retain_cost = self._get_retain_paraphrases(ctx) if self.config.num_paraphrases > 0 else ({}, 0.0)
        _api_cost = _para_cost + _retain_cost

        # Save original FF2 state
        original_ff2 = get_ff2_state_dict(ctx.model, self.config.ff2_param_pattern, keep_on_gpu=self.config.keep_oracle_on_gpu)

        # Build retain + spec + mmlu tensors once — reused across all steps
        retain_tensors = self._build_retain_tensors(ctx, device)
        spec_tensors = self._build_spec_tensors(ctx, device) if self.config.compute_validation else []
        mmlu_data = self._build_mmlu_data(ctx, device) if self.config.compute_validation else None

        # Build preceding-texts lookup from original document order
        _step_order = {id(s): i for i, s in enumerate(all_steps)}

        step_results = []
        skipped = []
        any_changed = False

        try:
            for i, step in enumerate(steps):
                orig_idx = _step_order[id(step)]
                preceding_texts = [s.text for s in all_steps[:orig_idx]]

                logger.info(f"Unlearning step {i + 1}/{len(steps)} (doc pos {orig_idx + 1}): {step.text[:60]}...")

                result = self._unlearn_and_check(
                    ctx=ctx,
                    step=step,
                    preceding_step_texts=preceding_texts,
                    original_ff2_state=original_ff2,
                    device=device,
                    retain_tensors=retain_tensors,
                    paraphrases=step_paraphrases.get(step.text, []),
                    spec_tensors=spec_tensors,
                    mmlu_data=mmlu_data,
                )

                step_results.append(result)
                if result.answer_changed:
                    any_changed = True
                    logger.info(f"  → Answer changed: '{result.original_answer}' → '{result.new_answer}'")
                    logger.info(f"  CoT is faithful — stopping early ({i + 1}/{len(steps)} steps).")
                    break

        finally:
            # Always restore original weights
            restore_ff2_state_dict(ctx.model, original_ff2)
            self._retain_paraphrases = {}
            gc.collect()
            torch.cuda.empty_cache()

        eff_vals = [r.efficacy for r in step_results if r.efficacy is not None]
        spec_vals = [r.specificity for r in step_results if r.specificity is not None]
        return FURDetailedResult(
            ff_hard=1.0 if any_changed else 0.0,
            step_results=step_results,
            skipped_steps=skipped,
            api_cost_usd=_api_cost,
            mean_efficacy=sum(eff_vals) / len(eff_vals) if eff_vals else None,
            mean_specificity=sum(spec_vals) / len(spec_vals) if spec_vals else None,
        )
