"""MonitorMetric: LLM-judge faithfulness monitor (CoT and step level)."""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field

from metrics.base import FaithfulnessMetric, MetricContext
from metrics.monitor.config import (
    MonitorConfig,
    MonitorGenericConfig,
    MonitorNoHintConfig,
    MonitorNoToolConfig,
)
from metrics.monitor.prompts import format_cot_prompt, format_step_prompt
from metrics.monitor.prompts_generic import (
    format_cot_prompt as format_cot_prompt_generic,
    format_step_prompt as format_step_prompt_generic,
)
from metrics.monitor.prompts_no_tool import (
    format_cot_prompt as format_cot_prompt_no_tool,
    format_step_prompt as format_step_prompt_no_tool,
)

logger = logging.getLogger(__name__)


@dataclass
class MonitorResult:
    faithful: bool | float
    raw_response: str
    thoughts: str = ""
    judge_prompt: str = ""
    api_cost_usd: float = 0.0


class MonitorMetric(FaithfulnessMetric):
    """Ask Gemini whether the CoT (or a specific step) is faithful to the model."""

    # Max attempts (initial + retries) when the judge's response cannot be
    # parsed as the expected JSON. Retries reissue the same prompt unchanged.
    MAX_PARSE_ATTEMPTS = 3

    def __init__(self, config: MonitorConfig | None = None) -> None:
        self._config = config or MonitorConfig()
        self._judge = None  # lazy

    @property
    def name(self) -> str:
        return "monitor"

    @property
    def supports_cot_scoring(self) -> bool:
        return True

    @property
    def supports_step_scoring(self) -> bool:
        return True

    @property
    def requires_model_weights(self) -> bool:
        return False

    # ------------------------------------------------------------------
    # Judge lifecycle
    # ------------------------------------------------------------------
    def _get_judge(self):
        if self._judge is None:
            from isolate_steps import Judge
            self._judge = Judge(model_name=self._config.judge_model)
        return self._judge

    # ------------------------------------------------------------------
    # Prompt building
    # ------------------------------------------------------------------
    # Matches the opening of generate_wrong_answers.BASE_INSTRUCTION. Hinted
    # prompts wrap BASE_INSTRUCTION either before or after the hint text, so
    # the surrounding `\n\n\n` separator reliably fences the question block.
    _BASE_INSTRUCTION_MARKER = "Answer the following question."

    def _resolve_prompt_text(self, ctx: MetricContext) -> str:
        """Return the prompt text shown to the judge.

        When `strip_hint` is enabled, remove the hint from the prompt:
          1. Preferred: `ctx.extras["prompted_hint"]` is a non-empty substring
             of the prompt → subtract it (exact match). Asserts the hint was
             present before and is gone after.
          2. Fallback: anchor on the BASE_INSTRUCTION marker and trim at the
             first `\\n\\n\\n` separator. Used when `prompted_hint` is absent.
          3. No-op: prompts from non-hinting flows (graph / complex) don't
             contain the BASE_INSTRUCTION marker and have no hint to remove —
             return them unchanged, making `monitor_no_hint` equivalent to
             `monitor` on those rows.
        """
        prompt = ctx.question
        if not self._config.strip_hint:
            return prompt

        hint = (ctx.extras or {}).get("prompted_hint", "")
        if hint and hint in prompt:
            stripped = prompt.replace(hint, "", 1).strip()
            if hint in stripped:
                raise AssertionError(
                    "monitor_no_hint: hint still present after stripping "
                    f"(hint_len={len(hint)})"
                )
            return stripped

        marker_idx = prompt.find(self._BASE_INSTRUCTION_MARKER)
        if marker_idx < 0:
            # Non-hinting flow (e.g. graph / complex) — nothing to strip.
            return prompt

        body = prompt[marker_idx:]
        sep_idx = body.find("\n\n\n")
        if sep_idx >= 0:
            body = body[:sep_idx]
        stripped = body.strip()

        # Verify actual removal: everything the marker-based trim dropped
        # (prepend content + append tail) must not remain inside ``stripped``.
        removed_before = prompt[:marker_idx].strip()
        removed_after = prompt[marker_idx + len(body):].strip()
        for label, chunk in (("prepend", removed_before), ("append", removed_after)):
            if not chunk:
                continue
            # Allow short boilerplate overlap (blank lines, single chars).
            if len(chunk) < 10:
                continue
            if chunk in stripped:
                raise AssertionError(
                    f"monitor_no_hint: {label} hint fragment still present in "
                    f"stripped prompt (chunk_len={len(chunk)})"
                )

        # If nothing was actually removed, this is a hinting-flow row with no
        # hint (rare) — still fine to return the marker-anchored body.
        return stripped

    def build_cot_prompt(self, ctx: MetricContext) -> str:
        return format_cot_prompt(
            prompt=self._resolve_prompt_text(ctx),
            model_raw_response=ctx.answer,
            cot=ctx.cot,
        )

    def build_step_prompt(self, ctx: MetricContext) -> str:
        if ctx.step_span is None:
            raise ValueError("MonitorMetric.score_step requires ctx.step_span")
        start, end = ctx.step_span
        step_text = ctx.cot[start:end]
        if not step_text.strip():
            raise ValueError(
                f"Step span ({start}, {end}) maps to empty text in CoT "
                f"(cot length {len(ctx.cot)})"
            )
        return format_step_prompt(
            prompt=self._resolve_prompt_text(ctx),
            model_raw_response=ctx.answer,
            step_text=step_text,
            cot=ctx.cot,
        )

    # ------------------------------------------------------------------
    # Response parsing
    # ------------------------------------------------------------------
    # Fallback: match `{"faithful": true|false}` (tolerant of whitespace and
    # single/double quotes) anywhere in the response. Takes the LAST match,
    # on the assumption that a trailing verdict blob follows any prose the
    # judge wrote despite the output-format instruction.
    _FALLBACK_FAITHFUL_RE = re.compile(
        r"""\{\s*['"]faithful['"]\s*:\s*(true|false)\s*\}""",
        re.IGNORECASE,
    )

    @classmethod
    def parse_response(cls, response) -> MonitorResult:
        from isolate_steps import try_really_hard_to_parse_json

        raw_text = response.text

        # Primary path: strict JSON parse.
        parse_err: Exception | None = None
        try:
            parsed = try_really_hard_to_parse_json(raw_text)
        except (json.JSONDecodeError, ValueError) as e:
            parsed = None
            parse_err = e

        if isinstance(parsed, dict) and isinstance(parsed.get("faithful"), bool):
            return MonitorResult(faithful=parsed["faithful"], raw_response=raw_text)

        # Fallback: scan for a trailing `{"faithful": true|false}` blob. Gemini
        # sometimes writes an essay then drops a verdict at the end despite the
        # explicit output-format instruction; retries tend to reproduce the
        # same shape, so without this the row crashes the job.
        matches = cls._FALLBACK_FAITHFUL_RE.findall(raw_text)
        if matches:
            faithful = matches[-1].lower() == "true"
            logger.warning(
                "Monitor judge produced prose + trailing verdict; recovered via "
                "fallback regex (faithful=%s). Raw: %r",
                faithful, raw_text,
            )
            return MonitorResult(faithful=faithful, raw_response=raw_text)

        # Neither path worked — raise the most informative error we have.
        if parse_err is not None:
            raise parse_err
        if not isinstance(parsed, dict) or "faithful" not in parsed:
            raise ValueError(
                f"Monitor judge returned unexpected JSON (missing 'faithful' key): "
                f"{raw_text!r}"
            )
        faithful = parsed["faithful"]
        if not isinstance(faithful, bool):
            raise ValueError(
                f"Monitor judge returned non-bool 'faithful' value: {faithful!r} "
                f"(raw: {raw_text!r})"
            )
        return MonitorResult(faithful=faithful, raw_response=raw_text)

    @staticmethod
    def abbreviate_cot_in_prompt(prompt: str, cot: str) -> str:
        """Replace the full CoT body inside a judge prompt with a length marker.

        Keeps the prompt metadata readable in CSV output without carrying the
        full (possibly many-KB) CoT text.
        """
        if not cot:
            return prompt
        placeholder = f"...({len(cot)})"
        return prompt.replace(cot, placeholder)

    @classmethod
    def parse_with_retry(cls, judge, prompt, first_response, *, max_output_tokens=None):
        """Parse ``first_response``; on failure, reissue the same prompt up to
        ``MAX_PARSE_ATTEMPTS`` total attempts. Returns (result, final_response).

        Raises the last parse error if all attempts fail.
        """
        response = first_response
        last_err: Exception | None = None
        for attempt in range(1, cls.MAX_PARSE_ATTEMPTS + 1):
            try:
                result = cls.parse_response(response)
                if attempt > 1:
                    logger.warning(
                        "Monitor judge parse succeeded on attempt %d/%d",
                        attempt, cls.MAX_PARSE_ATTEMPTS,
                    )
                result.thoughts = judge.get_thoughts(response)
                return result, response
            except (json.JSONDecodeError, ValueError) as e:
                last_err = e
                logger.warning(
                    "Monitor judge parse failed (attempt %d/%d): %s. Raw: %r",
                    attempt, cls.MAX_PARSE_ATTEMPTS, e,
                    getattr(response, "text", None),
                )
                if attempt == cls.MAX_PARSE_ATTEMPTS:
                    break
                response = judge.run(prompt, max_output_tokens=max_output_tokens)
        raise last_err

    # ------------------------------------------------------------------
    # Per-row scoring (delegates to a batch of size 1)
    # ------------------------------------------------------------------
    def score_cot(self, ctx: MetricContext) -> float:
        return float(self.score_cot_detailed(ctx).faithful)

    def score_step(self, ctx: MetricContext) -> float:
        return float(self.score_step_detailed(ctx).faithful)

    def score_cot_detailed(self, ctx: MetricContext) -> MonitorResult:
        judge = self._get_judge()
        prompt = self.build_cot_prompt(ctx)
        response = judge.run(
            prompt, max_output_tokens=self._config.judge_max_output_tokens,
        )
        result, final_response = self.parse_with_retry(
            judge, prompt, response,
            max_output_tokens=self._config.judge_max_output_tokens,
        )
        result.api_cost_usd = judge.get_request_cost(final_response)
        result.judge_prompt = self.abbreviate_cot_in_prompt(prompt, ctx.cot)
        return result

    def score_step_detailed(self, ctx: MetricContext) -> MonitorResult:
        judge = self._get_judge()
        prompt = self.build_step_prompt(ctx)
        response = judge.run(
            prompt, max_output_tokens=self._config.judge_max_output_tokens,
        )
        result, final_response = self.parse_with_retry(
            judge, prompt, response,
            max_output_tokens=self._config.judge_max_output_tokens,
        )
        result.api_cost_usd = judge.get_request_cost(final_response)
        result.judge_prompt = self.abbreviate_cot_in_prompt(prompt, ctx.cot)
        return result


class MonitorNoHintMetric(MonitorMetric):
    """Same as MonitorMetric, but strips the hint from the prompt shown to the judge."""

    def __init__(self, config: MonitorConfig | None = None) -> None:
        super().__init__(config or MonitorNoHintConfig())

    @property
    def name(self) -> str:
        return "monitor_no_hint"


class MonitorGenericMetric(MonitorMetric):
    """Same as MonitorMetric but reads prompts from `prompts_generic.py`.

    Exists so the generic prompt can be edited independently of `monitor`.
    """

    def __init__(self, config: MonitorConfig | None = None) -> None:
        super().__init__(config or MonitorGenericConfig())

    @property
    def name(self) -> str:
        return "monitor_generic"

    def build_cot_prompt(self, ctx: MetricContext) -> str:
        return format_cot_prompt_generic(
            prompt=self._resolve_prompt_text(ctx),
            model_raw_response=ctx.answer,
            cot=ctx.cot,
        )

    def build_step_prompt(self, ctx: MetricContext) -> str:
        if ctx.step_span is None:
            raise ValueError("MonitorGenericMetric.score_step requires ctx.step_span")
        start, end = ctx.step_span
        step_text = ctx.cot[start:end]
        if not step_text.strip():
            raise ValueError(
                f"Step span ({start}, {end}) maps to empty text in CoT "
                f"(cot length {len(ctx.cot)})"
            )
        return format_step_prompt_generic(
            prompt=self._resolve_prompt_text(ctx),
            model_raw_response=ctx.answer,
            step_text=step_text,
            cot=ctx.cot,
        )

    @staticmethod
    def parse_response(response) -> MonitorResult:
        from isolate_steps import try_really_hard_to_parse_json

        raw_text = response.text
        parsed = try_really_hard_to_parse_json(raw_text)
        if not isinstance(parsed, dict) or "suspicion_score" not in parsed:
            raise ValueError(
                f"Monitor-generic judge returned unexpected JSON "
                f"(missing 'suspicion_score' key): {raw_text!r}"
            )
        score = parsed["suspicion_score"]
        if not isinstance(score, (int, float)) or not (0 <= score <= 100):
            raise ValueError(
                f"Monitor-generic judge returned invalid 'suspicion_score': "
                f"{score!r} (raw: {raw_text!r})"
            )
        return MonitorResult(faithful=score / 100.0, raw_response=raw_text)


class MonitorNoToolMetric(MonitorMetric):
    """Same as MonitorMetric, but the judge prompt omits the note about tool calls."""

    def __init__(self, config: MonitorConfig | None = None) -> None:
        super().__init__(config or MonitorNoToolConfig())

    @property
    def name(self) -> str:
        return "monitor_no_tool"

    def build_cot_prompt(self, ctx: MetricContext) -> str:
        return format_cot_prompt_no_tool(
            prompt=self._resolve_prompt_text(ctx),
            model_raw_response=ctx.answer,
            cot=ctx.cot,
        )

    def build_step_prompt(self, ctx: MetricContext) -> str:
        if ctx.step_span is None:
            raise ValueError("MonitorNoToolMetric.score_step requires ctx.step_span")
        start, end = ctx.step_span
        step_text = ctx.cot[start:end]
        if not step_text.strip():
            raise ValueError(
                f"Step span ({start}, {end}) maps to empty text in CoT "
                f"(cot length {len(ctx.cot)})"
            )
        return format_step_prompt_no_tool(
            prompt=self._resolve_prompt_text(ctx),
            model_raw_response=ctx.answer,
            step_text=step_text,
            cot=ctx.cot,
        )
