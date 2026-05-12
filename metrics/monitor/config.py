from __future__ import annotations

from dataclasses import dataclass


@dataclass
class MonitorConfig:
    """Configuration for the LLM-judge MonitorMetric."""

    judge_model: str = "gemini-3-flash-preview"
    judge_batch_size: int = 80
    judge_max_output_tokens: int | None = None
    # When True, strip the hint (row's `prompted_hint` column) out of the
    # prompt before showing it to the judge. Enables the `monitor_no_hint`
    # variant without duplicating the metric class.
    strip_hint: bool = False


@dataclass
class MonitorNoHintConfig(MonitorConfig):
    """Same as MonitorConfig, but hint-stripping is on by default."""

    strip_hint: bool = True


@dataclass
class MonitorGenericConfig(MonitorConfig):
    """Same as MonitorConfig; used by MonitorGenericMetric (independent prompt)."""

    pass


@dataclass
class MonitorNoToolConfig(MonitorConfig):
    """Same as MonitorConfig; used by MonitorNoToolMetric (prompt without tool-call note)."""

    pass
