"""Base interface for faithfulness metrics.

All faithfulness metrics implement the FaithfulnessMetric ABC and receive
a MetricContext containing everything needed for evaluation.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class MetricContext:
    """Rich context object passed to all faithfulness metrics.

    Metrics ignore fields they don't need. For example, a purely LLM-judge-based
    metric only needs question/cot/answer, while a parametric metric like FUR
    also needs model weights and other dataset instances.
    """

    # Core — required by all metrics
    question: str  # The original question text (user message content)
    cot: str  # Full chain-of-thought text
    answer: str  # Model's parsed final answer

    # For step-level metrics
    step_span: tuple[int, int] | None = None  # (char_start, char_end) into cot

    # For parametric metrics (FUR, etc.) — require model weight access
    model: Any = None  # HF AutoModelForCausalLM (loaded, on device)
    tokenizer: Any = None  # HF AutoTokenizer
    model_name: str | None = None  # HF model identifier (for registry lookup)

    # Dataset context (for retain set construction, etc.)
    other_instances: list[dict] | None = None  # Other {"question", "cot", "answer"} dicts

    # Generation config (for re-generation after unlearning)
    generation_config: Any = None  # GenerationConfig or equivalent

    # Extensible
    extras: dict = field(default_factory=dict)


class FaithfulnessMetric(ABC):
    """Abstract base class for all faithfulness metrics.

    Each metric must support at least one of CoT-level or step-level scoring.
    """

    @abstractmethod
    def score_cot(self, ctx: MetricContext) -> float:
        """Score entire CoT faithfulness (0 to 1).

        Higher = more faithful.
        """
        ...

    @abstractmethod
    def score_step(self, ctx: MetricContext) -> float:
        """Score a single step's faithfulness (0 to 1).

        The step is identified by ctx.step_span = (char_start, char_end) into ctx.cot.
        Higher = more faithful.
        """
        ...

    @property
    @abstractmethod
    def name(self) -> str:
        """Short identifier for this metric (e.g. 'fur', 'add_mistake')."""
        ...

    @property
    @abstractmethod
    def supports_cot_scoring(self) -> bool:
        """Whether this metric supports CoT-level scoring."""
        ...

    @property
    @abstractmethod
    def supports_step_scoring(self) -> bool:
        """Whether this metric supports step-level scoring."""
        ...

    @property
    @abstractmethod
    def requires_model_weights(self) -> bool:
        """Whether this metric needs model/tokenizer in MetricContext."""
        ...
