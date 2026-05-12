"""Configuration for the simulatability faithfulness metric."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class SimulatabilityConfig:
    """Hyperparameters for the simulatability metric.

    Reference: Chan et al., "FRAME: Evaluating Rationale-Label Consistency
    Metrics for Free-Text Rationales" (arXiv 2207.00779).
    """

    # Generation ----------------------------------------------------------
    backend: str = "vllm"  # "vllm" | "hf" — NO silent fallback
    generation_temperature: float = 0.0  # Greedy decoding for reproducibility
    tensor_parallel_size: int = 1  # For vLLM multi-GPU

    def __post_init__(self) -> None:
        if self.backend not in ("vllm", "hf"):
            raise ValueError(
                f"backend must be 'vllm' or 'hf', got {self.backend!r}"
            )
