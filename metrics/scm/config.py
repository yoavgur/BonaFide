"""Configuration for the SCM-based faithfulness metric."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class SCMConfig:
    """Hyperparameters for the SCM causal analysis metric.

    Reference: Bao et al., "How Likely Do LLMs with CoT Mimic Human Reasoning?"
    (COLING 2025).
    """

    # CoT corruption (H1) ------------------------------------------------
    corruption_ratio: float = 1.0  # Fraction of CoT sentences to replace (from end). 1.0 = full-CoT corruption, matching Bao et al.'s "random CoT" intervention.
    corruption_seed: int | None = None  # For reproducible donor selection

    # Instruction modification (H2) --------------------------------------
    roles: tuple[str, ...] = ("detective", "chef", "judge", "artist")

    # Generation ----------------------------------------------------------
    backend: str = "vllm"  # "vllm" | "hf" — NO silent fallback
    generation_temperature: float = 0.0  # Greedy decoding for reproducibility
    tensor_parallel_size: int = 1  # For vLLM multi-GPU

    def __post_init__(self) -> None:
        if self.backend not in ("vllm", "hf"):
            raise ValueError(
                f"backend must be 'vllm' or 'hf', got {self.backend!r}"
            )
        if not 0.0 < self.corruption_ratio <= 1.0:
            raise ValueError(
                f"corruption_ratio must be in (0, 1], got {self.corruption_ratio}"
            )
        if not self.roles:
            raise ValueError("roles must be non-empty")
