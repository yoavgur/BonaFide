"""Configuration for CC-SHAP metric."""

from __future__ import annotations

from dataclasses import dataclass

VALID_DIVERGENCES = {"cosine", "pearson", "mse", "variance", "kl", "jsd"}


@dataclass
class CCSHAPConfig:
    """Hyperparameters for CC-SHAP faithfulness metric.

    Attributes:
        divergence: Divergence measure for score_cot(). One of "cosine" (default,
            paper Eq. 5), "pearson", "mse", "variance", "kl", "jsd".
        max_evals: Evaluation budget for the Partition explainer. Higher = more
            accurate SHAP values but slower. Default 500 matches the SHAP library
            default. The Partition explainer prioritizes the most informative
            coalitions first, so reducing to 200 still captures coarse contribution
            structure — useful when speed matters more than token-level precision.
        batch_size: Max number of masked inputs per forward pass in TeacherForcing.
            Acts as the ceiling when auto_batch is enabled; higher = better GPU
            utilization. Increase if GPU memory allows.
        auto_batch: When True, shrink the forward-pass batch size automatically
            based on free GPU memory (capped by batch_size). Per-call measurement,
            so it adapts to variable target lengths and residual activations.
    """

    divergence: str = "cosine"
    max_evals: int = 200
    batch_size: int = 64
    auto_batch: bool = True

    def __post_init__(self) -> None:
        if self.divergence not in VALID_DIVERGENCES:
            raise ValueError(
                f"Invalid divergence '{self.divergence}'. Must be one of: {sorted(VALID_DIVERGENCES)}"
            )
