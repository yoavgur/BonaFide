"""Divergence measures for comparing CC-SHAP contribution profiles.

Implements the 6 divergence metrics from the CC-SHAP paper for comparing
input contribution distributions between prediction and explanation.
"""

from __future__ import annotations

from typing import Callable

import numpy as np
from numpy.typing import NDArray
from scipy import spatial, special, stats
from sklearn.metrics import mean_squared_error


def cosine_distance(
    c_pred: NDArray[np.float64],
    c_expl: NDArray[np.float64],
) -> float:
    """Cosine distance: 1 - cosine_similarity. Range [0, 2]."""
    # Handle zero vectors
    if np.allclose(c_pred, 0) or np.allclose(c_expl, 0):
        return 1.0
    return float(spatial.distance.cosine(c_pred, c_expl))


def pearson_distance(
    c_pred: NDArray[np.float64],
    c_expl: NDArray[np.float64],
) -> float:
    """Pearson correlation distance: 1 - correlation. Range [0, 2]."""
    if np.std(c_pred) < 1e-12 or np.std(c_expl) < 1e-12:
        return 1.0
    return float(spatial.distance.correlation(c_pred, c_expl))


def mse_divergence(
    c_pred: NDArray[np.float64],
    c_expl: NDArray[np.float64],
) -> float:
    """Mean squared error. Range [0, ∞)."""
    return float(mean_squared_error(c_pred, c_expl))


def variance_of_squared_errors(
    c_pred: NDArray[np.float64],
    c_expl: NDArray[np.float64],
) -> float:
    """Variance of squared errors (spread of residuals)."""
    mse = mean_squared_error(c_pred, c_expl)
    squared_errors = (c_pred - c_expl) ** 2
    return float(np.sum((squared_errors - mse) ** 2) / len(c_pred))


def kl_divergence(
    c_pred: NDArray[np.float64],
    c_expl: NDArray[np.float64],
) -> float:
    """KL divergence after softmax normalisation. KL(Q_expl || P_pred)."""
    p = special.softmax(c_pred)
    q = special.softmax(c_expl)
    return float(stats.entropy(q, p))


def jsd_divergence(
    c_pred: NDArray[np.float64],
    c_expl: NDArray[np.float64],
) -> float:
    """Jensen-Shannon divergence after softmax normalisation. Range [0, 1]."""
    p = special.softmax(c_pred)
    q = special.softmax(c_expl)
    return float(spatial.distance.jensenshannon(p, q) ** 2)


DIVERGENCE_REGISTRY: dict[str, Callable] = {
    "cosine": cosine_distance,
    "pearson": pearson_distance,
    "mse": mse_divergence,
    "variance": variance_of_squared_errors,
    "kl": kl_divergence,
    "jsd": jsd_divergence,
}


def compute_divergence(
    c_pred: NDArray[np.float64],
    c_expl: NDArray[np.float64],
    method: str = "cosine",
) -> float:
    """Compute divergence between two contribution vectors using the named method."""
    if method not in DIVERGENCE_REGISTRY:
        raise ValueError(f"Unknown divergence '{method}'. Choose from: {list(DIVERGENCE_REGISTRY)}")
    return DIVERGENCE_REGISTRY[method](c_pred, c_expl)


def compute_all_divergences(
    c_pred: NDArray[np.float64],
    c_expl: NDArray[np.float64],
) -> dict[str, float]:
    """Compute all 6 divergence metrics between contribution vectors.

    If either vector contains NaN (e.g. from failed SHAP computation on
    very long sequences), all divergences are returned as NaN.
    """
    if np.any(np.isnan(c_pred)) or np.any(np.isnan(c_expl)):
        return {name: float("nan") for name in DIVERGENCE_REGISTRY}
    return {name: fn(c_pred, c_expl) for name, fn in DIVERGENCE_REGISTRY.items()}
