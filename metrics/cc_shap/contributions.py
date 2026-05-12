"""Contribution normalization for CC-SHAP.

Converts raw SHAP values into normalized contribution ratios (Eq. 3-4 of
the CC-SHAP paper) that indicate each input token's relative importance.
"""

from __future__ import annotations

import logging

import numpy as np
from numpy.typing import NDArray

logger = logging.getLogger(__name__)


def normalize_contributions(
    phi: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Normalize SHAP values to contribution ratios and aggregate.

    Per output token t, compute contribution ratio for each input token j:
        r_j^t = φ_j^t / Σ_i |φ_i^t|     (Eq. 3, range [-1, 1])

    Then average across output tokens:
        c_j = (1/T) Σ_t r_j^t             (Eq. 4)

    Args:
        phi: SHAP values. Shape (p,) for single-token or (p, T) for multi-token.

    Returns:
        Contribution vector c of shape (p,), values in [-1, 1].
    """
    if phi.ndim == 1:
        # Single output token
        abs_sum = np.abs(phi).sum()
        if abs_sum < 1e-12:
            logger.warning(
                "All SHAP values are near-zero (abs_sum=%.2e). "
                "This means no input token has measurable contribution. "
                "Returning zero contribution vector.",
                abs_sum,
            )
            return np.zeros_like(phi)
        return phi / abs_sum

    # Multi-token: normalize per output token, then average
    # phi shape: (p, T)
    abs_sums = np.abs(phi).sum(axis=0)  # (T,)
    zero_cols = (abs_sums < 1e-12).sum()
    if zero_cols > 0:
        logger.warning(
            "%d of %d output tokens have near-zero SHAP values "
            "(no measurable input contribution). These will contribute "
            "zero to the averaged contribution vector.",
            zero_cols,
            phi.shape[1],
        )
    # Avoid division by zero for tokens where all SHAP values are ~0
    safe_sums = np.where(abs_sums < 1e-12, 1.0, abs_sums)
    ratios = phi / safe_sums[None, :]  # (p, T), each column in [-1, 1]
    return ratios.mean(axis=1)  # (p,)
