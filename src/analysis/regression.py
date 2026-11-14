"""Metrics for held-out regression predictions."""

import numpy as np
from numpy.typing import NDArray
from sklearn.metrics import r2_score


def held_out_r2(y_true: NDArray, y_pred: NDArray) -> float | None:
    """Compute R² or return None when the measured targets have no variation."""
    if len(y_true) < 2 or np.all(y_true == y_true[0]):
        return None
    score = r2_score(y_true, y_pred, force_finite=False)
    return float(score) if np.isfinite(score) else None
