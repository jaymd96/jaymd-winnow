"""Phase 2: Stability selection to find features with reliable target association."""

import logging

import numpy as np

from jaymd_winnow.config import ScreeningConfig
from jaymd_winnow.stability_selection import stability_selection

logger = logging.getLogger(__name__)


def screen_features(
    X: np.ndarray,
    y: np.ndarray,
    task: str,
    config: ScreeningConfig,
) -> np.ndarray:
    """Select features with statistically reliable association via stability selection.

    Args:
        X: Feature matrix of cluster representatives, shape (n_samples, n_reduced_features).
        y: Target vector, shape (n_samples,).
        task: One of "regression", "binary", "multiclass".
        config: Screening configuration.

    Returns:
        selected_indices: Column indices into X of selected features, shape (n_selected,).
    """
    y = np.asarray(y).ravel()  # ensure 1D
    selected, scores = stability_selection(
        X, y,
        task=task,
        n_bootstraps=config.n_bootstraps,
        threshold=config.threshold,
    )

    if len(selected) == 0:
        logger.warning(
            "Stability selection returned zero features; "
            "falling back to top-10 by absolute correlation"
        )
        correlations = np.array([
            abs(np.corrcoef(X[:, j], y)[0, 1]) for j in range(X.shape[1])
        ])
        correlations = np.nan_to_num(correlations, nan=0.0)
        k = min(10, X.shape[1])
        selected = np.argsort(correlations)[-k:]

    return selected
