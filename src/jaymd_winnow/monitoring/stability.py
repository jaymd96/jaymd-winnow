"""Feature importance stability monitoring via SHAP."""

import warnings

import numpy as np
import shap
from scipy.stats import spearmanr
from sklearn.pipeline import Pipeline


def compute_shap_importances(
    model: Pipeline,
    X: np.ndarray,
) -> np.ndarray:
    """Compute feature importances as mean(|SHAP values|).

    Args:
        model: The base sklearn Pipeline (not the MAPIE wrapper).
        X: Background data, typically recent training data.

    Returns:
        Feature importances, shape (n_features,).
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        explainer = shap.Explainer(model, X)
        shap_values = explainer(X)

    vals = shap_values.values
    # For multiclass: (n_samples, n_features, n_classes) → mean over classes
    if vals.ndim == 3:
        vals = vals.mean(axis=2)
    return np.abs(vals).mean(axis=0)


def compute_importance_rank_stability(
    importances_current: np.ndarray,
    importances_previous: np.ndarray,
) -> float:
    """Spearman rank correlation between two importance vectors.

    Returns correlation coefficient in [-1, 1]. High = stable.
    """
    corr, _ = spearmanr(importances_current, importances_previous)
    return float(corr)
