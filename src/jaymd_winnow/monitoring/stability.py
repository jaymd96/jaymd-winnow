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
    # Use a subsample for background to keep SHAP fast
    max_bg = min(100, X.shape[0])
    background = X[:max_bg]

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        # Try specialized explainers first, fall back to model-agnostic
        inner_model = model.named_steps.get("model") if isinstance(model, Pipeline) else model

        try:
            # For tree-based models, use TreeExplainer (fast)
            explainer = shap.TreeExplainer(inner_model)
            X_for_shap = model.named_steps["scaler"].transform(X) if isinstance(model, Pipeline) else X
        except Exception:
            # Fall back to model-agnostic explainer using Pipeline.predict
            masker = shap.maskers.Independent(background)
            predict_fn = model.predict
            try:
                # Check if model supports predict_proba (classifiers)
                model.predict_proba
                predict_fn = model.predict_proba
            except AttributeError:
                pass
            explainer = shap.Explainer(predict_fn, masker)
            X_for_shap = X

        shap_values = explainer(X_for_shap)

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
