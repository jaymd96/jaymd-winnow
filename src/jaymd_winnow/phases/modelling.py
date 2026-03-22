"""Phase 3: Model factory, conformal wrapping, and regime detection."""

import warnings

import numpy as np
import ruptures as rpt
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from jaymd_winnow.config import ModelConfig, RegimeConfig


# ---------------------------------------------------------------------------
# Model factory
# ---------------------------------------------------------------------------

def build_base_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    task: str,
    config: ModelConfig,
) -> Pipeline:
    """Fit and return a sklearn Pipeline based on model_type.

    The returned Pipeline always has ("scaler", StandardScaler()) as the first step.
    """
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train)

    if config.model_type == "elastic_net":
        model = _build_linear(X_scaled, y_train, task, config)
    elif config.model_type == "lightgbm":
        model = _build_lightgbm(X_scaled, y_train, task, config)
    elif config.model_type == "ensemble":
        model = _build_ensemble(X_scaled, y_train, task, config)
    else:
        raise ValueError(f"Unknown model_type: {config.model_type}")

    return Pipeline([("scaler", scaler), ("model", model)])


def _build_linear(X: np.ndarray, y: np.ndarray, task: str, config: ModelConfig):
    if task == "regression":
        from sklearn.linear_model import ElasticNetCV
        model = ElasticNetCV(l1_ratio=config.l1_ratios, cv=config.cv_folds)
    elif task == "binary":
        from sklearn.linear_model import LogisticRegressionCV
        model = LogisticRegressionCV(
            penalty="elasticnet", solver="saga",
            l1_ratios=config.l1_ratios, cv=config.cv_folds, max_iter=5000,
        )
    elif task == "multiclass":
        from sklearn.linear_model import LogisticRegressionCV
        model = LogisticRegressionCV(
            penalty="elasticnet", solver="saga",
            l1_ratios=config.l1_ratios, cv=config.cv_folds,
            multi_class="multinomial", max_iter=5000,
        )
    else:
        raise ValueError(f"Unknown task: {task}")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(X, y)
    return model


def _build_lightgbm(X: np.ndarray, y: np.ndarray, task: str, config: ModelConfig):
    try:
        import lightgbm as lgb
    except ImportError:
        raise ImportError(
            "lightgbm is required for model_type='lightgbm'. "
            "Install it with: pip install lightgbm>=4.0"
        )

    if task == "regression":
        model = lgb.LGBMRegressor(**config.lgbm_params)
    elif task == "binary":
        model = lgb.LGBMClassifier(**config.lgbm_params, objective="binary")
    elif task == "multiclass":
        model = lgb.LGBMClassifier(**config.lgbm_params, objective="multiclass")
    else:
        raise ValueError(f"Unknown task: {task}")

    model.fit(X, y)
    return model


def _build_ensemble(X: np.ndarray, y: np.ndarray, task: str, config: ModelConfig):
    from sklearn.ensemble import VotingClassifier, VotingRegressor

    linear = _build_linear(X, y, task, config)
    tree = _build_lightgbm(X, y, task, config)
    weights = config.ensemble_weights

    named_estimators = [("linear", linear), ("tree", tree)]

    if task == "regression":
        ensemble = VotingRegressor(estimators=named_estimators, weights=weights)
    else:
        ensemble = VotingClassifier(
            estimators=named_estimators, voting="soft", weights=weights,
        )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ensemble.fit(X, y)
    return ensemble


# ---------------------------------------------------------------------------
# Conformal wrapping
# ---------------------------------------------------------------------------

def conformalise_model(
    fitted_pipeline: Pipeline,
    X_conf: np.ndarray,
    y_conf: np.ndarray,
    task: str,
    config: ModelConfig,
) -> object:
    """Wrap a fitted model in a MAPIE conformal predictor."""
    if task == "regression":
        from mapie.regression import SplitConformalRegressor
        mapie = SplitConformalRegressor(
            estimator=fitted_pipeline,
            confidence_level=config.confidence_levels,
            prefit=True,
        )
        mapie.conformalize(X_conf, y_conf)
        return mapie
    elif task in ("binary", "multiclass"):
        from mapie.classification import SplitConformalClassifier
        mapie = SplitConformalClassifier(
            estimator=fitted_pipeline,
            confidence_level=config.confidence_levels,
            prefit=True,
        )
        mapie.conformalize(X_conf, y_conf)
        return mapie
    else:
        raise ValueError(f"Unknown task: {task}")


# ---------------------------------------------------------------------------
# Regime detection
# ---------------------------------------------------------------------------

def detect_regimes(
    y: np.ndarray,
    config: RegimeConfig,
) -> list[int]:
    """Detect regime breakpoints using ruptures.

    Returns breakpoint indices (excluding the terminal n).
    """
    algo_name = config.algorithm.capitalize()
    if algo_name == "Pelt":
        algo_name = "Pelt"
    elif algo_name == "Binseg":
        algo_name = "Binseg"
    elif algo_name == "Bottomup":
        algo_name = "BottomUp"

    algo_cls = getattr(rpt, algo_name)

    if config.custom_cost is not None:
        algo = algo_cls(custom_cost=config.custom_cost, min_size=config.min_segment_size)
    else:
        algo = algo_cls(model=config.cost_model, min_size=config.min_segment_size)

    algo.fit(y.reshape(-1, 1))
    breakpoints = algo.predict(pen=config.penalty)
    # ruptures always includes n as last element; strip it
    return breakpoints[:-1]
