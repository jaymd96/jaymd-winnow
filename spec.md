# `adaptive-pipeline` — Implementation Specification

**Version**: 0.1.0
**Target audience**: Experienced Python developer familiar with numpy/sklearn/unix tooling
**Estimated implementation effort**: 5–7 days for core, 2–3 days for tests

---

## 1. Purpose

An adaptive model lifecycle package for financial signal selection and prediction. The package composes existing libraries to solve one problem: given a large collection of potentially non-stationary signals and one or more targets, build models that automatically select relevant features, make calibrated probabilistic predictions, monitor their own health, and trigger retraining when their assumptions break.

The design principle is **composition over implementation**. We write orchestration code only. Every algorithm is delegated to a mature, maintained package.

The package handles regression, binary classification, and multi-class classification. It supports multiple simultaneous targets (shared feature selection, independent models). It supports both linear (ElasticNet/Logistic) and nonlinear (LightGBM) model types, and a simple ensemble of both.

---

## 2. Dependencies

### Required

| Package | Version | What it does for us |
|---|---|---|
| `numpy` | >=1.24 | Array operations |
| `scipy` | >=1.11 | Hierarchical clustering, KS tests, Spearman correlation |
| `scikit-learn` | >=1.5 | ElasticNetCV, LogisticRegressionCV, StandardScaler |
| `joblib` | >=1.3 | `Memory` — content-addressable caching of function outputs |
| `ruptures` | >=1.1 | Offline change-point detection |
| `stability-selection` | >=0.1 | Bootstrap stability selection, sklearn-compatible |
| `mapie` | >=1.0 | Conformal prediction intervals and sets (use v1 API) |
| `shap` | >=0.43 | Model-agnostic feature attribution |
| `lightgbm` | >=4.0 | Gradient boosted trees — nonlinear model option |

No optional dependencies. Everything required. One install, everything works.

---

## 3. Package Structure

```
adaptive_pipeline/
├── __init__.py              # Public API: AdaptivePipeline, PipelineTrace
├── pipeline.py              # AdaptivePipeline class — the step() state machine
├── config.py                # Dataclasses for all configuration
├── phases/
│   ├── __init__.py
│   ├── clustering.py        # Phase 1: feature clustering
│   ├── screening.py         # Phase 2: stability selection
│   └── modelling.py         # Phase 3: model factory + conformal wrapping
├── monitoring/
│   ├── __init__.py
│   ├── calibration.py       # PIT, Brier reliability, ECE
│   ├── stability.py         # Feature importance rank stability
│   └── triggers.py          # Retraining trigger logic
├── trace.py                 # PipelineTrace — query and analysis of results
├── types.py                 # StepResult, HealthSnapshot, LifecycleEvent dataclasses
└── cache.py                 # Thin wrapper around joblib.Memory for phase-keyed caching
```

No sub-packages beyond what is listed. No `utils.py`. If a function doesn't belong to a specific module, the design is wrong.

---

## 4. Configuration

All configuration is via frozen dataclasses. No mutable config after construction.

```python
# config.py

from dataclasses import dataclass, field
from typing import Literal, Optional, List, Any

@dataclass(frozen=True)
class ClusteringConfig:
    max_clusters: int = 200
    method: str = "ward"                    # scipy linkage method
    distance: str = "correlation"           # "correlation" or "euclidean"
    update_frequency: int = 63              # re-cluster every N steps (~quarterly at daily)

@dataclass(frozen=True)
class ScreeningConfig:
    threshold: float = 0.6                  # stability selection threshold
    n_bootstraps: int = 200
    base_model: str = "elastic_net"         # or "lasso", "logistic"
    update_frequency: int = 63              # re-screen every N steps

@dataclass(frozen=True)
class ModelConfig:
    model_type: Literal["elastic_net", "lightgbm", "ensemble"] = "elastic_net"
    # elastic_net / logistic params
    l1_ratios: List[float] = field(default_factory=lambda: [0.5, 0.7, 0.9, 0.95])
    # lightgbm params
    lgbm_params: dict = field(default_factory=lambda: {
        "n_estimators": 200,
        "max_depth": 5,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
        "verbose": -1,
    })
    # ensemble params (only used when model_type == "ensemble")
    ensemble_weights: Optional[List[float]] = None  # None = equal weighting
    # common params
    cv_folds: int = 5
    confidence_levels: List[float] = field(default_factory=lambda: [0.68, 0.95])
    conformal_method: str = "split"         # "split" or "cross"
    train_fraction: float = 0.6             # of each regime window
    conformalize_fraction: float = 0.2

@dataclass(frozen=True)
class RegimeConfig:
    algorithm: str = "pelt"                 # "pelt", "binseg", "bottomup"
    cost_model: str = "rbf"                 # "l1", "l2", "rbf", "linear", "normal", "ar"
    custom_cost: Any = None                 # a ruptures.BaseCost instance, overrides cost_model
    penalty: float = 10.0                   # penalty for Pelt/Binseg
    min_segment_size: int = 60              # minimum regime length

@dataclass(frozen=True)
class MonitorConfig:
    pit_window: int = 60                    # rolling window for PIT test
    stability_window: int = 20              # window for SHAP rank correlation
    shap_frequency: int = 20               # compute SHAP every N steps
    calibration_alert_threshold: float = 0.05  # KS p-value below this = alert

@dataclass(frozen=True)
class RetrainingConfig:
    calibration_trigger: float = 0.05       # PIT p-value threshold
    stability_trigger: float = 0.3          # rank correlation below this = structural break
    cooldown_steps: int = 20                # minimum steps between retrains
    min_regime_size: int = 60               # minimum data points to retrain on

@dataclass(frozen=True)
class TargetConfig:
    """Configuration for a single target."""
    name: str                               # human-readable name
    task: Literal["regression", "binary", "multiclass"] = "regression"
    model: ModelConfig = field(default_factory=ModelConfig)

@dataclass(frozen=True)
class PipelineConfig:
    targets: List[TargetConfig] = field(default_factory=lambda: [
        TargetConfig(name="target", task="regression")
    ])
    min_history: int = 252                  # warmup period before first prediction
    clustering: ClusteringConfig = field(default_factory=ClusteringConfig)
    screening: ScreeningConfig = field(default_factory=ScreeningConfig)
    regime: RegimeConfig = field(default_factory=RegimeConfig)
    monitor: MonitorConfig = field(default_factory=MonitorConfig)
    retraining: RetrainingConfig = field(default_factory=RetrainingConfig)
    cache_dir: Optional[str] = None         # None = no caching
```

**Key design decision — multi-target**: Phase 1 (clustering) and Phase 2 (screening) are shared across all targets. Phase 3 (model fitting) is per-target — each target gets its own model, its own conformal wrapper, and its own monitoring state. This means feature screening uses the FIRST target for selection (or, if targets have different tasks, the regression target). If this proves insufficient, a future version could union the selected features across targets. The per-target model config lives in `TargetConfig`, so you can use ElasticNet for one target and LightGBM for another.

**Key design decision — regime detection config**: `RegimeConfig` is separated from `RetrainingConfig` because it controls the ruptures algorithm (which is also used during training window construction), not just the retraining trigger. The `custom_cost` field accepts any `ruptures.BaseCost` instance, giving full extensibility. When `custom_cost` is not None, `cost_model` is ignored.

---

## 5. Core Types

```python
# types.py

from dataclasses import dataclass, field
from typing import Optional, Any, Dict, List
import numpy as np

@dataclass
class TargetHealth:
    """Health metrics for a single target."""
    target_name: str
    calibration_pvalue: Optional[float] = None       # KS test p-value (regression)
    brier_reliability: Optional[float] = None        # Brier decomposition (classification)
    ece: Optional[float] = None                      # Expected calibration error (multiclass)
    feature_stability: Optional[float] = None        # Spearman rank corr of importances
    coverage: Optional[Dict[float, float]] = None    # {confidence_level: empirical_coverage}

@dataclass
class HealthSnapshot:
    """Aggregate health across all targets."""
    per_target: Dict[str, TargetHealth] = field(default_factory=dict)

    @property
    def worst_calibration(self) -> Optional[float]:
        """Lowest calibration p-value across all targets. None if no data yet."""
        pvals = [t.calibration_pvalue for t in self.per_target.values()
                 if t.calibration_pvalue is not None]
        return min(pvals) if pvals else None

    @property
    def worst_stability(self) -> Optional[float]:
        """Lowest feature stability across all targets."""
        stabs = [t.feature_stability for t in self.per_target.values()
                 if t.feature_stability is not None]
        return min(stabs) if stabs else None

@dataclass
class LifecycleEvent:
    event_type: str                                  # "retrain_refit", "retrain_reselect",
                                                     # "retrain_rebuild", "alert", "warmup_complete"
    timestamp: Any                                   # user-provided timestamp
    reason: str                                      # human-readable explanation
    details: Dict[str, Any] = field(default_factory=dict)

@dataclass
class TargetPrediction:
    """Prediction for a single target."""
    target_name: str
    point: Optional[np.ndarray] = None               # point prediction
    intervals: Optional[np.ndarray] = None           # shape (2, n_confidence_levels): regression
    prediction_sets: Optional[np.ndarray] = None     # shape (n_classes, n_confidence_levels): classification

@dataclass
class StepResult:
    timestamp: Any
    predictions: Dict[str, TargetPrediction] = field(default_factory=dict)
    selected_features: Optional[np.ndarray] = None   # indices of currently selected features
    health: Optional[HealthSnapshot] = None
    event: Optional[LifecycleEvent] = None
    is_warmup: bool = True
```

---

## 6. Phase Implementations

### 6.1 Phase 1 — Feature Clustering (`phases/clustering.py`)

**Purpose**: Reduce 500–10k correlated features down to ~200 independent cluster representatives.

**Algorithm**:
1. Compute pairwise absolute correlation matrix: `np.corrcoef(X.T)`
2. Convert to distance: `1 - |corr|`
3. Hierarchical clustering: `scipy.cluster.hierarchy.linkage(squareform(dist), method=config.method)`
4. Cut tree: `fcluster(Z, t=config.max_clusters, criterion='maxclust')`
5. Select representative per cluster: feature with highest variance in that cluster.

**Function signature**:

```python
def cluster_features(
    X: np.ndarray,                      # (n_samples, n_features)
    config: ClusteringConfig
) -> tuple[np.ndarray, np.ndarray]:
    """
    Returns:
        cluster_labels: shape (n_features,) — cluster assignment per feature
        representative_indices: shape (n_clusters,) — column indices of representatives
    """
```

**Caching**: This function is a pure function of `(X, config)`. Cache it.

**Edge cases**:
- If `n_features < max_clusters`, skip clustering, return all features as representatives.
- If any feature has zero variance, drop it before clustering (causes NaN correlations).

### 6.2 Phase 2 — Feature Screening (`phases/screening.py`)

**Purpose**: From ~200 cluster representatives, find features with statistically reliable association with target.

**Algorithm**: Stability selection via the `stability-selection` package.

**Function signature**:

```python
def screen_features(
    X: np.ndarray,                      # (n_samples, n_reduced_features)
    y: np.ndarray,                      # (n_samples,)
    task: str,                          # "regression", "binary", "multiclass"
    config: ScreeningConfig
) -> np.ndarray:
    """
    Returns:
        selected_indices: shape (n_selected,) — column indices into X of selected features
    """
```

**Implementation detail — base estimator selection**:

```python
from stability_selection import StabilitySelection
from sklearn.linear_model import LassoCV, LogisticRegressionCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

if task == "regression":
    base = Pipeline([
        ("scaler", StandardScaler()),
        ("model", LassoCV(cv=config.cv_folds))
    ])
    lambda_name = "model__alphas"
elif task in ("binary", "multiclass"):
    base = Pipeline([
        ("scaler", StandardScaler()),
        ("model", LogisticRegressionCV(
            penalty="l1", solver="saga",
            cv=config.cv_folds, max_iter=5000
        ))
    ])
    lambda_name = "model__Cs"

selector = StabilitySelection(
    base_estimator=base,
    lambda_name=lambda_name,
    threshold=config.threshold,
    n_bootstrap_iterations=config.n_bootstraps,
    n_jobs=-1,
)
selector.fit(X, y)
return np.where(selector.get_support())[0]
```

**Important**: the `stability-selection` package requires that `lambda_name` matches a parameter on the estimator. Since we use a Pipeline, the param name is `"model__Cs"` or `"model__alphas"` (pipeline-prefixed). The `lambda_grid` parameter of `StabilitySelection` provides an explicit grid to sweep. Use `np.logspace(-5, -1, 25)` for `Cs` and `np.logspace(-3, 1, 25)` for `alphas`.

**Multi-target screening**: Phase 2 screens using the FIRST target in `config.targets`. The rationale is that features relevant to one financial target (e.g., returns) are likely relevant to related targets (e.g., volatility, direction). If the targets are genuinely unrelated, the implementer should consider running screening per-target and taking the union of selected features. But for v0.1, single-target screening with shared features is the default.

**Caching**: Pure function. Cache it. This is the most expensive phase (~minutes for 200 features × 200 bootstraps).

**Edge case**: If stability selection returns zero features, fall back to the top-10 by univariate correlation with the target. Log a warning.

### 6.3 Phase 3 — Model Fitting + Conformal Wrapping (`phases/modelling.py`)

**Purpose**: For each target, fit a model on the screened features, wrap it in MAPIE for calibrated intervals/sets.

This module contains a **model factory** that produces fitted sklearn-compatible estimators based on `model_type`, and a **conformalisation wrapper** that adds MAPIE prediction intervals.

#### 6.3.1 Model Factory

```python
def build_base_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    task: str,
    config: ModelConfig,
) -> Pipeline:
    """
    Fit and return a sklearn Pipeline based on model_type.
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
```

**Linear models** (`_build_linear`):

```python
def _build_linear(X, y, task, config):
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
    model.fit(X, y)
    return model
```

**LightGBM models** (`_build_lightgbm`):

```python
def _build_lightgbm(X, y, task, config):
    import lightgbm as lgb

    if task == "regression":
        model = lgb.LGBMRegressor(**config.lgbm_params)
    elif task == "binary":
        model = lgb.LGBMClassifier(**config.lgbm_params, objective="binary")
    elif task == "multiclass":
        model = lgb.LGBMClassifier(**config.lgbm_params, objective="multiclass")

    model.fit(X, y)
    return model
```

**Ensemble models** (`_build_ensemble`):

The ensemble fits BOTH a linear model and a LightGBM model, then wraps them in sklearn's built-in `VotingRegressor` or `VotingClassifier`. No custom classes needed.

```python
from sklearn.ensemble import VotingRegressor, VotingClassifier

def _build_ensemble(X, y, task, config):
    linear = _build_linear(X, y, task, config)
    tree = _build_lightgbm(X, y, task, config)
    weights = config.ensemble_weights  # None = equal

    named_estimators = [("linear", linear), ("tree", tree)]

    if task == "regression":
        ensemble = VotingRegressor(estimators=named_estimators, weights=weights)
    else:
        ensemble = VotingClassifier(
            estimators=named_estimators, voting="soft", weights=weights,
        )

    # VotingRegressor/VotingClassifier need .fit() called even with pre-fitted
    # sub-estimators, because it sets internal attributes (e.g., .estimators_,
    # .le_ for classifiers). The sub-estimators are already fitted, so this is
    # cheap — it just calls .fit() on each, which is a near-no-op for already-
    # converged models since sklearn checks convergence.
    ensemble.fit(X, y)
    return ensemble
```

`VotingRegressor` and `VotingClassifier` are sklearn-native, fully picklable, provide `predict` and `predict_proba` (with `voting="soft"`), and work correctly with both MAPIE's `prefit=True` and SHAP.

**SHAP compatibility**: For the ensemble, SHAP will use its default `Permutation` explainer, which is model-agnostic but slower. For pure linear or pure LightGBM models, SHAP auto-selects `LinearExplainer` or `TreeExplainer`, which are much faster. This is a deliberate trade-off: ensembles give better predictions but slower SHAP computation.

#### 6.3.2 Conformalisation

Unchanged from previous spec. The MAPIE wrapper is cheap and not cached:

```python
from mapie.regression import SplitConformalRegressor
from mapie.classification import SplitConformalClassifier

def conformalise_model(
    fitted_pipeline: Pipeline,
    X_conf: np.ndarray,
    y_conf: np.ndarray,
    task: str,
    config: ModelConfig,
) -> object:
    """
    Returns a MAPIE conformal wrapper around the fitted model.
    """
    if task == "regression":
        mapie = SplitConformalRegressor(
            estimator=fitted_pipeline,
            confidence_level=config.confidence_levels,
            prefit=True,
        )
        mapie.conformalize(X_conf, y_conf)
        return mapie
    elif task in ("binary", "multiclass"):
        mapie = SplitConformalClassifier(
            estimator=fitted_pipeline,
            confidence_level=config.confidence_levels,
            prefit=True,
        )
        mapie.conformalize(X_conf, y_conf)
        return mapie
```

#### 6.3.3 Multi-target orchestration

The pipeline calls Phase 3 in a loop over targets:

```python
# Inside pipeline retrain logic:
for target_cfg in self._config.targets:
    y_target = y_window[:, target_idx]   # extract this target's column
    base = self._cached_build_base_model(
        X_window_screened_train, y_target_train,
        target_cfg.task, target_cfg.model,
    )
    conformal = conformalise_model(
        base, X_window_screened_conf, y_target_conf,
        target_cfg.task, target_cfg.model,
    )
    self._state.models[target_cfg.name] = conformal
```

**Caching per target**: `build_base_model` is cached per `(X_train, y_train, task, config)`. Since each target has different `y_train`, different targets always cache-miss against each other. But re-running the same target with the same data and config is a cache hit.

**Data splitting within a training window** (unchanged):

When the pipeline decides to train, the regime window is split TEMPORALLY:

```
|--- train_fraction ---|--- conformalize_fraction ---|--- remainder ---|
```

#### 6.3.4 Regime window detection

Regime boundaries are detected using ruptures. The `RegimeConfig` controls this:

```python
import ruptures as rpt

def detect_regimes(
    y: np.ndarray,                      # target series
    config: RegimeConfig,
) -> list[int]:
    """
    Returns breakpoint indices (excluding the terminal n).
    """
    if config.custom_cost is not None:
        algo_cls = getattr(rpt, config.algorithm.capitalize())
        algo = algo_cls(custom_cost=config.custom_cost, min_size=config.min_segment_size)
    else:
        algo_cls = getattr(rpt, config.algorithm.capitalize())
        algo = algo_cls(model=config.cost_model, min_size=config.min_segment_size)

    algo.fit(y.reshape(-1, 1))
    breakpoints = algo.predict(pen=config.penalty)
    # ruptures always includes n as the last element; strip it
    return breakpoints[:-1]
```

**MAPIE v1 API notes** (unchanged):

- `SplitConformalRegressor.predict_interval(X)` returns `(y_pred, y_pis)` where `y_pis` has shape `(n_samples, 2, n_confidence_levels)`. `y_pis[:, 0, :]` = lower bounds, `y_pis[:, 1, :]` = upper bounds.
- `SplitConformalClassifier.predict_set(X)` returns `(y_pred, y_sets)` where `y_sets` is a boolean array of shape `(n_samples, n_classes, n_confidence_levels)`.
- The conformity scores are stored in `mapie._mapie_regressor.conformity_scores_` (for regression). This is needed for PIT monitoring.
- `prefit=True` means the estimator is already fitted; MAPIE will NOT re-fit it during `.conformalize()`.

---

## 7. Monitoring (`monitoring/`)

### 7.1 Calibration Monitoring (`monitoring/calibration.py`)

**For regression — PIT uniformity**:

```python
def compute_pit_value(
    y_actual: float,
    conformity_scores: np.ndarray,    # from MAPIE calibration set
    y_pred: float,
) -> float:
    """
    Compute the Probability Integral Transform value for one observation.
    PIT = fraction of calibration residuals <= the observed residual.
    Under correct model, PIT ~ Uniform(0,1).
    """
    observed_residual = np.abs(y_actual - y_pred)
    return np.mean(conformity_scores <= observed_residual)


def test_pit_uniformity(
    pit_values: np.ndarray,           # rolling window of PIT values
) -> tuple[float, float]:
    """
    Returns (ks_statistic, p_value).
    Low p_value => PIT values are NOT uniform => model is miscalibrated.
    """
    from scipy.stats import kstest
    return kstest(pit_values, 'uniform')
```

**For binary classification — Brier reliability**:

```python
def compute_brier_reliability(
    y_true: np.ndarray,               # binary 0/1
    y_prob: np.ndarray,               # predicted probabilities for class 1
    n_bins: int = 10,
) -> float:
    """
    Reliability component of the Brier score decomposition.
    Low reliability = well-calibrated. High = miscalibrated.
    """
    bin_edges = np.linspace(0, 1, n_bins + 1)
    reliability = 0.0
    n = len(y_true)
    for i in range(n_bins):
        mask = (y_prob >= bin_edges[i]) & (y_prob < bin_edges[i + 1])
        if mask.sum() == 0:
            continue
        bin_mean_pred = y_prob[mask].mean()
        bin_mean_true = y_true[mask].mean()
        reliability += mask.sum() * (bin_mean_pred - bin_mean_true) ** 2
    return reliability / n
```

**For multiclass — Expected Calibration Error (ECE)**:

```python
def compute_ece(
    y_true: np.ndarray,
    y_prob: np.ndarray,               # (n_samples, n_classes) probability matrix
    n_bins: int = 10,
) -> float:
    """
    Expected Calibration Error. Weighted average of per-bin |accuracy - confidence|.
    """
    confidences = y_prob.max(axis=1)
    predictions = y_prob.argmax(axis=1)
    accuracies = (predictions == y_true).astype(float)

    bin_edges = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    n = len(y_true)
    for i in range(n_bins):
        mask = (confidences >= bin_edges[i]) & (confidences < bin_edges[i + 1])
        if mask.sum() == 0:
            continue
        bin_accuracy = accuracies[mask].mean()
        bin_confidence = confidences[mask].mean()
        ece += mask.sum() * np.abs(bin_accuracy - bin_confidence)
    return ece / n
```

### 7.2 Feature Importance Stability (`monitoring/stability.py`)

```python
import shap
from scipy.stats import spearmanr

def compute_shap_importances(
    model: Pipeline,                   # the base sklearn Pipeline (not the MAPIE wrapper)
    X: np.ndarray,                    # background data, typically recent training data
) -> np.ndarray:
    """
    Returns feature importances as mean(|SHAP values|), shape (n_features,).
    """
    explainer = shap.Explainer(model, X)
    shap_values = explainer(X)
    # shap_values.values has shape (n_samples, n_features) for regression
    # For multiclass it's (n_samples, n_features, n_classes) — take mean over classes
    vals = shap_values.values
    if vals.ndim == 3:
        vals = vals.mean(axis=2)
    return np.abs(vals).mean(axis=0)


def compute_importance_rank_stability(
    importances_current: np.ndarray,
    importances_previous: np.ndarray,
) -> float:
    """
    Spearman rank correlation between two importance vectors.
    Returns correlation coefficient in [-1, 1]. High = stable.
    """
    corr, _ = spearmanr(importances_current, importances_previous)
    return corr
```

**Caching**: `compute_shap_importances` is expensive. Cache it keyed on `(model_hash, X_hash)`. Only compute every `config.monitor.shap_frequency` steps. Note: for ensembles, SHAP uses the slower `PermutationExplainer`. For `shap_frequency`, consider using a larger value (e.g., 40 instead of 20) when `model_type == "ensemble"` to compensate.

**Multi-target**: Compute SHAP importances per target model. Feature stability is tracked per target in `TargetHealth`. The `HealthSnapshot.worst_stability` property gives the worst-case across targets for triggering.

### 7.3 Retraining Triggers (`monitoring/triggers.py`)

```python
def evaluate_trigger(
    health: HealthSnapshot,
    config: RetrainingConfig,
    steps_since_last_retrain: int,
) -> Optional[str]:
    """
    Returns:
        None — no action needed
        "refit" — rerun Phase 3 only (coefficients drifted)
        "reselect" — rerun Phase 2 + 3 (feature-target structure changed)
        "rebuild" — rerun Phase 1 + 2 + 3 (correlation structure changed)
    """
    # Respect cooldown
    if steps_since_last_retrain < config.cooldown_steps:
        return None

    # Use worst-case across all targets
    worst_cal = health.worst_calibration
    worst_stab = health.worst_stability

    # Check calibration
    if worst_cal is not None and worst_cal > config.calibration_trigger:
        return None  # all targets calibrated

    if worst_cal is None:
        return None  # not enough data yet

    # Calibration failed for at least one target — diagnose
    if worst_stab is not None and worst_stab < config.stability_trigger:
        return "reselect"

    return "refit"
```

Note: "rebuild" (Phase 1 re-clustering) is triggered by the pipeline when the cluster update frequency is reached, not by the trigger logic. It's periodic, not reactive.

---

## 8. The Pipeline State Machine (`pipeline.py`)

### 8.1 Internal State

```python
class _PipelineState:
    # Data buffers (grow over time)
    timestamps: list
    features_buffer: list[np.ndarray]       # list of 1D arrays, each shape (n_features,)
    targets_buffer: list[np.ndarray]        # list of 1D arrays, each shape (n_targets,)
    pending_predictions: Dict[str, Any]     # target_name -> last prediction (for monitoring)

    # Phase 1 + 2 outputs (shared across targets)
    cluster_labels: Optional[np.ndarray] = None
    representative_indices: Optional[np.ndarray] = None
    selected_feature_indices: Optional[np.ndarray] = None
    active_feature_indices: Optional[np.ndarray] = None  # into original feature space

    # Phase 3 outputs (per target)
    models: Dict[str, object] = {}          # target_name -> MAPIE conformal model
    base_models: Dict[str, Pipeline] = {}   # target_name -> sklearn Pipeline (for SHAP)
    conformity_scores: Dict[str, np.ndarray] = {}  # target_name -> calibration residuals

    # Monitoring state (per target)
    pit_values: Dict[str, list] = {}
    brier_values: Dict[str, list] = {}
    shap_importances_previous: Dict[str, np.ndarray] = {}
    shap_importances_current: Dict[str, np.ndarray] = {}

    # Lifecycle
    step_count: int = 0
    steps_since_last_retrain: int = 0
    steps_since_last_cluster: int = 0
    steps_since_last_screen: int = 0
    steps_since_last_shap: int = 0
    is_warm: bool = False
```

### 8.2 The `step()` method — main loop

This is the heart of the package. Every call to `step()` executes the following logic, in this exact order:

```
step(timestamp, features, targets_for_previous):
│
├─ 1. INGEST: buffer the new features, and the targets for the prior prediction
│
├─ 2. UPDATE MONITORS (per target, if we have a live model and received targets):
│   ├─ Compute PIT / Brier / ECE for each target's realised value
│   ├─ Run rolling calibration test per target
│   ├─ Compute SHAP importances per target (if shap_frequency reached)
│   ├─ Compute feature stability per target (if we have prev + current importances)
│   └─ Assemble HealthSnapshot with per-target TargetHealth
│
├─ 3. EVALUATE TRIGGERS:
│   ├─ If still in warmup and min_history reached → trigger initial training
│   ├─ If cluster update_frequency reached → trigger rebuild
│   ├─ If screen update_frequency reached → trigger reselect
│   └─ If monitor says recalibrate (worst-case across targets) → trigger retrain
│
├─ 4. RETRAIN (if triggered):
│   ├─ "rebuild": run Phase 1 → Phase 2 → Phase 3 (all targets)
│   ├─ "reselect": run Phase 2 → Phase 3 (all targets)
│   ├─ "refit": run Phase 3 only (all targets)
│   └─ Record LifecycleEvent
│
├─ 5. PREDICT (per target, if model is live):
│   ├─ Select features using active_feature_indices
│   ├─ Call conformal_model.predict_interval() or predict_set()
│   └─ Store prediction for target comparison on next step
│
└─ 6. RETURN StepResult with Dict[str, TargetPrediction]
```

### 8.3 Public API

```python
class AdaptivePipeline:

    def __init__(self, config: PipelineConfig):
        """Construct pipeline from config."""

    @classmethod
    def regression(cls, min_history=252, cache_dir=None, **overrides) -> "AdaptivePipeline":
        """Convenience constructor for single-target regression with elastic_net."""

    @classmethod
    def classification(cls, min_history=500, cache_dir=None, **overrides) -> "AdaptivePipeline":
        """Convenience constructor for single-target binary classification."""

    @classmethod
    def multi_target(cls, targets: list[dict], min_history=252, cache_dir=None,
                     **overrides) -> "AdaptivePipeline":
        """
        Convenience constructor for multi-target.
        targets: list of dicts like [{"name": "returns", "task": "regression"},
                                     {"name": "direction", "task": "binary", "model_type": "lightgbm"}]
        """

    def step(
        self,
        timestamp: Any,
        features: np.ndarray,              # shape (n_features,) — one observation
        targets: Optional[np.ndarray] = None  # shape (n_targets,) — realised targets for PREVIOUS prediction
    ) -> StepResult:
        """
        Advance the pipeline by one time step.

        Args:
            timestamp: User-provided timestamp (opaque — pipeline does not interpret it).
            features: Feature vector for this time step.
            targets: Realised target values for the PREVIOUS prediction, one per target
                     in the same order as config.targets. None if not yet available.
                     For single-target pipelines, a scalar is also accepted.

        Returns:
            StepResult with per-target predictions, intervals, health diagnostics, and events.
        """

    def predict(self, timestamp: Any, features: np.ndarray) -> StepResult:
        """
        Make a prediction without ingesting targets. For split predict/observe workflows.
        """

    def observe(self, timestamp: Any, targets: np.ndarray) -> StepResult:
        """
        Ingest realised targets without making a new prediction.
        """

    def save(self, path: str) -> None:
        """
        Serialise entire pipeline state to disk.
        Uses joblib.dump for numpy-optimised serialisation.
        """

    @classmethod
    def load(cls, path: str) -> "AdaptivePipeline":
        """
        Deserialise a pipeline. Calling step() after load() MUST produce identical
        results to having run all prior steps inline.
        """

    def set_cache_dir(self, cache_dir: str) -> None:
        """Re-attach a cache directory after loading. Not required for correctness."""

    @property
    def config(self) -> PipelineConfig:
        """Immutable config."""

    @property
    def is_warm(self) -> bool:
        """Whether the pipeline has completed warmup and has a live model."""
```

### 8.4 Implementation notes

**Target alignment**: `targets` is always for a PREVIOUS prediction. The pipeline tracks which prediction the incoming targets correspond to. For single-target convenience, `step()` should accept a scalar float and wrap it into a 1-element array internally.

**Feature subsetting**: `active_feature_indices` is the composition of cluster representatives and stability selection. When predicting, the pipeline selects `features[active_feature_indices]` before passing to models. All per-target models share the same feature subset.

**Training window construction**: When retraining is triggered, regime detection uses the FIRST target's series:

```python
breakpoints = detect_regimes(y_first_target, config.regime)
last_break = breakpoints[-1] if breakpoints else 0
# Training window: data[last_break:]
```

If the last regime is shorter than `min_regime_size`, extend backward. For Phase 1 and 2, use the full history (these benefit from more data).

---

## 9. Caching (`cache.py`)

```python
from joblib import Memory

class PipelineCache:
    def __init__(self, cache_dir: Optional[str]):
        if cache_dir is not None:
            self._memory = Memory(location=cache_dir, verbose=0)
        else:
            self._memory = Memory(location=None, verbose=0)

    def cache(self, func):
        return self._memory.cache(func)
```

The expensive functions are cached:

```python
# In pipeline.__init__:
self._cache = PipelineCache(config.cache_dir)
self._cluster_features = self._cache.cache(cluster_features)
self._screen_features = self._cache.cache(screen_features)
self._build_base_model = self._cache.cache(build_base_model)
# Note: conformalise_model is NOT cached (cheap, depends on varying calibration data)
# Note: compute_shap_importances IS cached (expensive)
self._compute_shap = self._cache.cache(compute_shap_importances)
```

---

## 10. PipelineTrace (`trace.py`)

```python
class PipelineTrace:
    """Analysis and investigation of pipeline results."""

    def __init__(self, results: list[StepResult]):
        self._results = results

    # --- Performance ---

    def predictions(self, target: str = None) -> tuple[list, np.ndarray, np.ndarray]:
        """Returns (timestamps, y_pred, y_actual). If target is None and there's
        only one target, use that. Otherwise target name is required."""

    def performance(self, target: str = None, metric: str = "auto",
                    after=None, before=None) -> dict:
        """Compute performance metrics over a time range."""

    # --- Calibration ---

    def calibration_over_time(self, target: str = None) -> tuple[list, np.ndarray]:
        """Returns (timestamps, p_values)."""

    def coverage_over_time(self, target: str = None) -> tuple[list, dict]:
        """Rolling empirical coverage for each confidence level."""

    # --- Feature dynamics ---

    def feature_importance_over_time(self, target: str = None) -> tuple[list, np.ndarray]:
        """Returns (timestamps, importances_matrix) shape (n_timestamps, n_features)."""

    def feature_stability_over_time(self, target: str = None) -> tuple[list, np.ndarray]:
        """Returns (timestamps, rank_correlations)."""

    def feature_set_changes(self) -> list[dict]:
        """List of {timestamp, features_added, features_removed, trigger}."""

    # --- Lifecycle ---

    def events(self, event_type: Optional[str] = None) -> list[LifecycleEvent]:
        """All lifecycle events, optionally filtered by type."""

    # --- Comparison ---

    @staticmethod
    def compare(a: "PipelineTrace", b: "PipelineTrace") -> dict:
        """Compare two traces from different configs on same data."""
```

All methods that return data return plain numpy arrays and lists. No pandas dependency.

---

## 11. Serialisation

The critical invariant:

> Run a backtest to step N, `save()`, `load()` in new process, `step()` with step N+1 data → identical result to running all N+1 steps inline.

```python
def save(self, path: str):
    import joblib
    joblib.dump({"config": self._config, "state": self._state, "version": __version__}, path)

@classmethod
def load(cls, path: str):
    import joblib
    data = joblib.load(path)
    pipeline = cls(data["config"])
    pipeline._state = data["state"]
    return pipeline
```

The cache is NOT saved. It's a performance optimisation, not state.

---

## 12. Error Handling

- Feature column all NaN or constant: **drop and warn**. Store in trace.
- Stability selection returns zero features: **fall back** to top-10 by absolute correlation. Log `LifecycleEvent`.
- MAPIE conformalisation fails (too few calibration points): **raise `ValueError`**.
- Ruptures finds zero breakpoints: **use entire history** as single regime. This is correct.
- Feature dimensionality changes between steps: **raise `ValueError`** immediately.
- LightGBM not installed but `model_type="lightgbm"`: **raise `ImportError`** with install instructions.

---

## 13. Testing Strategy

### Unit tests:

| Module | Key tests |
|---|---|
| `phases/clustering.py` | Correct cluster count; zero-variance handling; determinism |
| `phases/screening.py` | Recovers known features in synthetic data; zero-selection fallback |
| `phases/modelling.py` | Factory produces correct model for each `model_type` × `task` combination (9 combos). VotingRegressor/VotingClassifier correctly wrap sub-models. MAPIE wrapper produces correct interval shapes. |
| `monitoring/calibration.py` | PIT uniform for correct model; non-uniform for misspecified |
| `monitoring/triggers.py` | Cooldown respected; trigger levels correct; worst-case aggregation across targets |
| `cache.py` | Cache hit/miss; no-op when `cache_dir=None` |

### Integration tests:

| Test | Validates |
|---|---|
| **Regression backtest** (elastic_net) | Synthetic `pw_constant` data. Correct features selected, correct coverage, retraining near regime boundaries. |
| **Regression backtest** (lightgbm) | Same data, `model_type="lightgbm"`. Verify coverage and SHAP TreeExplainer usage. |
| **Regression backtest** (ensemble) | Same data, `model_type="ensemble"`. Verify VotingRegressor wraps both sub-models, averaged predictions, coverage. |
| **Classification backtest** | Binary synthetic data, logistic model. Prediction sets and Brier reliability. |
| **Multi-target backtest** | Two targets (one regression, one binary) on same features. Shared clustering/screening, independent models/monitoring. |
| **Custom ruptures cost** | Pass a `ruptures.BaseCost` via `RegimeConfig.custom_cost`. Verify regime detection uses it. |
| **Save/load invariant** | Run to step 500, save, load, continue. Assert identical results. |
| **Cache hit test** | Run twice with same config/data. Second run >10x faster. |

### Synthetic data generator:

```python
def make_regime_switching_data(
    n_samples=1000,
    n_features=100,
    n_relevant=5,
    n_regimes=3,
    n_targets=1,
    target_tasks=("regression",),
    noise_std=1.0,
    seed=42,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """
    Generate data where:
    - Only n_relevant features are truly predictive
    - The relevant features CHANGE between regimes
    - The coefficients change between regimes
    - Multiple targets can be generated with different tasks
    Returns (X, Y, metadata) where Y has shape (n_samples, n_targets).
    """
```

---

## 14. Performance Expectations

| Operation | Expected time | Notes |
|---|---|---|
| `step()` — warmup | <1ms | Just buffering |
| `step()` — prediction (single target) | <10ms | Feature selection + MAPIE predict |
| `step()` — prediction (3 targets) | <30ms | Linear in target count |
| `step()` — prediction + monitoring | <50ms | Add PIT/calibration |
| `step()` — prediction + SHAP (linear) | ~1–5s | Per target |
| `step()` — prediction + SHAP (ensemble) | ~5–20s | PermutationExplainer is slower |
| Retrain — Phase 3, elastic_net | ~5–30s | ElasticNetCV on ~50 features |
| Retrain — Phase 3, lightgbm | ~2–10s | Faster than CV'd linear |
| Retrain — Phase 3, ensemble | ~10–40s | Both models |
| Retrain — Phase 2 + 3 | ~1–10min | Stability selection dominates |
| Retrain — Phase 1 + 2 + 3 | ~2–15min | Clustering + screening + fit |

---

## 15. Out of Scope

The only feature deliberately excluded:

- **Streaming / online learning (River)**: The user specified on-demand reprocessing. Streaming requires a fundamentally different Phase 3 architecture. The current design does not preclude adding this later — it would be an alternative `model_type` that replaces the batch fit with incremental updates, while Phases 1, 2, 5, 6, 7 remain unchanged.

---

## 16. Implementation Order

1. **`config.py` + `types.py`** — All data structures. Test: instantiation, freezing, defaults, multi-target config.

2. **`phases/clustering.py`** — Pure function. Test on synthetic correlated data.

3. **`phases/screening.py`** — Pure function. Test feature recovery on synthetic data.

4. **`phases/modelling.py`** — Model factory (dispatches to linear/lightgbm/sklearn VotingRegressor|VotingClassifier) + conformalisation. Test all 9 combinations of `model_type` × `task`. Test MAPIE interval shapes.

5. **`monitoring/calibration.py`** — PIT, Brier, ECE. Test on synthetic well-specified and misspecified models.

6. **`monitoring/stability.py`** — SHAP + rank correlation. Test shapes, determinism.

7. **`monitoring/triggers.py`** — Decision tree. Test all branches, cooldown, multi-target worst-case.

8. **`cache.py`** — Wrapper. Test hit/miss, no-op mode.

9. **`pipeline.py`** — State machine. Integration test: full lifecycle on synthetic data for each `model_type` and single/multi-target.

10. **`trace.py`** — Query layer. Test aggregation, per-target filtering.

11. **Save/load invariant test + cache hit test.** Acceptance tests.

---

## Appendix A: MAPIE v1 API Quick Reference

```python
# Regression
from mapie.regression import SplitConformalRegressor, CrossConformalRegressor
# Classification
from mapie.classification import SplitConformalClassifier, CrossConformalClassifier
# Conformity scores
from mapie.conformity_scores import AbsoluteConformityScore, ResidualNormalisedScore

# Workflow:
#   1. estimator.fit(X_train, y_train)
#   2. mapie = SplitConformalRegressor(estimator=..., prefit=True, confidence_level=[0.9])
#   3. mapie.conformalize(X_conf, y_conf)
#   4. y_pred, y_pis = mapie.predict_interval(X_test)
```

## Appendix B: ruptures API Quick Reference

```python
import ruptures as rpt

# Algorithms: rpt.Pelt, rpt.Binseg, rpt.BottomUp, rpt.Window
# Cost models: "l1", "l2", "rbf", "linear", "normal", "ar"
# Custom cost: pass custom_cost=MyBaseCost() to any algorithm

algo = rpt.Pelt(model="rbf", min_size=60).fit(signal)
breakpoints = algo.predict(pen=10)  # returns [bp1, bp2, ..., n]
# Last element is always n. True breakpoints are breakpoints[:-1].
```

## Appendix C: stability-selection API Quick Reference

```python
from stability_selection import StabilitySelection

selector = StabilitySelection(
    base_estimator=sklearn_estimator,
    lambda_name="C",
    lambda_grid=np.logspace(-5, -1, 25),
    n_bootstrap_iterations=200,
    sample_fraction=0.5,
    threshold=0.6,
    n_jobs=-1,
)
selector.fit(X, y)
mask = selector.get_support()              # boolean
indices = selector.get_support(indices=True)  # integer
scores = selector.stability_scores_        # (n_features, n_lambda_values)
```