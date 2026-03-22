"""All configuration via frozen dataclasses. Immutable after construction."""

from dataclasses import dataclass, field
from typing import Any, Literal, Optional


@dataclass(frozen=True)
class ClusteringConfig:
    max_clusters: int = 200
    method: str = "ward"
    distance: str = "correlation"
    update_frequency: int = 63


@dataclass(frozen=True)
class ScreeningConfig:
    threshold: float = 0.6
    n_bootstraps: int = 200
    cv_folds: int = 5
    update_frequency: int = 63


@dataclass(frozen=True)
class ModelConfig:
    model_type: Literal["elastic_net", "lightgbm", "ensemble"] = "elastic_net"
    l1_ratios: list[float] = field(default_factory=lambda: [0.5, 0.7, 0.9, 0.95])
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
    ensemble_weights: Optional[list[float]] = None
    decay_halflife: Optional[int] = None  # steps; None = uniform weighting
    cv_folds: int = 5
    confidence_levels: list[float] = field(default_factory=lambda: [0.68, 0.95])
    conformal_method: str = "split"
    train_fraction: float = 0.6
    conformalize_fraction: float = 0.2


@dataclass(frozen=True)
class RegimeConfig:
    algorithm: str = "pelt"
    cost_model: str = "rbf"
    custom_cost: Any = None
    penalty: float = 10.0
    min_segment_size: int = 60


@dataclass(frozen=True)
class MonitorConfig:
    pit_window: int = 60
    stability_window: int = 20
    shap_frequency: int = 20
    calibration_alert_threshold: float = 0.05


@dataclass(frozen=True)
class RetrainingConfig:
    calibration_trigger: float = 0.05
    stability_trigger: float = 0.3
    cooldown_steps: int = 20
    min_regime_size: int = 60


@dataclass(frozen=True)
class TargetConfig:
    """Configuration for a single target."""
    name: str
    task: Literal["regression", "binary", "multiclass"] = "regression"
    model: ModelConfig = field(default_factory=ModelConfig)


@dataclass(frozen=True)
class PipelineConfig:
    targets: list[TargetConfig] = field(default_factory=lambda: [
        TargetConfig(name="target", task="regression")
    ])
    min_history: int = 252
    clustering: ClusteringConfig = field(default_factory=ClusteringConfig)
    screening: ScreeningConfig = field(default_factory=ScreeningConfig)
    regime: RegimeConfig = field(default_factory=RegimeConfig)
    monitor: MonitorConfig = field(default_factory=MonitorConfig)
    retraining: RetrainingConfig = field(default_factory=RetrainingConfig)
    cache_dir: Optional[str] = None
