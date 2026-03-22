"""AdaptivePipeline — the step() state machine."""

import logging
from typing import Any, Optional

import joblib
import numpy as np

from jaymd_winnow import __version__
from jaymd_winnow.cache import PipelineCache
from jaymd_winnow.config import ModelConfig, PipelineConfig, TargetConfig
from jaymd_winnow.monitoring.calibration import (
    compute_brier_reliability,
    compute_ece,
    compute_pit_value,
    check_pit_uniformity,
)
from jaymd_winnow.monitoring.stability import (
    compute_importance_rank_stability,
    compute_shap_importances,
)
from jaymd_winnow.monitoring.triggers import evaluate_trigger
from jaymd_winnow.phases.clustering import cluster_features
from jaymd_winnow.phases.modelling import (
    build_base_model,
    conformalise_model,
    detect_regimes,
)
from jaymd_winnow.phases.screening import screen_features
from jaymd_winnow.types import (
    HealthSnapshot,
    LifecycleEvent,
    StepResult,
    TargetHealth,
    TargetPrediction,
)

logger = logging.getLogger(__name__)


class _PipelineState:
    """Mutable internal state of the pipeline."""

    def __init__(self):
        # Data buffers
        self.timestamps: list = []
        self.features_buffer: list[np.ndarray] = []
        self.targets_buffer: list[np.ndarray] = []
        self.pending_predictions: dict[str, Any] = {}

        # Phase 1+2 outputs (shared across targets)
        self.cluster_labels: Optional[np.ndarray] = None
        self.representative_indices: Optional[np.ndarray] = None
        self.selected_feature_indices: Optional[np.ndarray] = None
        self.active_feature_indices: Optional[np.ndarray] = None

        # Phase 3 outputs (per target)
        self.models: dict[str, object] = {}
        self.base_models: dict[str, object] = {}
        self.conformity_scores: dict[str, np.ndarray] = {}

        # Monitoring state (per target)
        self.pit_values: dict[str, list] = {}
        self.brier_values: dict[str, list] = {}
        self.shap_importances_previous: dict[str, np.ndarray] = {}
        self.shap_importances_current: dict[str, np.ndarray] = {}

        # Lifecycle
        self.step_count: int = 0
        self.steps_since_last_retrain: int = 0
        self.steps_since_last_cluster: int = 0
        self.steps_since_last_screen: int = 0
        self.steps_since_last_shap: int = 0
        self.is_warm: bool = False


class AdaptivePipeline:
    """Adaptive model lifecycle for financial signal selection and prediction.

    Feed data one step at a time via step(). The pipeline automatically
    clusters features, selects relevant signals, fits calibrated models,
    monitors health, and triggers retraining when assumptions break.
    """

    def __init__(self, config: PipelineConfig):
        self._config = config
        self._state = _PipelineState()
        self._init_cache(config.cache_dir)
        self._n_features: Optional[int] = None

    def _init_cache(self, cache_dir: Optional[str]):
        self._cache = PipelineCache(cache_dir)
        self._cached_cluster = self._cache.cache(cluster_features)
        self._cached_screen = self._cache.cache(screen_features)
        self._cached_build_model = self._cache.cache(build_base_model)
        self._cached_shap = self._cache.cache(compute_shap_importances)

    # ------------------------------------------------------------------
    # Convenience constructors
    # ------------------------------------------------------------------

    @classmethod
    def regression(cls, min_history: int = 252, cache_dir: Optional[str] = None, **overrides) -> "AdaptivePipeline":
        """Single-target regression with elastic_net."""
        model_kw = {k: v for k, v in overrides.items() if k in ModelConfig.__dataclass_fields__}
        other_kw = {k: v for k, v in overrides.items() if k not in ModelConfig.__dataclass_fields__}
        config = PipelineConfig(
            targets=[TargetConfig(name="target", task="regression", model=ModelConfig(**model_kw))],
            min_history=min_history,
            cache_dir=cache_dir,
            **other_kw,
        )
        return cls(config)

    @classmethod
    def classification(cls, min_history: int = 500, cache_dir: Optional[str] = None, **overrides) -> "AdaptivePipeline":
        """Single-target binary classification."""
        model_kw = {k: v for k, v in overrides.items() if k in ModelConfig.__dataclass_fields__}
        other_kw = {k: v for k, v in overrides.items() if k not in ModelConfig.__dataclass_fields__}
        config = PipelineConfig(
            targets=[TargetConfig(name="target", task="binary", model=ModelConfig(**model_kw))],
            min_history=min_history,
            cache_dir=cache_dir,
            **other_kw,
        )
        return cls(config)

    @classmethod
    def multi_target(
        cls,
        targets: list[dict],
        min_history: int = 252,
        cache_dir: Optional[str] = None,
        **overrides,
    ) -> "AdaptivePipeline":
        """Multi-target pipeline.

        Args:
            targets: List of dicts like [{"name": "returns", "task": "regression"},
                     {"name": "direction", "task": "binary", "model_type": "lightgbm"}].
        """
        target_configs = []
        for t in targets:
            model_kw = {}
            if "model_type" in t:
                model_kw["model_type"] = t["model_type"]
            tc = TargetConfig(
                name=t["name"],
                task=t.get("task", "regression"),
                model=ModelConfig(**model_kw),
            )
            target_configs.append(tc)

        config = PipelineConfig(
            targets=target_configs,
            min_history=min_history,
            cache_dir=cache_dir,
            **overrides,
        )
        return cls(config)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def config(self) -> PipelineConfig:
        return self._config

    @property
    def is_warm(self) -> bool:
        return self._state.is_warm

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def step(
        self,
        timestamp: Any,
        features: np.ndarray,
        targets: Optional[np.ndarray] = None,
    ) -> StepResult:
        """Advance the pipeline by one time step."""
        features = np.asarray(features, dtype=float)
        if features.ndim != 1:
            raise ValueError(f"features must be 1D, got shape {features.shape}")

        if self._n_features is None:
            self._n_features = features.shape[0]
        elif features.shape[0] != self._n_features:
            raise ValueError(
                f"Feature dimensionality changed: expected {self._n_features}, "
                f"got {features.shape[0]}"
            )

        # Normalise targets
        targets_arr = self._normalise_targets(targets)

        # 1. INGEST
        self._state.timestamps.append(timestamp)
        self._state.features_buffer.append(features)
        if targets_arr is not None:
            self._state.targets_buffer.append(targets_arr)

        self._state.step_count += 1
        self._state.steps_since_last_retrain += 1
        self._state.steps_since_last_cluster += 1
        self._state.steps_since_last_screen += 1
        self._state.steps_since_last_shap += 1

        # 2. UPDATE MONITORS
        health = self._update_monitors(targets_arr)

        # 3. EVALUATE TRIGGERS
        event = self._evaluate_triggers(timestamp, health)

        # 4. RETRAIN (if triggered by event)
        if event is not None:
            self._execute_retrain(event.event_type, event)

        # 5. PREDICT
        predictions = self._make_predictions(features)

        # 6. RETURN
        return StepResult(
            timestamp=timestamp,
            predictions=predictions,
            selected_features=self._state.active_feature_indices,
            health=health,
            event=event,
            is_warmup=not self._state.is_warm,
        )

    def predict(self, timestamp: Any, features: np.ndarray) -> StepResult:
        """Make a prediction without ingesting targets."""
        return self.step(timestamp, features, targets=None)

    def observe(self, timestamp: Any, targets: np.ndarray) -> StepResult:
        """Ingest realised targets without making a new prediction.

        Re-uses the last feature vector for buffering consistency.
        """
        if not self._state.features_buffer:
            raise ValueError("Cannot observe before any features have been ingested")
        last_features = self._state.features_buffer[-1]
        return self.step(timestamp, last_features, targets=targets)

    def save(self, path: str) -> None:
        """Serialise entire pipeline state to disk."""
        joblib.dump({
            "config": self._config,
            "state": self._state,
            "version": __version__,
        }, path)

    @classmethod
    def load(cls, path: str) -> "AdaptivePipeline":
        """Deserialise a pipeline."""
        data = joblib.load(path)
        pipeline = cls(data["config"])
        pipeline._state = data["state"]
        if pipeline._state.features_buffer:
            pipeline._n_features = pipeline._state.features_buffer[0].shape[0]
        return pipeline

    def set_cache_dir(self, cache_dir: str) -> None:
        """Re-attach a cache directory after loading."""
        self._init_cache(cache_dir)

    # ------------------------------------------------------------------
    # Internal: target normalisation
    # ------------------------------------------------------------------

    def _normalise_targets(self, targets: Optional[np.ndarray]) -> Optional[np.ndarray]:
        if targets is None:
            return None
        targets = np.asarray(targets, dtype=float)
        n_targets = len(self._config.targets)
        if targets.ndim == 0:
            if n_targets != 1:
                raise ValueError(
                    f"Scalar target provided but pipeline has {n_targets} targets"
                )
            targets = targets.reshape(1)
        if targets.shape[0] != n_targets:
            raise ValueError(
                f"Expected {n_targets} targets, got {targets.shape[0]}"
            )
        return targets

    # ------------------------------------------------------------------
    # Internal: monitoring
    # ------------------------------------------------------------------

    def _update_monitors(self, targets_arr: Optional[np.ndarray]) -> Optional[HealthSnapshot]:
        if not self._state.is_warm or not self._state.models:
            return None

        if targets_arr is None:
            return None

        health = HealthSnapshot()

        for i, target_cfg in enumerate(self._config.targets):
            name = target_cfg.name
            if name not in self._state.models:
                continue

            y_actual = targets_arr[i]
            th = TargetHealth(target_name=name)

            # Calibration monitoring
            if target_cfg.task == "regression":
                th = self._monitor_regression(name, y_actual, th)
            elif target_cfg.task == "binary":
                th = self._monitor_binary(name, y_actual, th)
            elif target_cfg.task == "multiclass":
                th = self._monitor_multiclass(name, y_actual, th)

            # SHAP stability monitoring
            if self._state.steps_since_last_shap >= self._config.monitor.shap_frequency:
                th = self._monitor_shap_stability(name, target_cfg, th)

            health.per_target[name] = th

        # Reset SHAP counter if we computed SHAP this step
        if self._state.steps_since_last_shap >= self._config.monitor.shap_frequency:
            self._state.steps_since_last_shap = 0

        return health

    def _monitor_regression(self, name: str, y_actual: float, th: TargetHealth) -> TargetHealth:
        if name in self._state.pending_predictions and name in self._state.conformity_scores:
            y_pred = self._state.pending_predictions[name]
            scores = self._state.conformity_scores[name]
            pit = compute_pit_value(y_actual, scores, y_pred)

            if name not in self._state.pit_values:
                self._state.pit_values[name] = []
            self._state.pit_values[name].append(pit)

            window = self._config.monitor.pit_window
            if len(self._state.pit_values[name]) >= window:
                recent = np.array(self._state.pit_values[name][-window:])
                _, pvalue = check_pit_uniformity(recent)
                th.calibration_pvalue = pvalue

        return th

    def _monitor_binary(self, name: str, y_actual: float, th: TargetHealth) -> TargetHealth:
        if name in self._state.pending_predictions:
            pred_info = self._state.pending_predictions.get(f"{name}_prob")
            if pred_info is not None:
                if name not in self._state.brier_values:
                    self._state.brier_values[name] = {"y_true": [], "y_prob": []}
                self._state.brier_values[name]["y_true"].append(y_actual)
                self._state.brier_values[name]["y_prob"].append(pred_info)

                window = self._config.monitor.pit_window
                if len(self._state.brier_values[name]["y_true"]) >= window:
                    y_true = np.array(self._state.brier_values[name]["y_true"][-window:])
                    y_prob = np.array(self._state.brier_values[name]["y_prob"][-window:])
                    rel = compute_brier_reliability(y_true, y_prob)
                    # Convert to p-value-like metric: low reliability = good
                    # Use 1 - reliability as a proxy (0 = miscalibrated, 1 = perfect)
                    th.brier_reliability = rel
                    th.calibration_pvalue = max(0.0, 1.0 - rel * 10)

        return th

    def _monitor_multiclass(self, name: str, y_actual: float, th: TargetHealth) -> TargetHealth:
        if name in self._state.pending_predictions:
            pred_info = self._state.pending_predictions.get(f"{name}_prob")
            if pred_info is not None:
                if name not in self._state.brier_values:
                    self._state.brier_values[name] = {"y_true": [], "y_prob": []}
                self._state.brier_values[name]["y_true"].append(int(y_actual))
                self._state.brier_values[name]["y_prob"].append(pred_info)

                window = self._config.monitor.pit_window
                if len(self._state.brier_values[name]["y_true"]) >= window:
                    y_true = np.array(self._state.brier_values[name]["y_true"][-window:])
                    y_prob = np.array(self._state.brier_values[name]["y_prob"][-window:])
                    ece = compute_ece(y_true, y_prob)
                    th.ece = ece
                    th.calibration_pvalue = max(0.0, 1.0 - ece * 10)

        return th

    def _monitor_shap_stability(self, name: str, target_cfg: TargetConfig, th: TargetHealth) -> TargetHealth:
        if name not in self._state.base_models:
            return th

        base_model = self._state.base_models[name]
        X = self._get_recent_features_matrix()
        if X is None or X.shape[0] < 10:
            return th

        X_selected = X[:, self._state.active_feature_indices]

        try:
            importances = self._cached_shap(base_model, X_selected)
        except Exception:
            logger.warning("SHAP computation failed for target %s", name, exc_info=True)
            return th

        if name in self._state.shap_importances_current:
            self._state.shap_importances_previous[name] = self._state.shap_importances_current[name]

        self._state.shap_importances_current[name] = importances

        if name in self._state.shap_importances_previous:
            stability = compute_importance_rank_stability(
                importances, self._state.shap_importances_previous[name]
            )
            th.feature_stability = stability

        return th

    # ------------------------------------------------------------------
    # Internal: trigger evaluation
    # ------------------------------------------------------------------

    def _evaluate_triggers(self, timestamp: Any, health: Optional[HealthSnapshot]) -> Optional[LifecycleEvent]:
        # Warmup completion
        if not self._state.is_warm and self._state.step_count >= self._config.min_history:
            if len(self._state.targets_buffer) >= self._config.min_history:
                return LifecycleEvent(
                    event_type="warmup_complete",
                    timestamp=timestamp,
                    reason="Minimum history reached, triggering initial training",
                )

        if not self._state.is_warm:
            return None

        # Periodic cluster update
        if self._state.steps_since_last_cluster >= self._config.clustering.update_frequency:
            return LifecycleEvent(
                event_type="retrain_rebuild",
                timestamp=timestamp,
                reason=f"Periodic cluster update (every {self._config.clustering.update_frequency} steps)",
            )

        # Periodic screen update
        if self._state.steps_since_last_screen >= self._config.screening.update_frequency:
            return LifecycleEvent(
                event_type="retrain_reselect",
                timestamp=timestamp,
                reason=f"Periodic screening update (every {self._config.screening.update_frequency} steps)",
            )

        # Monitor-driven triggers
        if health is not None:
            trigger = evaluate_trigger(
                health, self._config.retraining, self._state.steps_since_last_retrain
            )
            if trigger == "refit":
                return LifecycleEvent(
                    event_type="retrain_refit",
                    timestamp=timestamp,
                    reason="Calibration degraded, refitting models",
                    details={"worst_calibration": health.worst_calibration},
                )
            elif trigger == "reselect":
                return LifecycleEvent(
                    event_type="retrain_reselect",
                    timestamp=timestamp,
                    reason="Feature importance structure changed, reselecting features",
                    details={
                        "worst_calibration": health.worst_calibration,
                        "worst_stability": health.worst_stability,
                    },
                )

        return None

    # ------------------------------------------------------------------
    # Internal: retraining
    # ------------------------------------------------------------------

    def _execute_retrain(self, event_type: str, event: LifecycleEvent):
        X = np.array(self._state.features_buffer)
        Y = np.array(self._state.targets_buffer)

        # Align X and Y: targets may be shorter (first step often has no target)
        min_len = min(len(X), len(Y))
        X = X[-min_len:]
        Y = Y[-min_len:]

        if event_type == "warmup_complete" or event_type == "retrain_rebuild":
            self._run_phase1(X)
            self._run_phase2(X, Y)
            self._run_phase3(X, Y)
            self._state.is_warm = True
            self._state.steps_since_last_cluster = 0
            self._state.steps_since_last_screen = 0

        elif event_type == "retrain_reselect":
            self._run_phase2(X, Y)
            self._run_phase3(X, Y)
            self._state.steps_since_last_screen = 0

        elif event_type == "retrain_refit":
            self._run_phase3(X, Y)

        self._state.steps_since_last_retrain = 0

    def _run_phase1(self, X: np.ndarray):
        labels, reps = self._cached_cluster(X, self._config.clustering)
        self._state.cluster_labels = labels
        self._state.representative_indices = reps
        logger.info("Phase 1: %d clusters, %d representatives", labels.max() + 1, len(reps))

    def _run_phase2(self, X: np.ndarray, Y: np.ndarray):
        if self._state.representative_indices is None:
            self._run_phase1(X)

        X_rep = X[:, self._state.representative_indices]
        # Use first target for screening
        first_target = self._config.targets[0]
        y_screen = Y[:, 0]

        selected = self._cached_screen(X_rep, y_screen, first_target.task, self._config.screening)
        self._state.selected_feature_indices = selected
        # Map back to original feature space
        self._state.active_feature_indices = self._state.representative_indices[selected]
        logger.info("Phase 2: %d features selected", len(selected))

    def _run_phase3(self, X: np.ndarray, Y: np.ndarray):
        if self._state.active_feature_indices is None:
            return

        # Detect regimes using first target
        y_first = Y[:, 0]
        breakpoints = detect_regimes(y_first, self._config.regime)

        last_break = breakpoints[-1] if breakpoints else 0
        # Ensure minimum regime size
        if (len(y_first) - last_break) < self._config.retraining.min_regime_size:
            last_break = max(0, len(y_first) - self._config.retraining.min_regime_size)

        X_window = X[last_break:]
        Y_window = Y[last_break:]
        X_window_selected = X_window[:, self._state.active_feature_indices]

        n = len(X_window_selected)
        train_end = int(n * self._config.targets[0].model.train_fraction)
        conf_end = train_end + int(n * self._config.targets[0].model.conformalize_fraction)

        X_train = X_window_selected[:train_end]
        X_conf = X_window_selected[train_end:conf_end]

        for i, target_cfg in enumerate(self._config.targets):
            y_target = Y_window[:, i]
            y_train = y_target[:train_end]
            y_conf = y_target[train_end:conf_end]

            if len(y_train) < 10 or len(y_conf) < 5:
                logger.warning("Insufficient data for target %s, skipping", target_cfg.name)
                continue

            base = self._cached_build_model(X_train, y_train, target_cfg.task, target_cfg.model)
            conformal = conformalise_model(base, X_conf, y_conf, target_cfg.task, target_cfg.model)

            self._state.models[target_cfg.name] = conformal
            self._state.base_models[target_cfg.name] = base

            # Store conformity scores for PIT monitoring
            if target_cfg.task == "regression":
                try:
                    scores = conformal._mapie_regressor.conformity_scores_
                    self._state.conformity_scores[target_cfg.name] = scores
                except AttributeError:
                    pass

        logger.info("Phase 3: models fitted for %d targets", len(self._config.targets))

    # ------------------------------------------------------------------
    # Internal: prediction
    # ------------------------------------------------------------------

    def _make_predictions(self, features: np.ndarray) -> dict[str, TargetPrediction]:
        predictions = {}
        if not self._state.is_warm or self._state.active_feature_indices is None:
            return predictions

        x = features[self._state.active_feature_indices].reshape(1, -1)

        for target_cfg in self._config.targets:
            name = target_cfg.name
            if name not in self._state.models:
                continue

            model = self._state.models[name]
            tp = TargetPrediction(target_name=name)

            if target_cfg.task == "regression":
                y_pred, y_pis = model.predict_interval(x)
                tp.point = y_pred.ravel()
                tp.intervals = y_pis[0]  # shape (2, n_confidence_levels)
                self._state.pending_predictions[name] = float(y_pred[0])

            elif target_cfg.task in ("binary", "multiclass"):
                y_pred, y_sets = model.predict_set(x)
                tp.point = y_pred.ravel()
                tp.prediction_sets = y_sets[0]  # shape (n_classes, n_confidence_levels)
                # Store predicted probabilities for monitoring
                try:
                    proba = model.estimator_.predict_proba(x)
                    if target_cfg.task == "binary":
                        self._state.pending_predictions[name] = float(y_pred[0])
                        self._state.pending_predictions[f"{name}_prob"] = float(proba[0, 1])
                    else:
                        self._state.pending_predictions[name] = float(y_pred[0])
                        self._state.pending_predictions[f"{name}_prob"] = proba[0]
                except Exception:
                    self._state.pending_predictions[name] = float(y_pred[0])

            predictions[name] = tp

        return predictions

    # ------------------------------------------------------------------
    # Internal: helpers
    # ------------------------------------------------------------------

    def _get_recent_features_matrix(self, max_rows: int = 200) -> Optional[np.ndarray]:
        if not self._state.features_buffer:
            return None
        recent = self._state.features_buffer[-max_rows:]
        return np.array(recent)
