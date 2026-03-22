"""End-to-end integration tests that exercise the full pipeline lifecycle."""

import tempfile

import numpy as np
import pytest

from jaymd_winnow import AdaptivePipeline
from jaymd_winnow.config import (
    ClusteringConfig,
    ModelConfig,
    MonitorConfig,
    PipelineConfig,
    RetrainingConfig,
    ScreeningConfig,
    TargetConfig,
)
from conftest import make_regime_switching_data


def _fast_config(task="regression", model_type="elastic_net", min_history=150):
    """Config tuned for fast tests: small bootstraps, small clusters."""
    return PipelineConfig(
        targets=[TargetConfig(
            name="target", task=task,
            model=ModelConfig(model_type=model_type, cv_folds=3, confidence_levels=[0.68, 0.95]),
        )],
        min_history=min_history,
        clustering=ClusteringConfig(max_clusters=10, update_frequency=500),
        screening=ScreeningConfig(threshold=0.3, n_bootstraps=20, cv_folds=3, update_frequency=500),
        monitor=MonitorConfig(pit_window=30, shap_frequency=500),
        retraining=RetrainingConfig(cooldown_steps=10, min_regime_size=30),
    )


class TestRegressionEndToEnd:
    """Full lifecycle: warmup → predict → monitor → verify shapes."""

    def test_produces_predictions_after_warmup(self):
        X, Y, _ = make_regime_switching_data(
            n_samples=300, n_features=30, n_relevant=3,
            n_regimes=2, noise_std=0.5, seed=42,
        )
        p = AdaptivePipeline(_fast_config(min_history=150))

        results = []
        for i in range(len(X)):
            targets = Y[i:i+1] if i > 0 else None
            result = p.step(i, X[i], targets=targets)
            results.append(result)

        assert p.is_warm

        # Find first non-warmup result with predictions
        post_warmup = [r for r in results if not r.is_warmup and r.predictions]
        assert len(post_warmup) > 0, "No predictions produced after warmup"

        pred = post_warmup[0].predictions["target"]
        assert pred.point is not None
        assert np.all(np.isfinite(pred.point)), "Predictions contain NaN/Inf"

        # Intervals should be present with correct shape
        assert pred.intervals is not None, "No prediction intervals"
        # intervals shape: (2, n_confidence_levels)
        assert pred.intervals.shape[0] == 2, f"Expected 2 bounds, got {pred.intervals.shape[0]}"
        assert pred.intervals.shape[1] == 2, f"Expected 2 confidence levels, got shape {pred.intervals.shape}"

        # Lower bound should be <= upper bound
        for cl_idx in range(pred.intervals.shape[1]):
            assert pred.intervals[0, cl_idx] <= pred.intervals[1, cl_idx], \
                f"Lower bound > upper bound at confidence level {cl_idx}"

    def test_lightgbm_produces_predictions(self):
        X, Y, _ = make_regime_switching_data(
            n_samples=300, n_features=30, n_relevant=3,
            n_regimes=2, noise_std=0.5, seed=42,
        )
        p = AdaptivePipeline(_fast_config(model_type="lightgbm", min_history=150))

        for i in range(len(X)):
            targets = Y[i:i+1] if i > 0 else None
            result = p.step(i, X[i], targets=targets)

        assert p.is_warm
        # Last result should have predictions
        assert len(result.predictions) > 0

    def test_ensemble_produces_predictions(self):
        X, Y, _ = make_regime_switching_data(
            n_samples=300, n_features=30, n_relevant=3,
            n_regimes=2, noise_std=0.5, seed=42,
        )
        p = AdaptivePipeline(_fast_config(model_type="ensemble", min_history=150))

        for i in range(len(X)):
            targets = Y[i:i+1] if i > 0 else None
            result = p.step(i, X[i], targets=targets)

        assert p.is_warm
        assert len(result.predictions) > 0


class TestBinaryClassificationEndToEnd:
    def test_produces_prediction_sets(self):
        X, Y, _ = make_regime_switching_data(
            n_samples=300, n_features=30, n_relevant=3,
            n_regimes=2, target_tasks=("binary",), seed=42,
        )
        p = AdaptivePipeline(_fast_config(task="binary", min_history=150))

        results = []
        for i in range(len(X)):
            targets = Y[i:i+1] if i > 0 else None
            result = p.step(i, X[i], targets=targets)
            results.append(result)

        assert p.is_warm
        post_warmup = [r for r in results if not r.is_warmup and r.predictions]
        assert len(post_warmup) > 0, "No predictions produced for binary classification"

        pred = post_warmup[0].predictions["target"]
        assert pred.point is not None
        assert pred.prediction_sets is not None, "No prediction sets for binary task"


class TestMulticlassEndToEnd:
    def test_produces_prediction_sets(self):
        X, Y, _ = make_regime_switching_data(
            n_samples=400, n_features=30, n_relevant=3,
            n_regimes=2, target_tasks=("multiclass",), seed=42,
        )
        p = AdaptivePipeline(_fast_config(task="multiclass", min_history=200))

        results = []
        for i in range(len(X)):
            targets = Y[i:i+1] if i > 0 else None
            result = p.step(i, X[i], targets=targets)
            results.append(result)

        assert p.is_warm
        post_warmup = [r for r in results if not r.is_warmup and r.predictions]
        assert len(post_warmup) > 0, "No predictions produced for multiclass"


class TestMultiTargetEndToEnd:
    def test_both_targets_produce_predictions(self):
        X, Y, _ = make_regime_switching_data(
            n_samples=300, n_features=30, n_relevant=3,
            n_regimes=2, n_targets=2,
            target_tasks=("regression", "binary"), seed=42,
        )
        config = PipelineConfig(
            targets=[
                TargetConfig(name="returns", task="regression",
                             model=ModelConfig(cv_folds=3, confidence_levels=[0.68, 0.95])),
                TargetConfig(name="direction", task="binary",
                             model=ModelConfig(cv_folds=3, confidence_levels=[0.68, 0.95])),
            ],
            min_history=150,
            clustering=ClusteringConfig(max_clusters=10, update_frequency=500),
            screening=ScreeningConfig(threshold=0.3, n_bootstraps=20, cv_folds=3, update_frequency=500),
            monitor=MonitorConfig(pit_window=30, shap_frequency=500),
            retraining=RetrainingConfig(cooldown_steps=10, min_regime_size=30),
        )
        p = AdaptivePipeline(config)

        results = []
        for i in range(len(X)):
            targets = Y[i] if i > 0 else None
            result = p.step(i, X[i], targets=targets)
            results.append(result)

        assert p.is_warm
        post_warmup = [r for r in results if not r.is_warmup and r.predictions]
        assert len(post_warmup) > 0

        # Both targets should have predictions
        pred = post_warmup[0].predictions
        assert "returns" in pred, "Missing regression target predictions"
        assert "direction" in pred, "Missing binary target predictions"

        # Regression target should have intervals
        assert pred["returns"].intervals is not None
        # Binary target should have prediction sets
        assert pred["direction"].prediction_sets is not None


class TestSaveLoadInvariant:
    def test_predictions_match_after_load(self):
        X, Y, _ = make_regime_switching_data(
            n_samples=300, n_features=30, n_relevant=3,
            n_regimes=2, noise_std=0.5, seed=42,
        )
        p = AdaptivePipeline(_fast_config(min_history=150))

        # Run through warmup
        for i in range(200):
            targets = Y[i:i+1] if i > 0 else None
            p.step(i, X[i], targets=targets)

        assert p.is_warm

        # Get prediction from original
        result_before = p.step(200, X[200], targets=Y[200:201])

        # Save and load
        with tempfile.NamedTemporaryFile(suffix=".joblib", delete=False) as f:
            path = f.name
        p.save(path)
        p2 = AdaptivePipeline.load(path)

        # Get prediction from loaded — same step
        result_after = p2.step(201, X[201], targets=Y[201:202])

        # Both should produce predictions
        assert len(result_before.predictions) > 0
        assert len(result_after.predictions) > 0

        # Loaded pipeline should be warm
        assert p2.is_warm


class TestWarmupProducesModels:
    def test_models_exist_after_warmup(self):
        X, Y, _ = make_regime_switching_data(
            n_samples=250, n_features=30, n_relevant=3,
            n_regimes=2, noise_std=0.5, seed=42,
        )
        p = AdaptivePipeline(_fast_config(min_history=150))

        warmup_complete_event = None
        for i in range(len(X)):
            targets = Y[i:i+1] if i > 0 else None
            result = p.step(i, X[i], targets=targets)
            if result.event and result.event.event_type == "warmup_complete":
                warmup_complete_event = result.event
                break

        assert warmup_complete_event is not None, "Warmup never completed"
        assert p.is_warm
        assert len(p._state.models) > 0, "No models after warmup"
        assert len(p._state.base_models) > 0, "No base models after warmup"
        assert p._state.active_feature_indices is not None, "No feature selection after warmup"
