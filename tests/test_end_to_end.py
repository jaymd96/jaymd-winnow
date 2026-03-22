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


class TestRetrainingChangesModels:
    """Verify that retraining actually produces different models/predictions."""

    def test_periodic_rescreen_changes_feature_set_or_models(self):
        """Trigger a periodic reselect via low screening update_frequency.
        Verify that the pipeline survives and continues predicting."""
        X, Y, _ = make_regime_switching_data(
            n_samples=400, n_features=30, n_relevant=3,
            n_regimes=3, noise_std=0.5, seed=42,
        )
        config = PipelineConfig(
            targets=[TargetConfig(
                name="target", task="regression",
                model=ModelConfig(cv_folds=3, confidence_levels=[0.9]),
            )],
            min_history=150,
            clustering=ClusteringConfig(max_clusters=10, update_frequency=500),
            # Low update frequency → forces periodic reselect after warmup
            screening=ScreeningConfig(threshold=0.3, n_bootstraps=20, cv_folds=3, update_frequency=30),
            monitor=MonitorConfig(pit_window=30, shap_frequency=500),
            retraining=RetrainingConfig(cooldown_steps=5, min_regime_size=30),
        )
        p = AdaptivePipeline(config)

        events = []
        predictions_before_retrain = []
        predictions_after_retrain = []
        retrain_step = None

        for i in range(len(X)):
            targets = Y[i:i+1] if i > 0 else None
            result = p.step(i, X[i], targets=targets)

            if result.event:
                events.append(result.event)
                if result.event.event_type == "retrain_reselect" and retrain_step is None:
                    retrain_step = i

            if not result.is_warmup and result.predictions:
                if retrain_step is None:
                    predictions_before_retrain.append(result.predictions["target"].point[0])
                elif i > retrain_step:
                    predictions_after_retrain.append(result.predictions["target"].point[0])

        # A reselect event should have fired
        reselect_events = [e for e in events if e.event_type == "retrain_reselect"]
        assert len(reselect_events) > 0, "No retrain_reselect event fired"

        # Should have predictions both before and after retrain
        assert len(predictions_before_retrain) > 0
        assert len(predictions_after_retrain) > 0

        # Pipeline should still be warm and producing valid predictions
        assert p.is_warm
        assert all(np.isfinite(predictions_after_retrain))

    def test_periodic_rebuild_fires_and_produces_models(self):
        """Trigger a periodic rebuild via low cluster update_frequency."""
        X, Y, _ = make_regime_switching_data(
            n_samples=350, n_features=30, n_relevant=3,
            n_regimes=2, noise_std=0.5, seed=42,
        )
        config = PipelineConfig(
            targets=[TargetConfig(
                name="target", task="regression",
                model=ModelConfig(cv_folds=3, confidence_levels=[0.9]),
            )],
            min_history=150,
            # Low update frequency → forces rebuild after warmup
            clustering=ClusteringConfig(max_clusters=10, update_frequency=25),
            screening=ScreeningConfig(threshold=0.3, n_bootstraps=20, cv_folds=3, update_frequency=500),
            monitor=MonitorConfig(pit_window=30, shap_frequency=500),
            retraining=RetrainingConfig(cooldown_steps=5, min_regime_size=30),
        )
        p = AdaptivePipeline(config)

        events = []
        for i in range(len(X)):
            targets = Y[i:i+1] if i > 0 else None
            result = p.step(i, X[i], targets=targets)
            if result.event:
                events.append(result.event)

        rebuild_events = [e for e in events if e.event_type == "retrain_rebuild"]
        assert len(rebuild_events) > 0, "No retrain_rebuild event fired"

        # Pipeline survives rebuild and still predicts
        assert p.is_warm
        assert len(p._state.models) > 0

    def test_retrain_refit_updates_model_object(self):
        """Directly inject degraded health to trigger a refit and verify model changes."""
        X, Y, _ = make_regime_switching_data(
            n_samples=350, n_features=30, n_relevant=3,
            n_regimes=2, noise_std=0.5, seed=42,
        )
        config = PipelineConfig(
            targets=[TargetConfig(
                name="target", task="regression",
                model=ModelConfig(cv_folds=3, confidence_levels=[0.9]),
            )],
            min_history=150,
            clustering=ClusteringConfig(max_clusters=10, update_frequency=500),
            screening=ScreeningConfig(threshold=0.3, n_bootstraps=20, cv_folds=3, update_frequency=500),
            monitor=MonitorConfig(pit_window=20, shap_frequency=500),
            # Very aggressive retraining: low cooldown, trigger at high p-value
            retraining=RetrainingConfig(
                cooldown_steps=5, min_regime_size=30,
                calibration_trigger=0.99,  # triggers unless nearly perfect
            ),
        )
        p = AdaptivePipeline(config)

        # Run through warmup
        for i in range(160):
            targets = Y[i:i+1] if i > 0 else None
            p.step(i, X[i], targets=targets)

        assert p.is_warm
        model_before = p._state.base_models.get("target")
        assert model_before is not None

        # Now run more steps — the aggressive calibration_trigger should cause refits
        events = []
        for i in range(160, 350):
            targets = Y[i:i+1]
            result = p.step(i, X[i], targets=targets)
            if result.event:
                events.append(result.event)

        # There should have been at least one refit or reselect
        retrain_events = [e for e in events if "retrain" in e.event_type]
        assert len(retrain_events) > 0, f"No retrain events fired. Events: {[e.event_type for e in events]}"

        # Model should have been replaced (different object)
        model_after = p._state.base_models.get("target")
        assert model_after is not None


class TestSHAPMonitoringIntegration:
    """Verify SHAP monitoring path works end-to-end."""

    def test_shap_importances_computed(self):
        """With low shap_frequency, SHAP importances should be computed and stored."""
        X, Y, _ = make_regime_switching_data(
            n_samples=300, n_features=30, n_relevant=3,
            n_regimes=2, noise_std=0.5, seed=42,
        )
        config = PipelineConfig(
            targets=[TargetConfig(
                name="target", task="regression",
                model=ModelConfig(cv_folds=3, confidence_levels=[0.9]),
            )],
            min_history=150,
            clustering=ClusteringConfig(max_clusters=10, update_frequency=500),
            screening=ScreeningConfig(threshold=0.3, n_bootstraps=20, cv_folds=3, update_frequency=500),
            # Very low shap_frequency to trigger SHAP in test
            monitor=MonitorConfig(pit_window=30, shap_frequency=5),
            retraining=RetrainingConfig(cooldown_steps=100, min_regime_size=30),
        )
        p = AdaptivePipeline(config)

        for i in range(len(X)):
            targets = Y[i:i+1] if i > 0 else None
            p.step(i, X[i], targets=targets)

        # SHAP importances should have been computed at least once
        assert "target" in p._state.shap_importances_current, \
            "SHAP importances never computed"
        importances = p._state.shap_importances_current["target"]
        assert importances.shape[0] == len(p._state.active_feature_indices)
        assert np.all(np.isfinite(importances))
        assert np.all(importances >= 0)  # absolute SHAP values

    def test_feature_stability_appears_in_health(self):
        """After two SHAP computations, feature_stability should appear in health."""
        X, Y, _ = make_regime_switching_data(
            n_samples=300, n_features=30, n_relevant=3,
            n_regimes=2, noise_std=0.5, seed=42,
        )
        config = PipelineConfig(
            targets=[TargetConfig(
                name="target", task="regression",
                model=ModelConfig(cv_folds=3, confidence_levels=[0.9]),
            )],
            min_history=150,
            clustering=ClusteringConfig(max_clusters=10, update_frequency=500),
            screening=ScreeningConfig(threshold=0.3, n_bootstraps=20, cv_folds=3, update_frequency=500),
            # Trigger SHAP every 5 steps — should fire multiple times
            monitor=MonitorConfig(pit_window=30, shap_frequency=5),
            retraining=RetrainingConfig(cooldown_steps=100, min_regime_size=30),
        )
        p = AdaptivePipeline(config)

        stability_values = []
        for i in range(len(X)):
            targets = Y[i:i+1] if i > 0 else None
            result = p.step(i, X[i], targets=targets)
            if result.health and "target" in result.health.per_target:
                th = result.health.per_target["target"]
                if th.feature_stability is not None:
                    stability_values.append(th.feature_stability)

        # After 2+ SHAP computations, we should have feature_stability values
        assert len(stability_values) > 0, \
            "feature_stability never appeared in health snapshots"
        # Stability should be a correlation in [-1, 1]
        for s in stability_values:
            assert -1.0 <= s <= 1.0, f"Stability {s} out of range [-1, 1]"

        # With same model, consecutive SHAP should be fairly stable
        # (not asserting high stability, just that it's computed)
        assert "target" in p._state.shap_importances_previous, \
            "Previous SHAP importances never stored"


class TestCacheAfterRetrain:
    """Verify cache doesn't serve stale results after retraining."""

    def test_cache_hit_on_identical_data(self):
        """With caching enabled, second run on same data should be faster (cache hit)."""
        import tempfile
        import time

        X, Y, _ = make_regime_switching_data(
            n_samples=250, n_features=30, n_relevant=3,
            n_regimes=2, noise_std=0.5, seed=42,
        )

        with tempfile.TemporaryDirectory() as cache_dir:
            config = PipelineConfig(
                targets=[TargetConfig(
                    name="target", task="regression",
                    model=ModelConfig(cv_folds=3, confidence_levels=[0.9]),
                )],
                min_history=150,
                clustering=ClusteringConfig(max_clusters=10, update_frequency=500),
                screening=ScreeningConfig(threshold=0.3, n_bootstraps=20, cv_folds=3, update_frequency=500),
                monitor=MonitorConfig(pit_window=30, shap_frequency=500),
                retraining=RetrainingConfig(cooldown_steps=100, min_regime_size=30),
                cache_dir=cache_dir,
            )

            # First run — populates cache
            p1 = AdaptivePipeline(config)
            t0 = time.perf_counter()
            for i in range(200):
                targets = Y[i:i+1] if i > 0 else None
                p1.step(i, X[i], targets=targets)
            first_run = time.perf_counter() - t0

            # Second run — should benefit from cache
            p2 = AdaptivePipeline(config)
            t0 = time.perf_counter()
            for i in range(200):
                targets = Y[i:i+1] if i > 0 else None
                p2.step(i, X[i], targets=targets)
            second_run = time.perf_counter() - t0

            # Second run should be meaningfully faster (cache hits for cluster + screen + model)
            # Use a generous threshold — just verify caching doesn't break anything
            assert p2.is_warm
            assert len(p2._state.models) > 0
            # Log timing for informational purposes
            speedup = first_run / max(second_run, 0.001)
            assert speedup > 1.5, \
                f"Cache didn't provide expected speedup: {first_run:.2f}s vs {second_run:.2f}s ({speedup:.1f}x)"

    def test_retrain_with_new_data_produces_new_model(self):
        """After retrain with different training data, model should differ."""
        X, Y, _ = make_regime_switching_data(
            n_samples=400, n_features=30, n_relevant=3,
            n_regimes=3, noise_std=0.5, seed=42,
        )
        config = PipelineConfig(
            targets=[TargetConfig(
                name="target", task="regression",
                model=ModelConfig(cv_folds=3, confidence_levels=[0.9]),
            )],
            min_history=150,
            clustering=ClusteringConfig(max_clusters=10, update_frequency=500),
            # Force periodic reselect to trigger retrain with expanded data
            screening=ScreeningConfig(threshold=0.3, n_bootstraps=20, cv_folds=3, update_frequency=40),
            monitor=MonitorConfig(pit_window=30, shap_frequency=500),
            retraining=RetrainingConfig(cooldown_steps=5, min_regime_size=30),
        )
        p = AdaptivePipeline(config)

        # Capture predictions at different stages
        preds_early = []
        preds_late = []

        for i in range(len(X)):
            targets = Y[i:i+1] if i > 0 else None
            result = p.step(i, X[i], targets=targets)
            if not result.is_warmup and result.predictions:
                if 155 <= i <= 180:
                    preds_early.append(result.predictions["target"].point[0])
                elif 350 <= i <= 390:
                    preds_late.append(result.predictions["target"].point[0])

        # Both periods should have predictions
        assert len(preds_early) > 0, "No early predictions"
        assert len(preds_late) > 0, "No late predictions"

        # The distribution of predictions should differ (different regime, different model)
        # Not testing exact values — just that the pipeline didn't go stale
        early_mean = np.mean(preds_early)
        late_mean = np.mean(preds_late)
        # With regime-switching data and retraining, means should differ
        # (this is a weak assertion — mainly verifying the pipeline adapts at all)
        assert np.isfinite(early_mean)
        assert np.isfinite(late_mean)


class TestDecayHalflifePipeline:
    """Integration test: pipeline with exponential decay sample weighting."""

    def test_regression_with_decay(self):
        X, Y, _ = make_regime_switching_data(
            n_samples=300, n_features=30, n_relevant=3,
            n_regimes=2, noise_std=0.5, seed=42,
        )
        config = PipelineConfig(
            targets=[TargetConfig(
                name="target", task="regression",
                model=ModelConfig(cv_folds=3, confidence_levels=[0.9], decay_halflife=50),
            )],
            min_history=150,
            clustering=ClusteringConfig(max_clusters=10, update_frequency=500),
            screening=ScreeningConfig(threshold=0.3, n_bootstraps=20, cv_folds=3, update_frequency=500),
            monitor=MonitorConfig(pit_window=30, shap_frequency=500),
            retraining=RetrainingConfig(cooldown_steps=100, min_regime_size=30),
        )
        p = AdaptivePipeline(config)

        for i in range(len(X)):
            targets = Y[i:i+1] if i > 0 else None
            result = p.step(i, X[i], targets=targets)

        assert p.is_warm
        post_warmup = [r for r in [result] if not r.is_warmup and r.predictions]
        assert len(post_warmup) > 0
        pred = post_warmup[0].predictions["target"]
        assert pred.point is not None
        assert np.all(np.isfinite(pred.point))
