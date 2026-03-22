"""Integration tests for the AdaptivePipeline state machine."""

import tempfile

import numpy as np
import pytest

from jaymd_winnow import AdaptivePipeline
from jaymd_winnow.config import ModelConfig, PipelineConfig, TargetConfig
from conftest import make_regime_switching_data


class TestPipelineConstruction:
    def test_regression_constructor(self):
        p = AdaptivePipeline.regression(min_history=100)
        assert p.config.targets[0].task == "regression"
        assert p.config.min_history == 100

    def test_classification_constructor(self):
        p = AdaptivePipeline.classification(min_history=200)
        assert p.config.targets[0].task == "binary"

    def test_multi_target_constructor(self):
        p = AdaptivePipeline.multi_target(
            targets=[
                {"name": "returns", "task": "regression"},
                {"name": "direction", "task": "binary", "model_type": "lightgbm"},
            ]
        )
        assert len(p.config.targets) == 2
        assert p.config.targets[1].model.model_type == "lightgbm"


class TestPipelineStep:
    def test_warmup_buffering(self):
        p = AdaptivePipeline.regression(min_history=10)
        rng = np.random.RandomState(42)

        for i in range(5):
            result = p.step(i, rng.randn(20), targets=np.array([rng.randn()]) if i > 0 else None)
            assert result.is_warmup is True
            assert len(result.predictions) == 0

    def test_feature_dim_change_raises(self):
        p = AdaptivePipeline.regression(min_history=10)
        rng = np.random.RandomState(42)
        p.step(0, rng.randn(20))
        with pytest.raises(ValueError, match="dimensionality"):
            p.step(1, rng.randn(30))

    def test_scalar_target_accepted(self):
        p = AdaptivePipeline.regression(min_history=10)
        rng = np.random.RandomState(42)
        p.step(0, rng.randn(20))
        result = p.step(1, rng.randn(20), targets=1.5)
        assert result is not None


class TestRegressionBacktest:
    def test_elastic_net_lifecycle(self):
        X, Y, _ = make_regime_switching_data(
            n_samples=400, n_features=30, n_relevant=3,
            n_regimes=2, noise_std=0.5, seed=42,
        )
        config = PipelineConfig(
            targets=[TargetConfig(name="returns", task="regression",
                                  model=ModelConfig(cv_folds=3))],
            min_history=200,
            clustering=__import__("jaymd_winnow.config", fromlist=["ClusteringConfig"]).ClusteringConfig(
                max_clusters=15, update_frequency=500,
            ),
            screening=__import__("jaymd_winnow.config", fromlist=["ScreeningConfig"]).ScreeningConfig(
                threshold=0.5, n_bootstraps=30, cv_folds=3, update_frequency=500,
            ),
        )
        p = AdaptivePipeline(config)

        results = []
        for i in range(len(X)):
            targets = Y[i:i+1] if i > 0 else None
            result = p.step(i, X[i], targets=targets)
            results.append(result)

        # Should have exited warmup
        assert p.is_warm
        # Should have predictions after warmup
        post_warmup = [r for r in results if not r.is_warmup]
        assert len(post_warmup) > 0
        # Predictions should have the right target name
        for r in post_warmup:
            if r.predictions:
                assert "returns" in r.predictions


class TestSaveLoad:
    def test_save_load_invariant(self):
        X, Y, _ = make_regime_switching_data(
            n_samples=300, n_features=20, n_relevant=3,
            n_regimes=2, noise_std=0.5, seed=42,
        )

        from jaymd_winnow.config import ClusteringConfig, ScreeningConfig
        config = PipelineConfig(
            targets=[TargetConfig(name="target", task="regression",
                                  model=ModelConfig(cv_folds=3))],
            min_history=150,
            clustering=ClusteringConfig(max_clusters=10, update_frequency=500),
            screening=ScreeningConfig(threshold=0.5, n_bootstraps=30, cv_folds=3, update_frequency=500),
        )

        p = AdaptivePipeline(config)
        for i in range(250):
            targets = Y[i:i+1] if i > 0 else None
            p.step(i, X[i], targets=targets)

        with tempfile.NamedTemporaryFile(suffix=".joblib", delete=False) as f:
            path = f.name
            p.save(path)

        p2 = AdaptivePipeline.load(path)
        assert p2.is_warm == p.is_warm

        # Continue stepping — should not raise
        result = p2.step(250, X[250], targets=Y[249:250])
        assert result is not None
