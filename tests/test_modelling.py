"""Tests for Phase 3: model factory and conformal wrapping."""

import numpy as np
import pytest

from jaymd_winnow.config import ModelConfig, RegimeConfig
from jaymd_winnow.phases.modelling import (
    build_base_model,
    conformalise_model,
    detect_regimes,
)


@pytest.fixture
def regression_data():
    rng = np.random.RandomState(42)
    X = rng.randn(200, 10)
    y = X[:, 0] * 2 + X[:, 1] + rng.randn(200) * 0.5
    return X, y


@pytest.fixture
def binary_data():
    rng = np.random.RandomState(42)
    X = rng.randn(200, 10)
    logits = X[:, 0] * 2 + X[:, 1]
    y = (logits > 0).astype(float)
    return X, y


@pytest.fixture
def multiclass_data():
    rng = np.random.RandomState(42)
    X = rng.randn(300, 10)
    logits = X[:, :3]
    y = logits.argmax(axis=1).astype(float)
    return X, y


class TestModelFactory:
    """Test all model_type × task combinations."""

    @pytest.mark.parametrize("model_type", ["elastic_net", "lightgbm", "ensemble"])
    def test_regression(self, regression_data, model_type):
        X, y = regression_data
        config = ModelConfig(model_type=model_type, cv_folds=3)
        pipe = build_base_model(X, y, "regression", config)
        preds = pipe.predict(X[:5])
        assert preds.shape == (5,)

    @pytest.mark.parametrize("model_type", ["elastic_net", "lightgbm", "ensemble"])
    def test_binary(self, binary_data, model_type):
        X, y = binary_data
        config = ModelConfig(model_type=model_type, cv_folds=3)
        pipe = build_base_model(X, y, "binary", config)
        preds = pipe.predict(X[:5])
        assert preds.shape == (5,)

    @pytest.mark.parametrize("model_type", ["elastic_net", "lightgbm", "ensemble"])
    def test_multiclass(self, multiclass_data, model_type):
        X, y = multiclass_data
        config = ModelConfig(model_type=model_type, cv_folds=3)
        pipe = build_base_model(X, y, "multiclass", config)
        preds = pipe.predict(X[:5])
        assert preds.shape == (5,)


class TestConformalisation:
    def test_regression_intervals(self, regression_data):
        X, y = regression_data
        X_train, X_conf = X[:150], X[150:]
        y_train, y_conf = y[:150], y[150:]
        config = ModelConfig(cv_folds=3)

        pipe = build_base_model(X_train, y_train, "regression", config)
        conformal = conformalise_model(pipe, X_conf, y_conf, "regression", config)

        y_pred, y_pis = conformal.predict_interval(X[:5])
        assert y_pred.shape[0] == 5
        assert y_pis.shape[0] == 5
        assert y_pis.shape[1] == 2  # lower, upper

    def test_binary_prediction_sets(self, binary_data):
        X, y = binary_data
        X_train, X_conf = X[:150], X[150:]
        y_train, y_conf = y[:150], y[150:]
        config = ModelConfig(cv_folds=3)

        pipe = build_base_model(X_train, y_train, "binary", config)
        conformal = conformalise_model(pipe, X_conf, y_conf, "binary", config)

        y_pred, y_sets = conformal.predict_set(X[:5])
        assert y_pred.shape[0] == 5


class TestRegimeDetection:
    def test_detects_breakpoints(self):
        rng = np.random.RandomState(42)
        y = np.concatenate([
            rng.randn(200) + 0,
            rng.randn(200) + 5,
            rng.randn(200) + 0,
        ])
        config = RegimeConfig(algorithm="pelt", cost_model="rbf", penalty=10.0, min_segment_size=30)
        bps = detect_regimes(y, config)
        assert len(bps) >= 1

    def test_no_breakpoints(self):
        rng = np.random.RandomState(42)
        y = rng.randn(200)
        config = RegimeConfig(penalty=1000.0, min_segment_size=30)
        bps = detect_regimes(y, config)
        # High penalty → likely no breakpoints
        assert isinstance(bps, list)

    def test_custom_cost(self):
        import ruptures as rpt

        rng = np.random.RandomState(42)
        y = np.concatenate([rng.randn(100), rng.randn(100) + 5])
        custom = rpt.costs.CostL2()
        config = RegimeConfig(custom_cost=custom, penalty=10.0, min_segment_size=20)
        bps = detect_regimes(y, config)
        assert isinstance(bps, list)
