"""Tests for Phase 2: feature screening."""

import numpy as np
import pytest

from jaymd_winnow.config import ScreeningConfig
from jaymd_winnow.phases.screening import screen_features


@pytest.fixture
def clear_signal_data():
    """Data where first 3 features are strongly predictive."""
    rng = np.random.RandomState(42)
    n = 300
    X = rng.randn(n, 20)
    y = 3.0 * X[:, 0] + 2.0 * X[:, 1] + 1.5 * X[:, 2] + rng.randn(n) * 0.1
    return X, y


def test_recovers_known_features(clear_signal_data):
    X, y = clear_signal_data
    config = ScreeningConfig(threshold=0.5, n_bootstraps=50, cv_folds=3)
    selected = screen_features(X, y, "regression", config)

    # The first 3 features should be among the selected
    assert 0 in selected
    assert 1 in selected
    assert 2 in selected


def test_returns_array(clear_signal_data):
    X, y = clear_signal_data
    config = ScreeningConfig(threshold=0.5, n_bootstraps=50, cv_folds=3)
    selected = screen_features(X, y, "regression", config)
    assert isinstance(selected, np.ndarray)
    assert selected.ndim == 1


def test_binary_task(clear_signal_data):
    X, y_raw = clear_signal_data
    y = (y_raw > np.median(y_raw)).astype(float)
    config = ScreeningConfig(threshold=0.5, n_bootstraps=50, cv_folds=3)
    selected = screen_features(X, y, "binary", config)
    assert len(selected) > 0


def test_zero_selection_fallback():
    """With pure noise, stability selection should return 0 features → fallback to top-10."""
    rng = np.random.RandomState(42)
    X = rng.randn(100, 20)
    y = rng.randn(100)
    # Very high threshold → likely zero features
    config = ScreeningConfig(threshold=0.99, n_bootstraps=20, cv_folds=3)
    selected = screen_features(X, y, "regression", config)
    # Should fall back to top-10
    assert len(selected) > 0
    assert len(selected) <= 10
