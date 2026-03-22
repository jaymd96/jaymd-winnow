"""Tests for the stability selection module."""

import numpy as np

from jaymd_winnow.stability_selection import stability_selection


def test_recovers_known_features():
    """Strong signal features should have high stability scores."""
    rng = np.random.RandomState(42)
    n = 300
    X = rng.randn(n, 20)
    y = 3.0 * X[:, 0] + 2.0 * X[:, 1] + 1.5 * X[:, 2] + rng.randn(n) * 0.1

    selected, scores = stability_selection(X, y, task="regression", n_bootstraps=30, threshold=0.3)

    assert 0 in selected
    assert 1 in selected
    assert 2 in selected


def test_returns_scores():
    rng = np.random.RandomState(42)
    X = rng.randn(100, 10)
    y = X[:, 0] + rng.randn(100) * 0.5

    selected, scores = stability_selection(X, y, task="regression", n_bootstraps=20)

    assert scores.shape == (10,)
    assert all(0 <= s <= 1 for s in scores)


def test_binary_task():
    rng = np.random.RandomState(42)
    X = rng.randn(200, 10)
    y = (X[:, 0] > 0).astype(float)

    selected, scores = stability_selection(X, y, task="binary", n_bootstraps=20, threshold=0.3)

    assert len(selected) > 0
    assert 0 in selected
