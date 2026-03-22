"""Tests for Phase 1: feature clustering."""

import numpy as np
import pytest

from jaymd_winnow.config import ClusteringConfig
from jaymd_winnow.phases.clustering import cluster_features


def test_correct_cluster_count():
    rng = np.random.RandomState(42)
    X = rng.randn(200, 50)
    config = ClusteringConfig(max_clusters=10)
    labels, reps = cluster_features(X, config)

    assert len(reps) == 10
    assert labels.shape == (50,)
    # Each representative is a valid column index
    assert all(0 <= r < 50 for r in reps)


def test_skip_when_fewer_features():
    rng = np.random.RandomState(42)
    X = rng.randn(100, 5)
    config = ClusteringConfig(max_clusters=10)
    labels, reps = cluster_features(X, config)

    # Should return all features since n_features < max_clusters
    assert len(reps) == 5


def test_zero_variance_handling():
    rng = np.random.RandomState(42)
    X = rng.randn(100, 20)
    X[:, 5] = 0.0  # zero variance
    X[:, 10] = 3.0  # constant = zero variance

    config = ClusteringConfig(max_clusters=10)
    labels, reps = cluster_features(X, config)

    # Zero-variance features should not be representatives
    assert 5 not in reps
    assert 10 not in reps


def test_all_zero_variance_raises():
    X = np.zeros((100, 10))
    config = ClusteringConfig(max_clusters=5)
    with pytest.raises(ValueError, match="zero variance"):
        cluster_features(X, config)


def test_determinism():
    rng = np.random.RandomState(42)
    X = rng.randn(200, 50)
    config = ClusteringConfig(max_clusters=10)

    labels1, reps1 = cluster_features(X, config)
    labels2, reps2 = cluster_features(X, config)

    np.testing.assert_array_equal(labels1, labels2)
    np.testing.assert_array_equal(reps1, reps2)
