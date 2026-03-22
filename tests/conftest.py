"""Shared fixtures and synthetic data generation for tests."""

import numpy as np
import pytest


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
    """Generate regime-switching synthetic data.

    Only n_relevant features are truly predictive.
    The relevant features and coefficients change between regimes.
    """
    rng = np.random.RandomState(seed)
    X = rng.randn(n_samples, n_features)

    # Add correlation structure: groups of 5 features are highly correlated
    for g in range(0, n_features - 4, 5):
        base = X[:, g].copy()
        for j in range(1, min(5, n_features - g)):
            X[:, g + j] = base + rng.randn(n_samples) * 0.3

    regime_size = n_samples // n_regimes
    regime_boundaries = [i * regime_size for i in range(1, n_regimes)]

    Y = np.zeros((n_samples, n_targets))
    metadata = {
        "regime_boundaries": regime_boundaries,
        "relevant_features": [],
        "coefficients": [],
    }

    for regime in range(n_regimes):
        start = 0 if regime == 0 else regime_boundaries[regime - 1]
        end = regime_boundaries[regime] if regime < n_regimes - 1 else n_samples
        seg = slice(start, end)

        # Different relevant features per regime
        relevant = rng.choice(n_features, size=n_relevant, replace=False)
        coeffs = rng.randn(n_relevant) * 2.0
        metadata["relevant_features"].append(relevant.tolist())
        metadata["coefficients"].append(coeffs.tolist())

        signal = X[seg][:, relevant] @ coeffs

        for t in range(n_targets):
            task = target_tasks[t] if t < len(target_tasks) else target_tasks[-1]
            if task == "regression":
                Y[seg, t] = signal + rng.randn(end - start) * noise_std
            elif task == "binary":
                prob = 1.0 / (1.0 + np.exp(-signal))
                Y[seg, t] = (rng.rand(end - start) < prob).astype(float)
            elif task == "multiclass":
                # 3-class problem
                logits = np.column_stack([
                    signal,
                    signal * 0.5 + rng.randn(end - start) * 0.5,
                    -signal,
                ])
                exp_logits = np.exp(logits - logits.max(axis=1, keepdims=True))
                probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
                Y[seg, t] = np.array([
                    rng.choice(3, p=probs[i]) for i in range(end - start)
                ]).astype(float)

    return X, Y, metadata


@pytest.fixture
def synthetic_regression_data():
    """Simple regression data with 100 features, 5 relevant, 3 regimes."""
    X, Y, meta = make_regime_switching_data(
        n_samples=500, n_features=50, n_relevant=5,
        n_regimes=2, noise_std=0.5, seed=42,
    )
    return X, Y[:, 0], meta


@pytest.fixture
def synthetic_binary_data():
    """Binary classification data."""
    X, Y, meta = make_regime_switching_data(
        n_samples=500, n_features=50, n_relevant=5,
        n_regimes=2, target_tasks=("binary",), seed=42,
    )
    return X, Y[:, 0], meta


@pytest.fixture
def synthetic_multi_target_data():
    """Multi-target data: one regression, one binary."""
    X, Y, meta = make_regime_switching_data(
        n_samples=500, n_features=50, n_relevant=5,
        n_regimes=2, n_targets=2,
        target_tasks=("regression", "binary"), seed=42,
    )
    return X, Y, meta
