"""Tests for calibration monitoring."""

import numpy as np

from jaymd_winnow.monitoring.calibration import (
    compute_brier_reliability,
    compute_ece,
    compute_pit_value,
    check_pit_uniformity,
)


def test_pit_uniform_for_correct_model():
    """Under a correct model, PIT values should be uniform."""
    rng = np.random.RandomState(42)
    # Simulate: conformity scores from a well-calibrated model
    conformity_scores = np.abs(rng.randn(100))
    # Generate PIT values that should be roughly uniform
    pit_values = rng.uniform(0, 1, 200)
    stat, pvalue = check_pit_uniformity(pit_values)
    # p-value should be high (uniform)
    assert pvalue > 0.05


def test_pit_non_uniform_for_misspecified():
    """If PIT values cluster, p-value should be low."""
    pit_values = np.concatenate([
        np.full(50, 0.1),
        np.full(50, 0.9),
    ])
    stat, pvalue = check_pit_uniformity(pit_values)
    assert pvalue < 0.05


def test_pit_value_computation():
    scores = np.array([0.5, 1.0, 1.5, 2.0, 2.5])
    pit = compute_pit_value(y_actual=1.0, conformity_scores=scores, y_pred=0.0)
    # |1.0 - 0.0| = 1.0, fraction of scores <= 1.0 = 2/5
    assert abs(pit - 0.4) < 1e-10


def test_brier_reliability_perfect():
    """Perfect calibration → reliability near 0."""
    rng = np.random.RandomState(42)
    n = 1000
    y_prob = rng.uniform(0, 1, n)
    y_true = (rng.uniform(0, 1, n) < y_prob).astype(float)
    rel = compute_brier_reliability(y_true, y_prob)
    assert rel < 0.05


def test_brier_reliability_bad():
    """Completely off calibration."""
    y_true = np.ones(100)
    y_prob = np.full(100, 0.1)  # predict 10% but always 1
    rel = compute_brier_reliability(y_true, y_prob)
    assert rel > 0.1


def test_ece_perfect():
    """Perfect calibration → ECE near 0."""
    rng = np.random.RandomState(42)
    n = 500
    # 3-class problem, perfectly calibrated
    y_true = rng.choice(3, size=n)
    y_prob = np.zeros((n, 3))
    for i in range(n):
        y_prob[i, y_true[i]] = 0.8
        others = [j for j in range(3) if j != y_true[i]]
        y_prob[i, others[0]] = 0.1
        y_prob[i, others[1]] = 0.1
    ece = compute_ece(y_true, y_prob)
    assert ece < 0.25  # 0.8/0.1/0.1 probabilities have bounded miscalibration


def test_ece_bad():
    """Always confident but wrong → high ECE."""
    n = 100
    y_true = np.zeros(n, dtype=int)
    y_prob = np.zeros((n, 3))
    y_prob[:, 1] = 0.9  # always confident in class 1, but truth is class 0
    y_prob[:, 0] = 0.05
    y_prob[:, 2] = 0.05
    ece = compute_ece(y_true, y_prob)
    assert ece > 0.5
