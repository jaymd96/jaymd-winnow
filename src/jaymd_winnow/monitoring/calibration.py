"""Calibration monitoring: PIT uniformity, Brier reliability, ECE."""

import numpy as np
from scipy.stats import kstest


def compute_pit_value(
    y_actual: float,
    conformity_scores: np.ndarray,
    y_pred: float,
) -> float:
    """Compute PIT value for one observation.

    PIT = fraction of calibration residuals <= the observed residual.
    Under correct model, PIT ~ Uniform(0,1).
    """
    observed_residual = np.abs(y_actual - y_pred)
    return float(np.mean(conformity_scores <= observed_residual))


def check_pit_uniformity(
    pit_values: np.ndarray,
) -> tuple[float, float]:
    """Test whether PIT values are uniform.

    Returns:
        (ks_statistic, p_value). Low p_value => model is miscalibrated.
    """
    stat, pvalue = kstest(pit_values, "uniform")
    return float(stat), float(pvalue)


def compute_brier_reliability(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10,
) -> float:
    """Reliability component of the Brier score decomposition.

    Low reliability = well-calibrated. High = miscalibrated.
    """
    bin_edges = np.linspace(0, 1, n_bins + 1)
    reliability = 0.0
    n = len(y_true)
    for i in range(n_bins):
        mask = (y_prob >= bin_edges[i]) & (y_prob < bin_edges[i + 1])
        if mask.sum() == 0:
            continue
        bin_mean_pred = y_prob[mask].mean()
        bin_mean_true = y_true[mask].mean()
        reliability += mask.sum() * (bin_mean_pred - bin_mean_true) ** 2
    return float(reliability / n)


def compute_ece(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10,
) -> float:
    """Expected Calibration Error.

    Weighted average of per-bin |accuracy - confidence|.
    """
    confidences = y_prob.max(axis=1)
    predictions = y_prob.argmax(axis=1)
    accuracies = (predictions == y_true).astype(float)

    bin_edges = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    n = len(y_true)
    for i in range(n_bins):
        mask = (confidences >= bin_edges[i]) & (confidences < bin_edges[i + 1])
        if mask.sum() == 0:
            continue
        bin_accuracy = accuracies[mask].mean()
        bin_confidence = confidences[mask].mean()
        ece += mask.sum() * np.abs(bin_accuracy - bin_confidence)
    return float(ece / n)
