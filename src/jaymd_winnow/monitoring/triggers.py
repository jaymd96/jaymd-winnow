"""Retraining trigger logic based on health diagnostics."""

from typing import Optional

from jaymd_winnow.config import RetrainingConfig
from jaymd_winnow.types import HealthSnapshot


def evaluate_trigger(
    health: HealthSnapshot,
    config: RetrainingConfig,
    steps_since_last_retrain: int,
) -> Optional[str]:
    """Evaluate whether retraining is needed based on health metrics.

    Returns:
        None: no action needed.
        "refit": rerun Phase 3 only (coefficients drifted).
        "reselect": rerun Phase 2 + 3 (feature-target structure changed).
        "rebuild": rerun Phase 1 + 2 + 3 (correlation structure changed).
    """
    if steps_since_last_retrain < config.cooldown_steps:
        return None

    worst_cal = health.worst_calibration
    worst_stab = health.worst_stability

    # Not enough data yet
    if worst_cal is None:
        return None

    # All targets calibrated — no action
    if worst_cal > config.calibration_trigger:
        return None

    # Calibration failed for at least one target — diagnose
    if worst_stab is not None and worst_stab < config.stability_trigger:
        return "reselect"

    return "refit"
