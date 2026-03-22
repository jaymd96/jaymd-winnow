"""Tests for retraining trigger logic."""

import pytest

from jaymd_winnow.config import RetrainingConfig
from jaymd_winnow.monitoring.triggers import evaluate_trigger
from jaymd_winnow.types import HealthSnapshot, TargetHealth


@pytest.fixture
def config():
    return RetrainingConfig(
        calibration_trigger=0.05,
        stability_trigger=0.3,
        cooldown_steps=20,
        min_regime_size=60,
    )


def test_cooldown_respected(config):
    health = HealthSnapshot(per_target={
        "t": TargetHealth(target_name="t", calibration_pvalue=0.001),
    })
    result = evaluate_trigger(health, config, steps_since_last_retrain=5)
    assert result is None


def test_no_data_yet(config):
    health = HealthSnapshot()
    result = evaluate_trigger(health, config, steps_since_last_retrain=100)
    assert result is None


def test_all_calibrated(config):
    health = HealthSnapshot(per_target={
        "t": TargetHealth(target_name="t", calibration_pvalue=0.5),
    })
    result = evaluate_trigger(health, config, steps_since_last_retrain=100)
    assert result is None


def test_refit_on_calibration_failure(config):
    health = HealthSnapshot(per_target={
        "t": TargetHealth(target_name="t", calibration_pvalue=0.01, feature_stability=0.8),
    })
    result = evaluate_trigger(health, config, steps_since_last_retrain=100)
    assert result == "refit"


def test_reselect_on_stability_failure(config):
    health = HealthSnapshot(per_target={
        "t": TargetHealth(target_name="t", calibration_pvalue=0.01, feature_stability=0.1),
    })
    result = evaluate_trigger(health, config, steps_since_last_retrain=100)
    assert result == "reselect"


def test_worst_case_across_targets(config):
    health = HealthSnapshot(per_target={
        "good": TargetHealth(target_name="good", calibration_pvalue=0.5, feature_stability=0.9),
        "bad": TargetHealth(target_name="bad", calibration_pvalue=0.01, feature_stability=0.8),
    })
    result = evaluate_trigger(health, config, steps_since_last_retrain=100)
    assert result == "refit"
