"""Tests for PipelineTrace query layer."""

import numpy as np
import pytest

from jaymd_winnow.trace import PipelineTrace
from jaymd_winnow.types import (
    HealthSnapshot,
    LifecycleEvent,
    StepResult,
    TargetHealth,
    TargetPrediction,
)


def _make_results(n=10):
    results = []
    for i in range(n):
        preds = {}
        if i >= 5:
            preds["target"] = TargetPrediction(
                target_name="target",
                point=np.array([float(i)]),
            )
        health = None
        if i >= 7:
            health = HealthSnapshot(per_target={
                "target": TargetHealth(
                    target_name="target",
                    calibration_pvalue=0.5 - i * 0.05,
                ),
            })
        event = None
        if i == 5:
            event = LifecycleEvent(
                event_type="warmup_complete",
                timestamp=i,
                reason="test",
            )
        results.append(StepResult(
            timestamp=i,
            predictions=preds,
            health=health,
            event=event,
            is_warmup=i < 5,
        ))
    return results


def test_trace_length():
    results = _make_results()
    trace = PipelineTrace(results)
    assert len(trace) == 10


def test_predictions():
    trace = PipelineTrace(_make_results())
    timestamps, y_pred, _ = trace.predictions()
    assert len(timestamps) == 5
    assert y_pred[0] == 5.0


def test_calibration_over_time():
    trace = PipelineTrace(_make_results())
    timestamps, pvalues = trace.calibration_over_time()
    assert len(timestamps) == 3
    assert len(pvalues) == 3


def test_events_filter():
    trace = PipelineTrace(_make_results())
    all_events = trace.events()
    assert len(all_events) == 1
    warmup_events = trace.events("warmup_complete")
    assert len(warmup_events) == 1
    other_events = trace.events("retrain_refit")
    assert len(other_events) == 0


def test_feature_set_changes():
    results = []
    for i in range(5):
        results.append(StepResult(
            timestamp=i,
            selected_features=np.array([0, 1, 2]) if i < 3 else np.array([1, 2, 3]),
        ))
    trace = PipelineTrace(results)
    changes = trace.feature_set_changes()
    assert len(changes) == 1
    assert 3 in changes[0]["features_added"]
    assert 0 in changes[0]["features_removed"]


def test_compare():
    a = PipelineTrace(_make_results(10))
    b = PipelineTrace(_make_results(8))
    comparison = PipelineTrace.compare(a, b)
    assert "a" in comparison
    assert "b" in comparison
