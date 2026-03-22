"""Core data types for pipeline results and health monitoring."""

from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np


@dataclass
class TargetHealth:
    """Health metrics for a single target."""
    target_name: str
    calibration_pvalue: Optional[float] = None
    brier_reliability: Optional[float] = None
    ece: Optional[float] = None
    feature_stability: Optional[float] = None
    coverage: Optional[dict[float, float]] = None


@dataclass
class HealthSnapshot:
    """Aggregate health across all targets."""
    per_target: dict[str, TargetHealth] = field(default_factory=dict)

    @property
    def worst_calibration(self) -> Optional[float]:
        """Lowest calibration p-value across all targets. None if no data yet."""
        pvals = [
            t.calibration_pvalue for t in self.per_target.values()
            if t.calibration_pvalue is not None
        ]
        return min(pvals) if pvals else None

    @property
    def worst_stability(self) -> Optional[float]:
        """Lowest feature stability across all targets."""
        stabs = [
            t.feature_stability for t in self.per_target.values()
            if t.feature_stability is not None
        ]
        return min(stabs) if stabs else None


@dataclass
class LifecycleEvent:
    event_type: str
    timestamp: Any
    reason: str
    details: dict[str, Any] = field(default_factory=dict)


@dataclass
class TargetPrediction:
    """Prediction for a single target."""
    target_name: str
    point: Optional[np.ndarray] = None
    intervals: Optional[np.ndarray] = None
    prediction_sets: Optional[np.ndarray] = None


@dataclass
class StepResult:
    timestamp: Any
    predictions: dict[str, TargetPrediction] = field(default_factory=dict)
    selected_features: Optional[np.ndarray] = None
    health: Optional[HealthSnapshot] = None
    event: Optional[LifecycleEvent] = None
    is_warmup: bool = True
