"""jaymd-winnow: Adaptive model lifecycle for financial signal selection and prediction."""

__version__ = "0.1.1"

from jaymd_winnow.pipeline import AdaptivePipeline
from jaymd_winnow.trace import PipelineTrace
from jaymd_winnow.config import (
    PipelineConfig,
    TargetConfig,
    ModelConfig,
    ClusteringConfig,
    ScreeningConfig,
    RegimeConfig,
    MonitorConfig,
    RetrainingConfig,
)
from jaymd_winnow.stability_selection import stability_selection
from jaymd_winnow.types import (
    StepResult,
    TargetPrediction,
    HealthSnapshot,
    TargetHealth,
    LifecycleEvent,
)

__all__ = [
    "AdaptivePipeline",
    "PipelineTrace",
    "PipelineConfig",
    "TargetConfig",
    "ModelConfig",
    "ClusteringConfig",
    "ScreeningConfig",
    "RegimeConfig",
    "MonitorConfig",
    "RetrainingConfig",
    "StepResult",
    "TargetPrediction",
    "HealthSnapshot",
    "TargetHealth",
    "LifecycleEvent",
    "stability_selection",
]
