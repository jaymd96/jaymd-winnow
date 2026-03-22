# jaymd-winnow

Adaptive model lifecycle for financial signal selection and prediction.

## Installation

```bash
pip install jaymd-winnow
```

## Quick Start

```python
import numpy as np
from jaymd_winnow import AdaptivePipeline

# Single-target regression
pipeline = AdaptivePipeline.regression(min_history=252)

# Feed data step by step
for i, (timestamp, features, target) in enumerate(your_data_stream):
    result = pipeline.step(
        timestamp=timestamp,
        features=features,
        targets=target if i > 0 else None,  # target for PREVIOUS prediction
    )

    if not result.is_warmup:
        pred = result.predictions["target"]
        print(f"Point: {pred.point}, Intervals: {pred.intervals}")
```

## Features

- **Automatic feature clustering** — reduces thousands of correlated features to independent representatives
- **Stability selection** — identifies features with statistically reliable target association
- **Conformal prediction** — calibrated prediction intervals (regression) and prediction sets (classification)
- **Health monitoring** — PIT uniformity, Brier reliability, ECE, SHAP importance stability
- **Adaptive retraining** — triggers model updates when calibration degrades or feature structure shifts
- **Regime detection** — uses change-point detection to select relevant training windows
- **Multi-target support** — shared feature selection, independent models per target

## Model Types

- `elastic_net` — ElasticNetCV (regression) / LogisticRegressionCV (classification)
- `lightgbm` — LightGBM gradient boosted trees
- `ensemble` — VotingRegressor/VotingClassifier combining both

## API

### Constructors

```python
# Single-target
pipeline = AdaptivePipeline.regression(min_history=252)
pipeline = AdaptivePipeline.classification(min_history=500)

# Multi-target
pipeline = AdaptivePipeline.multi_target(
    targets=[
        {"name": "returns", "task": "regression"},
        {"name": "direction", "task": "binary", "model_type": "lightgbm"},
    ]
)

# Full control
from jaymd_winnow import PipelineConfig, TargetConfig, ModelConfig
config = PipelineConfig(
    targets=[TargetConfig(name="target", task="regression", model=ModelConfig(model_type="ensemble"))],
    min_history=252,
)
pipeline = AdaptivePipeline(config)
```

### Step Loop

```python
result = pipeline.step(timestamp, features, targets)
# result.predictions: dict[str, TargetPrediction]
# result.health: HealthSnapshot
# result.event: LifecycleEvent (retraining, etc.)
# result.is_warmup: bool
```

### Serialisation

```python
pipeline.save("checkpoint.joblib")
pipeline = AdaptivePipeline.load("checkpoint.joblib")
```

### Trace Analysis

```python
from jaymd_winnow import PipelineTrace

trace = PipelineTrace(results)
timestamps, pvalues = trace.calibration_over_time()
events = trace.events("retrain_refit")
changes = trace.feature_set_changes()
```

## License

MIT
