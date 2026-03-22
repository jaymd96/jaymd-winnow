# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

**jaymd-winnow** (`import jaymd_winnow`) — adaptive model lifecycle for financial signal selection and prediction. Orchestrates feature clustering, stability selection, conformal prediction, health monitoring, and adaptive retraining. All algorithms are delegated to mature packages (sklearn, lightgbm, mapie, shap, ruptures); this package writes orchestration code only.

## Commands

```bash
# Install (editable, with test deps)
pip install -e ".[dev]"

# Run all tests
pytest tests/

# Run a single test file or class
pytest tests/test_modelling.py -v
pytest tests/test_end_to_end.py::TestRegressionEndToEnd -v

# Build and publish
python -m build
twine upload dist/*
```

Test suite is ~90 tests, takes ~90s. The `test_end_to_end.py` tests are the slowest (~60s) because they run full pipeline lifecycles.

## Architecture

### Data flow through the pipeline

```
step(timestamp, features, targets)
  │
  ├─ Phase 1: cluster_features() → reduce correlated features to ~200 representatives
  ├─ Phase 2: screen_features() → stability selection picks reliably predictive features
  ├─ Phase 3: build_base_model() → fit model + conformalise_model() → MAPIE wrapper
  │
  ├─ Monitoring: PIT/Brier/ECE calibration checks + SHAP importance stability
  ├─ Triggers: evaluate_trigger() → "refit" / "reselect" / "rebuild" / None
  │
  └─ Returns StepResult with per-target predictions, intervals, health, lifecycle events
```

Phases 1-2 are **shared** across all targets (expensive, run infrequently). Phase 3 is **per-target** (each target gets its own model, conformal wrapper, monitoring state).

### Key modules

- **`pipeline.py`** — `AdaptivePipeline` state machine. The `step()` method is the heart; `_PipelineState` holds all mutable state (buffers, models, counters).
- **`config.py`** — all frozen dataclasses. Config is immutable after construction.
- **`types.py`** — result dataclasses (`StepResult`, `HealthSnapshot`, `TargetPrediction`, etc.).
- **`phases/modelling.py`** — model factory dispatching to `_build_linear`/`_build_lightgbm`/`_build_ensemble`, plus MAPIE conformal wrapping and ruptures regime detection.
- **`stability_selection.py`** — self-contained bootstrap stability selection (replaces broken upstream `stability-selection` package).
- **`cache.py`** — thin `joblib.Memory` wrapper. Clustering, screening, model building, and SHAP are cached.

### Design patterns

- **Frozen config** — `@dataclass(frozen=True)` everywhere. No mutation after construction.
- **Composition over implementation** — every algorithm delegates to sklearn/lightgbm/mapie/shap/ruptures.
- **Counter-based state machine** — `steps_since_last_*` counters trigger phase re-execution at configured frequencies.
- **Temporal data splitting** — training windows split as `[train | conformalize | remainder]`; conformal calibration data is always the most recent slice before the test point.
- **Cache keys are implicit** — `joblib.Memory` hashes function args. Different training data or config → cache miss. Same data + config → cache hit.

## Testing conventions

- Synthetic data via `conftest.make_regime_switching_data()` — regime-switching with configurable features, targets, tasks.
- Use `_fast_config()` from `test_end_to_end.py` as a template for integration tests (small bootstraps, few clusters, fast convergence).
- Parametrize across `model_type` (elastic_net, lightgbm, ensemble) and `task` (regression, binary, multiclass) for model factory tests.
- End-to-end tests validate: predictions exist and are finite, intervals have correct shape, lifecycle events fire, models exist after warmup.

## Adding a new feature

- **New config field** → add to the appropriate frozen dataclass in `config.py`.
- **New phase behavior** → modify the relevant `phases/*.py` function. The function signature of `build_base_model` is a cache key boundary — adding parameters to `ModelConfig` propagates automatically without changing signatures.
- **New monitoring metric** → add to `monitoring/calibration.py` or `monitoring/stability.py`, wire into `_update_monitors` in `pipeline.py`, add field to `TargetHealth` in `types.py`.
- **New trigger logic** → modify `evaluate_trigger()` in `monitoring/triggers.py`.

Avoid function names starting with `test_` in source code — pytest collects them as tests.
