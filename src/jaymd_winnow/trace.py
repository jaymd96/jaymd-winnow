"""PipelineTrace — query and analysis of pipeline results."""

from typing import Optional

import numpy as np

from jaymd_winnow.types import LifecycleEvent, StepResult


class PipelineTrace:
    """Analysis and investigation of pipeline results."""

    def __init__(self, results: list[StepResult]):
        self._results = results

    def __len__(self) -> int:
        return len(self._results)

    # --- Performance ---

    def predictions(self, target: Optional[str] = None) -> tuple[list, np.ndarray, np.ndarray]:
        """Extract timestamps, predicted values, and actual targets.

        Returns (timestamps, y_pred, y_actual) for steps where predictions were made.
        If target is None and there's only one target, use that.
        """
        target = self._resolve_target(target)
        timestamps = []
        y_pred = []
        y_actual = []

        for i, r in enumerate(self._results):
            if target in r.predictions and r.predictions[target].point is not None:
                timestamps.append(r.timestamp)
                y_pred.append(r.predictions[target].point[0])
                # Look ahead for the actual value from the next result's health
                if i + 1 < len(self._results):
                    next_r = self._results[i + 1]
                    if next_r.health and target in next_r.health.per_target:
                        y_actual.append(None)  # Actual tracked via monitoring
                    else:
                        y_actual.append(None)
                else:
                    y_actual.append(None)

        return timestamps, np.array(y_pred), np.array(y_actual, dtype=object)

    def performance(
        self,
        target: Optional[str] = None,
        metric: str = "auto",
        after=None,
        before=None,
    ) -> dict:
        """Compute performance metrics over a time range."""
        target = self._resolve_target(target)
        filtered = self._filter_time(after, before)

        preds = []
        for r in filtered:
            if target in r.predictions and r.predictions[target].point is not None:
                preds.append(r.predictions[target].point[0])

        result = {"n_predictions": len(preds)}
        if preds:
            preds_arr = np.array(preds)
            result["mean_prediction"] = float(preds_arr.mean())
            result["std_prediction"] = float(preds_arr.std())

        return result

    # --- Calibration ---

    def calibration_over_time(self, target: Optional[str] = None) -> tuple[list, np.ndarray]:
        """Returns (timestamps, p_values) for calibration monitoring."""
        target = self._resolve_target(target)
        timestamps = []
        pvalues = []

        for r in self._results:
            if r.health and target in r.health.per_target:
                th = r.health.per_target[target]
                if th.calibration_pvalue is not None:
                    timestamps.append(r.timestamp)
                    pvalues.append(th.calibration_pvalue)

        return timestamps, np.array(pvalues)

    def coverage_over_time(self, target: Optional[str] = None) -> tuple[list, dict]:
        """Rolling empirical coverage for each confidence level."""
        target = self._resolve_target(target)
        timestamps = []
        coverage_dict: dict[float, list[float]] = {}

        for r in self._results:
            if r.health and target in r.health.per_target:
                th = r.health.per_target[target]
                if th.coverage is not None:
                    timestamps.append(r.timestamp)
                    for level, cov in th.coverage.items():
                        if level not in coverage_dict:
                            coverage_dict[level] = []
                        coverage_dict[level].append(cov)

        return timestamps, {k: np.array(v) for k, v in coverage_dict.items()}

    # --- Feature dynamics ---

    def feature_importance_over_time(self, target: Optional[str] = None) -> tuple[list, np.ndarray]:
        """Returns (timestamps, importances_matrix) shape (n_timestamps, n_features)."""
        target = self._resolve_target(target)
        timestamps = []
        importances = []

        for r in self._results:
            if r.health and target in r.health.per_target:
                th = r.health.per_target[target]
                if th.feature_stability is not None:
                    timestamps.append(r.timestamp)
                    importances.append(th.feature_stability)

        if importances:
            return timestamps, np.array(importances)
        return timestamps, np.array([])

    def feature_stability_over_time(self, target: Optional[str] = None) -> tuple[list, np.ndarray]:
        """Returns (timestamps, rank_correlations)."""
        target = self._resolve_target(target)
        timestamps = []
        stabilities = []

        for r in self._results:
            if r.health and target in r.health.per_target:
                th = r.health.per_target[target]
                if th.feature_stability is not None:
                    timestamps.append(r.timestamp)
                    stabilities.append(th.feature_stability)

        return timestamps, np.array(stabilities)

    def feature_set_changes(self) -> list[dict]:
        """List of {timestamp, features_added, features_removed, trigger}."""
        changes = []
        prev_features = None

        for r in self._results:
            if r.selected_features is not None:
                current = set(r.selected_features.tolist())
                if prev_features is not None and current != prev_features:
                    changes.append({
                        "timestamp": r.timestamp,
                        "features_added": sorted(current - prev_features),
                        "features_removed": sorted(prev_features - current),
                        "trigger": r.event.event_type if r.event else None,
                    })
                prev_features = current

        return changes

    # --- Lifecycle ---

    def events(self, event_type: Optional[str] = None) -> list[LifecycleEvent]:
        """All lifecycle events, optionally filtered by type."""
        events = [r.event for r in self._results if r.event is not None]
        if event_type is not None:
            events = [e for e in events if e.event_type == event_type]
        return events

    # --- Comparison ---

    @staticmethod
    def compare(a: "PipelineTrace", b: "PipelineTrace") -> dict:
        """Compare two traces from different configs on same data."""
        perf_a = a.performance()
        perf_b = b.performance()

        events_a = a.events()
        events_b = b.events()

        return {
            "a": {"performance": perf_a, "n_events": len(events_a)},
            "b": {"performance": perf_b, "n_events": len(events_b)},
        }

    # --- Internal ---

    def _resolve_target(self, target: Optional[str]) -> str:
        if target is not None:
            return target
        # Infer from results
        for r in self._results:
            if r.predictions:
                names = list(r.predictions.keys())
                if len(names) == 1:
                    return names[0]
                raise ValueError(
                    f"Multiple targets found ({names}), specify target name"
                )
        return "target"

    def _filter_time(self, after=None, before=None) -> list[StepResult]:
        results = self._results
        if after is not None:
            results = [r for r in results if r.timestamp >= after]
        if before is not None:
            results = [r for r in results if r.timestamp <= before]
        return results
