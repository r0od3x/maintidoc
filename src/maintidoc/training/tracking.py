"""MLflow experiment tracking.

Logs to a local `mlruns/` directory by default, so nothing needs a server.
`MLFLOW_TRACKING_URI` points it elsewhere when there is one.

Tracking is optional at runtime but never silently skipped: if MLflow is
absent or fails, `NullTracker` records the fact and the caller can say the run
was not tracked. An untracked experiment that looks tracked is worse than an
obviously untracked one.
"""

from __future__ import annotations

import os
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

EXPERIMENT = "maintidoc-reranker"

# MLflow accepts only alphanumerics, `_ - . / ` in metric names, so the
# conventional `recall@5` has to be rewritten rather than silently dropped.
METRIC_SUBSTITUTIONS = {"@": "_at_", "%": "_pct", ":": "_"}


def sanitise_metric_name(name: str) -> str:
    cleaned = name
    for bad, good in METRIC_SUBSTITUTIONS.items():
        cleaned = cleaned.replace(bad, good)
    return cleaned


@runtime_checkable
class Tracker(Protocol):
    enabled: bool

    def log_params(self, params: dict[str, Any]) -> None: ...

    def log_metrics(self, metrics: dict[str, float]) -> None: ...

    def log_artifact(self, path: Path) -> None: ...


@dataclass
class NullTracker:
    """Records nothing, and says so."""

    enabled: bool = False
    reason: str = "MLflow non configure"
    params: dict[str, Any] = field(default_factory=dict)
    metrics: dict[str, float] = field(default_factory=dict)

    def log_params(self, params: dict[str, Any]) -> None:
        self.params.update(params)

    def log_metrics(self, metrics: dict[str, float]) -> None:
        self.metrics.update(metrics)

    def log_artifact(self, path: Path) -> None:
        return None


@dataclass
class MlflowTracker:
    run_name: str
    experiment: str = EXPERIMENT
    enabled: bool = True

    def __post_init__(self) -> None:
        import mlflow

        self._mlflow = mlflow
        uri = os.environ.get("MLFLOW_TRACKING_URI")
        if uri:
            mlflow.set_tracking_uri(uri)
        mlflow.set_experiment(self.experiment)

    def log_params(self, params: dict[str, Any]) -> None:
        self._mlflow.log_params({k: str(v) for k, v in params.items()})

    def log_metrics(self, metrics: dict[str, float]) -> None:
        usable = {
            sanitise_metric_name(key): float(value)
            for key, value in metrics.items()
            if isinstance(value, (int, float))
        }
        if usable:
            self._mlflow.log_metrics(usable)

    def log_artifact(self, path: Path) -> None:
        if path.is_file():
            self._mlflow.log_artifact(str(path))


@contextmanager
def tracking_run(run_name: str, *, enabled: bool = True, experiment: str = EXPERIMENT):
    """Yield a tracker, degrading to `NullTracker` rather than failing a run.

    A training or evaluation run must not be lost because the experiment
    tracker was unreachable — but the caller is told, through `enabled`.
    """
    if not enabled:
        yield NullTracker(reason="suivi desactive")
        return
    try:
        import mlflow
    except ImportError:
        yield NullTracker(reason="mlflow non installe")
        return

    try:
        tracker = MlflowTracker(run_name, experiment=experiment)
    except Exception as exc:  # pragma: no cover - environment dependent
        yield NullTracker(reason=f"mlflow indisponible : {exc}")
        return

    with mlflow.start_run(run_name=run_name):
        yield tracker
