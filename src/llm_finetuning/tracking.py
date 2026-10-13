"""Small MLflow adapter for evaluation and benchmark artifacts."""

from __future__ import annotations

from pathlib import Path
from typing import Any


def log_result(
    config: dict[str, Any],
    run_name: str,
    params: dict[str, Any],
    metrics: dict[str, Any],
    artifact_path: str | Path,
) -> None:
    import mlflow

    mlflow.set_tracking_uri(config["project"]["mlflow_tracking_uri"])
    mlflow.set_experiment(config["project"]["experiment_name"])
    with mlflow.start_run(run_name=run_name):
        mlflow.log_params({key: str(value) for key, value in params.items() if value is not None})
        for key, value in metrics.items():
            if isinstance(value, (int, float)) and value is not None:
                mlflow.log_metric(key, float(value))
        mlflow.log_artifact(str(artifact_path))
