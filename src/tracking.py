from __future__ import annotations

import os
from pathlib import Path

from src.model import TrainedModels


def log_training_run(models: TrainedModels, tracking_db: str = "mlflow.db") -> None:
    """Persist the latest model metrics to a local SQLite-backed MLflow store."""
    import mlflow

    database = Path(tracking_db).resolve()
    try:
        mlflow.set_tracking_uri(f"sqlite:///{database.as_posix()}")
        mlflow.set_experiment("real-estate-investment-advisor")
    except ImportError:
        # Some locked-down Windows installations block SQLAlchemy's native DLL.
        tracking_dir = database.parent / "mlruns"
        tracking_dir.mkdir(parents=True, exist_ok=True)
        os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"
        mlflow.set_tracking_uri(tracking_dir.resolve().as_uri())
        mlflow.set_experiment("real-estate-investment-advisor")
    with mlflow.start_run():
        mlflow.log_metrics({
            "classification_accuracy": models.classification_accuracy,
            "regression_rmse": models.regression_rmse,
            "regression_r2": models.regression_r2,
        })
