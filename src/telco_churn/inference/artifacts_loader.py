import json
from typing import Any
from joblib import load
from sklearn.pipeline import Pipeline

from configs.telco_churn_config import Config

def load_classification_artifacts(config: Config) -> dict:
    artifact_dir = config.artifacts_dir / "classification"

    with open(artifact_dir / "best_model.json") as f:
        metadata = json.load(f)

    model_name = metadata["best_model_name"]
    pipeline = load(artifact_dir / f"{model_name}_pipeline.joblib")
    calibrator_path = artifact_dir / f"{model_name}_calibrator.joblib"
    calibrator = (
        load(calibrator_path)
        if calibrator_path.exists()
        else None
    )

    threshold = metadata["threshold"]
    return {"pipeline": pipeline, "calibrator": calibrator, "threshold": threshold}


def load_regression_artifacts(config: Config) -> dict:
    artifact_dir = config.artifacts_dir / "value_regression"

    with open(artifact_dir / "best_model.json") as f:
        metadata = json.load(f)

    model_name = metadata["best_model_name"]
    pipeline = load(artifact_dir / f"{model_name}_pipeline.joblib")
    return {"pipeline": pipeline}

