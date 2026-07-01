import json
from pathlib import Path
from typing import Optional

import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from configs.telco_churn_config import Config
from src.telco_churn.postprocessing.calibration import ProbabilityCalibrator
from src.telco_churn.postprocessing.threshold import ThresholdTuner
from src.telco_churn.model_factory import ModelFactory
from src.telco_churn.preprocessor import Preprocessor
from src.telco_churn.training.trainer import ClassificationTrainer
from src.telco_churn.training.tuning import LGBMTuner, KNNTuner
from src.telco_churn.utils import make_json_serializable
from src.telco_churn.visualisation import plot_roc_curve, plot_pr_curve, plot_threshold_metrics, plot_confusion_matrix, \
    plot_calibration_curve


def build_pipeline(
        config: Config,
        model_name: str
) -> Pipeline:
    model = ModelFactory.create_model(
        config=config,
        model_name=model_name,
        task_type="churn_classification"
    )

    preprocessor = Preprocessor(config)

    return Pipeline([("preprocessor", preprocessor), ("model", model)])


def fit_calibrator_and_threshold(
        config: Config,
        trainer: ClassificationTrainer,
        model_name: str,
        X_val: pd.DataFrame,
        y_val: pd.Series
):
    calibrator = None
    # Калибровка и подбор порога только для LGBM и KNN
    if model_name.lower() in config.calibrated_models:
        calibrator = ProbabilityCalibrator(method="sigmoid")

        X_val_cal, X_val_thresh, y_val_cal, y_val_thresh = train_test_split(
            X_val, y_val,
            test_size=0.5, stratify=y_val,
            random_state=config.random_state
        )

        calibrator.fit(trainer.pipeline, X_val_cal, y_val_cal)  # калибруем на одной половине
    else:
        X_val_thresh = X_val
        y_val_thresh = y_val

    if calibrator is not None:
        val_proba = calibrator.predict_proba(X_val_thresh)
    else:
        val_proba = trainer.predict_proba(X_val_thresh)

    threshold_tuner = ThresholdTuner()
    model_threshold, threshold_score = threshold_tuner.tune(y_val_thresh, val_proba)

    print("\nBest threshold:", model_threshold)
    print("Best F1:", threshold_score)

    return {
        "calibrator": calibrator,
        "threshold": model_threshold,
        "threshold_score": threshold_score,
        "X_val_thresh": X_val_thresh,
        "y_val_thresh": y_val_thresh
    }


def tune_pipeline_if_needed(
        config: Config,
        model_name: str,
        pipeline: Pipeline,
        X_train,
        y_train,
) -> dict:
    model_name = model_name.lower()
    tuner_map = {"lightgbm": LGBMTuner, "knn": KNNTuner}
    tuner_class = tuner_map.get(model_name)

    # baseline → tuning не нужен
    if tuner_class is None:
        return {}

    print(f"\nTuning {model_name}...")
    tuner = tuner_class(config)
    best_params = tuner.tune(pipeline=pipeline, X=X_train, y=y_train)

    return best_params


def train_and_select_model(
        config: Config,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        selection_metric: str,
        artifact_path: Path,
) -> dict:
    best_artifacts = {
        "model_name": None,
        "pipeline": None,
        "calibrator": None,
        "threshold": 0.5,
        "score": -1,
        "params": None
    }

    all_metrics = {}

    for model_name in config.models_classification:
        print(f"\n=== Training: {model_name} ===")
        best_params = None

        pipeline = build_pipeline(config=config, model_name=model_name)

        # Если модель не baseline, выполняем tuning
        best_params = tune_pipeline_if_needed(
            config=config,
            model_name=model_name,
            pipeline=pipeline,
            X_train=X_train,
            y_train=y_train,
        )

        pipeline.set_params(**best_params)

        trainer = ClassificationTrainer(pipeline)
        trainer.fit(X_train, y_train)

        # Калибровка и подбор порога только для LGBM и KNN
        calibration_result = fit_calibrator_and_threshold(
            config=config,
            trainer=trainer,
            model_name=model_name,
            X_val=X_val,
            y_val=y_val
        )

        metrics = evaluate_and_visualise(
            trainer=trainer,
            calibrator=calibration_result["calibrator"],
            X_val=X_val,
            y_val=y_val,
            X_val_thresh=calibration_result["X_val_thresh"],
            y_val_thresh=calibration_result["y_val_thresh"],
            X_test=X_test,
            y_test=y_test,
            threshold=calibration_result["threshold"],
            model_name=model_name,
            artifact_path=artifact_path
        )

        all_metrics[model_name] = metrics

        if selection_metric not in metrics["validation"]:
            raise ValueError(
                f"Unknown metric: {selection_metric}"
            )

        current_score = metrics["validation"][selection_metric]

        if current_score > best_artifacts["score"]:
            best_artifacts["score"] = current_score
            best_artifacts["model_name"] = model_name
            best_artifacts["pipeline"] = pipeline
            best_artifacts["calibrator"] = calibration_result["calibrator"]
            best_artifacts["threshold"] = calibration_result["threshold"]
            best_artifacts["params"] = best_params
    return {
        **best_artifacts,
        "metrics": all_metrics
    }


def evaluate_model(
        trainer: ClassificationTrainer,
        calibrator: Optional[ProbabilityCalibrator],
        y_val: pd.Series,
        X_val: pd.DataFrame,
        X_test: pd.DataFrame = None,
        y_test: pd.Series = None,
        threshold: float = 0.5,
):
    def get_proba(X):
        if calibrator is not None:
            return calibrator.predict_proba(X)
        return trainer.predict_proba(X)

    print("\nValidation metrics:")
    val_proba = get_proba(X_val)
    val_metrics = trainer.evaluate(y_true=y_val, y_proba=val_proba, threshold=threshold)
    for metric, value in val_metrics.items():
        print(f"{metric}: {value}")

    print("\nTest metrics:")
    test_proba = get_proba(X_test)
    test_metrics = trainer.evaluate(y_true=y_test, y_proba=test_proba, threshold=threshold)
    for metric, value in test_metrics.items():
        print(f"{metric}: {value}")

    return {
        "val_proba": val_proba,
        "test_proba": test_proba,
        "val_metrics": val_metrics,
        "test_metrics": test_metrics
    }


def visualise_model(
    y_test,
    test_proba,
    model_name: str,
    artifact_path=None,
    threshold: float = 0.5,
    y_val=None,
    val_proba=None,
):
    if y_val is not None and val_proba is not None:
        plot_threshold_metrics(
            y_true=y_val,
            y_proba=val_proba,
            model_name=model_name.lower(),
            output_dir=artifact_path
        )

    plot_roc_curve(
        y_true=y_test,
        y_proba=test_proba,
        model_name=model_name.lower(),
        output_dir=artifact_path
    )

    plot_pr_curve(
        y_true=y_test,
        y_proba=test_proba,
        model_name=model_name.lower(),
        output_dir=artifact_path
    )

    test_pred = (test_proba >= threshold).astype(int)
    plot_confusion_matrix(
        y_true=y_test,
        y_pred=test_pred,
        model_name=model_name.lower(),
        class_names=["No Churn", "Churn"],
        output_dir=artifact_path
    )

    plot_calibration_curve(
        y_true=y_test,
        y_proba=test_proba,
        model_name=model_name,
        output_dir=artifact_path,
        show_plot=True
    )


def evaluate_and_visualise(
        trainer: ClassificationTrainer,
        calibrator: Optional[ProbabilityCalibrator],
        y_val: pd.Series,
        X_val: pd.DataFrame,
        X_val_thresh: Optional[pd.DataFrame] = None,
        y_val_thresh: Optional[pd.Series] = None,
        X_test: pd.DataFrame = None,
        y_test: pd.Series = None,
        threshold: float = 0.5,
        model_name: str = "",
        artifact_path=None) -> dict:

    results = evaluate_model(
        trainer=trainer,
        calibrator=calibrator,
        X_val=X_val_thresh if calibrator else X_val,
        y_val=y_val_thresh if calibrator else y_val,
        X_test=X_test,
        y_test=y_test,
        threshold=threshold
    )

    visualise_model(
        y_test=y_test,
        test_proba=results["test_proba"],
        model_name=model_name,
        artifact_path=artifact_path,
        threshold=threshold,
        y_val=y_val_thresh,
        val_proba=results["val_proba"]
    )

    return {
        "validation": results["val_metrics"],
        "test": results["test_metrics"]
    }


def fit_final_pipeline(
        config: Config,
        best_result: dict,
        X_train_val: pd.DataFrame,
        y_train_val: pd.Series
) -> dict:
    print(f"\nTraining final {best_result['model_name']} models")
    final_model = ModelFactory.create_model(config=config, model_name=best_result["model_name"],
                                            task_type="churn_classification")
    if best_result["params"] is not None:
        final_model.set_params(**best_result["params"])

    final_preprocessor = Preprocessor(config)
    final_pipeline = Pipeline([("preprocessor", final_preprocessor), ("model", final_model)])

    X_fit, X_holdout, y_fit, y_holdout = train_test_split(
        X_train_val,
        y_train_val,
        test_size=0.2,
        stratify=y_train_val,
        random_state=config.random_state
    )

    X_cal, X_threshold, y_cal, y_threshold = train_test_split(
        X_holdout,
        y_holdout,
        test_size=0.5,
        stratify=y_holdout,
        random_state=config.random_state
    )

    final_pipeline.fit(X_fit, y_fit)  # models intentionally trained on X_fit to preserve calibration validity

    final_calibrator = None

    if best_result["model_name"].lower() in config.calibrated_models:
        final_calibrator = ProbabilityCalibrator(method="sigmoid")
        final_calibrator.fit(final_pipeline, X_cal, y_cal)

    if final_calibrator is not None:
        threshold_proba = final_calibrator.predict_proba(X_threshold)

    else:
        threshold_proba = final_pipeline.predict_proba(X_threshold)

    threshold_tuner = ThresholdTuner()

    best_threshold, best_f1 = threshold_tuner.tune(y_true=y_threshold, y_proba=threshold_proba)

    print(f"Final threshold: {best_threshold:.3f}")
    print(f"Final threshold F1: {best_f1:.3f}")

    return {
        "pipeline": final_pipeline,
        "calibrator": final_calibrator,
        "threshold": best_threshold
    }


def save_artifacts(
        artifact_path: Path,
        best_result: dict,
        final_result: dict,
        metrics: dict
) -> None:
    model_name = best_result["model_name"].lower()

    with open(artifact_path / "metrics.json", "w") as f:
        json.dump(make_json_serializable(metrics), f, indent=4)

    joblib.dump(final_result["pipeline"], artifact_path / f"{model_name}_pipeline.joblib")

    if final_result["calibrator"] is not None:
        joblib.dump(final_result["calibrator"], artifact_path / f"{model_name}_calibrator.joblib")

    metadata = {
        "best_model_name": model_name,
        "threshold": final_result["threshold"],
        "best_params": best_result["params"]
    }

    with open(artifact_path / "best_model.json", "w") as f:
        json.dump(make_json_serializable(metadata), f, indent=4)

    print("\nArtifacts saved successfully")
