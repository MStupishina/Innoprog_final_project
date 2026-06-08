from pathlib import Path
from typing import Optional

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from configs.telco_churn_config import Config
from src.telco_churn.classification.calibration import ProbabilityCalibrator
from src.telco_churn.classification.threshold import ThresholdTuner
from src.telco_churn.model_factory import ModelFactory
from src.telco_churn.preprocessor import PreprocessorClassification
from src.telco_churn.trainer import ClassificationTrainer
from src.telco_churn.tuning import LGBMTuner, KNNTuner
from src.telco_churn.visualisation import plot_roc_curve, plot_pr_curve, plot_threshold_metrics, plot_confusion_matrix


def train_and_select_model(
        config: Config,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        selection_metric: str,
        model_path: Path
) -> dict:
    best_artifacts = {
        "model_name": None,
        "pipeline": None,
        "calibrator": None,
        "threshold": 0.5,
        "score": -1,
        "params": None
    }

    for model_name in config.models_classification:
        print(f"\n=== Training model: {model_name} ===")
        calibrated_models = {"lightgbm", "knn"}
        model_threshold = 0.5
        calibrator = None
        best_params = None
        X_val_thresh = None
        y_val_thresh = None

        model = ModelFactory.create_model(config=config, model_name=model_name, task_type="churn_classification")

        # Если модель не baseline, выполняем tuning
        if model_name.lower() == "lightgbm":
            tuner = LGBMTuner(config)
            best_params = tuner.tune(X_train, y_train, X_val, y_val)
            model.set_params(**best_params)
        elif model_name.lower() == "knn":
            tuner = KNNTuner(config)
            best_params = tuner.tune(X_train, y_train, X_val, y_val)
            model.set_params(**best_params)

        preprocessor = PreprocessorClassification(config)
        pipeline = Pipeline([("preprocessor", preprocessor), ("model", model)])

        trainer = ClassificationTrainer(pipeline)
        trainer.fit(X_train, y_train)

        # Калибровка и подбор порога только для LGBM и KNN
        if model_name.lower() in calibrated_models:
            calibrator = ProbabilityCalibrator(method="sigmoid")

            X_val_cal, X_val_thresh, y_val_cal, y_val_thresh = train_test_split(
                X_val, y_val,
                test_size=0.5, stratify=y_val,
                random_state=config.random_state
            )

            calibrator.fit(pipeline, X_val_cal, y_val_cal)  # калибруем на одной половине
            val_proba = calibrator.predict_proba(X_val_thresh)  # порог выбираем на другой
            threshold_tuner = ThresholdTuner()
            model_threshold, threshold_score = threshold_tuner.tune(
                y_val_thresh, val_proba
            )

            print("\nBest threshold:", model_threshold)
            print("Best F1:", threshold_score)

        metrics = evaluate_and_visualise(
            trainer=trainer,
            calibrator=calibrator,
            X_val=X_val,
            y_val=y_val,
            X_val_thresh=X_val_thresh if calibrator else None,
            y_val_thresh=y_val_thresh if calibrator else None,
            X_test=X_test,
            y_test=y_test,
            threshold=model_threshold,
            model_name=model_name,
            model_path=model_path
        )
        if selection_metric not in metrics["validation"]:
            raise ValueError(
                f"Unknown metric: {selection_metric}"
            )

        current_score = metrics["validation"][selection_metric]

        if current_score > best_artifacts["score"]:
            best_artifacts["score"] = current_score
            best_artifacts["model_name"] = model_name
            best_artifacts["pipeline"] = pipeline
            best_artifacts["calibrator"] = calibrator if model_name.lower() in calibrated_models else None
            best_artifacts["threshold"] = model_threshold if model_name.lower() in calibrated_models else 0.5
            best_artifacts["params"] = best_params
    return best_artifacts


def evaluate_and_visualise(
        trainer: ClassificationTrainer,
        calibrator,
        y_val: pd.Series,
        X_val: pd.DataFrame,
        X_val_thresh: Optional[pd.DataFrame] = None,
        y_val_thresh: Optional[pd.Series] = None,
        X_test: pd.DataFrame = None,
        y_test: pd.Series = None,
        threshold: float = 0.5,
        model_name: str = "",
        model_path=None) -> dict:
    def get_proba(X):
        if calibrator is not None:
            return calibrator.predict_proba(X)[:, 1]
        return trainer.predict_proba(X)

    print(f"\n=== {model_name.upper()} ===")
    print("\nValidation metrics:")
    val_proba = get_proba(X_val)
    val_metrics = trainer.evaluate(y_true=y_val, y_proba=val_proba, threshold=threshold)
    for metric, value in val_metrics.items():
        print(f"{metric}: {value}")

    print("\nTest metrics:")
    test_proba = get_proba(X_test)
    test_pred = (test_proba >= threshold).astype(int)
    test_metrics = trainer.evaluate(y_true=y_test, y_proba=test_proba, threshold=threshold)
    for metric, value in test_metrics.items():
        print(f"{metric}: {value}")

    if X_val_thresh is not None and y_val_thresh is not None:
        val_thresh_proba = get_proba(X_val_thresh)

        plot_threshold_metrics(
            y_true=y_val_thresh,
            y_proba=val_thresh_proba,
            model_name=model_name.lower(),
            output_dir=model_path
        )

    plot_roc_curve(
        y_true=y_test,
        y_proba=test_proba,
        model_name=model_name.lower(),
        output_dir=model_path
    )

    plot_pr_curve(
        y_true=y_test,
        y_proba=test_proba,
        model_name=model_name.lower(),
        output_dir=model_path
    )

    plot_confusion_matrix(
        y_true=y_test,
        y_pred=test_pred,
        model_name=model_name.lower(),
        class_names=["No Churn", "Churn"],
        output_dir=model_path
    )
    return {
        "validation": val_metrics,
        "test": test_metrics
    }

def fit_final_pipeline(
    config,
    best_result,
    X_train_val,
    y_train_val):
    print(f"\nTraining final{best_result['model_name']} model")
    calibrated_models = {"lightgbm", "knn"}
    final_model = ModelFactory.create_model(config=config, model_name=best_result["model_name"], task_type="churn_classification")
    if best_result["params"] is not None:
        final_model.set_params(**best_result["params"])

    final_preprocessor = PreprocessorClassification(config)
    final_pipeline = Pipeline([("preprocessor", final_preprocessor), ("model", final_model)])

    final_calibrator = None
    if best_result["model_name"].lower() in calibrated_models:
        X_fit, X_cal, y_fit, y_cal = train_test_split(
                X_train_val,
                y_train_val,
                test_size=0.2,
                stratify=y_train_val,
                random_state=config.random_state
        )

        final_pipeline.fit(X_fit, y_fit)
        final_calibrator = ProbabilityCalibrator(method="sigmoid")
        final_calibrator.fit(final_pipeline, X_cal, y_cal)
    else:
        final_pipeline.fit(X_train_val, y_train_val)
    best_threshold = best_result.get("threshold", 0.5)
    return {
    "pipeline": final_pipeline,
    "calibrator": final_calibrator,
    "threshold": best_threshold
    }

