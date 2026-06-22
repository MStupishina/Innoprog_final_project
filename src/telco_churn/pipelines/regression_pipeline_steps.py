import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn import clone
from sklearn.metrics import mean_absolute_error, root_mean_squared_error
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline

from configs.telco_churn_config import Config
from src.telco_churn.model_factory import ModelFactory
from src.telco_churn.preprocessor import Preprocessor
from src.telco_churn.training.trainer import RegressionTrainer
from src.telco_churn.training.tuning import MLPRegressorTuner, LGBMRegressorTuner
from src.telco_churn.utils import make_json_serializable
from src.telco_churn.visualisation import plot_model_comparison, plot_cv_boxplot, plot_error_by_quantiles, \
    plot_residuals, plot_actual_vs_predicted


def build_pipeline(
        config: Config,
        model_name: str
) -> Pipeline:
    model = ModelFactory.create_model(
        config=config,
        model_name=model_name,
        task_type="value_regression"
    )

    preprocessor = Preprocessor(config)

    return Pipeline([("preprocessor", preprocessor), ("model", model)])


def tune_pipeline_if_needed(
        config: Config,
        model_name: str,
        pipeline: Pipeline,
        X_train: pd.DataFrame,
        y_train: pd.Series,
) -> dict[str, Any]:
    model_name = model_name.lower()
    tuner_map = {"lightgbm": LGBMRegressorTuner, "mlp": MLPRegressorTuner}
    tuner_class = tuner_map.get(model_name)

    # baseline → tuning не нужен
    if tuner_class is None:
        return {}

    print(f"\nTuning {model_name}...")
    tuner = tuner_class(config)
    best_params = tuner.tune(pipeline=pipeline, X=X_train, y=y_train)

    return best_params


def cross_validation(
        config: Config,
        pipeline: Pipeline,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        model_name: str
) -> dict[str, Any]:
    kf = KFold(n_splits=config.n_splits_regression, shuffle=True, random_state=config.random_state)
    fold_metrics = []

    for fold_idx, (train_idx, val_idx) in enumerate(kf.split(X_train), start=1):
        X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
        y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]

        fold_pipeline = clone(pipeline)

        trainer = RegressionTrainer(fold_pipeline)
        trainer.fit(X_tr, y_tr)

        y_pred = trainer.predict(X_val)
        metrics = trainer.evaluate(y_val, y_pred)
        print(f"Fold {fold_idx}: MAE={metrics['MAE']:.4f}, RMSE={metrics['RMSE']:.4f}")
        fold_metrics.append(metrics)

    cv_metrics = {
        "MAE": np.mean([m["MAE"] for m in fold_metrics]),
        "MAE_std": np.std([m["MAE"] for m in fold_metrics]),
        "RMSE": np.mean([m["RMSE"] for m in fold_metrics]),
        "RMSE_std": np.std([m["RMSE"] for m in fold_metrics]),
    }

    print(
        f"\nAverage metrics for {model_name}: "
        f"MAE={cv_metrics['MAE']:.4f} ± {cv_metrics['MAE_std']:.4f}, "
        f"RMSE={cv_metrics['RMSE']:.4f} ± {cv_metrics['RMSE_std']:.4f}"
    )

    return {
        "summary": cv_metrics,
        "folds": fold_metrics
    }


def evaluate_best_model(
        pipeline: Pipeline,
        X_test: pd.DataFrame,
        y_test: pd.Series,
) -> tuple[dict, np.ndarray]:
    y_pred = pipeline.predict(X_test)

    metrics = {
        "MAE": mean_absolute_error(y_test, y_pred),
        "RMSE": root_mean_squared_error(y_test, y_pred)
    }

    return metrics, y_pred


def visualise_regression_models(
        results: dict[str, Any],
        plot_dir: Path,
        y_test: pd.Series,
        y_pred: np.ndarray,
        best_model_name: str,
):
    plot_model_comparison(results, plot_dir)

    plot_cv_boxplot(results, metric_name="RMSE", output_dir=plot_dir)

    plot_actual_vs_predicted(
        y_test,
        y_pred,
        model_name=best_model_name,
        output_dir=plot_dir
    )

    plot_residuals(
        y_test,
        y_pred,
        model_name=best_model_name,
        output_dir=plot_dir
    )

    plot_error_by_quantiles(
        y_test,
        y_pred,
        model_name=best_model_name,
        output_dir=plot_dir
    )


def train_and_select_model(
        config: Config,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        artifact_path: Path,
) -> dict[str, Any]:
    results = {}

    best_pipeline = None
    best_model_name = None
    best_rmse = float("inf")

    for model_name in config.models_regression:
        pipeline = build_pipeline(config=config, model_name=model_name)
        best_params = tune_pipeline_if_needed(
            config=config,
            model_name=model_name,
            pipeline=pipeline,
            X_train=X_train,
            y_train=y_train
        )
        pipeline.set_params(**best_params)

        cv_results = cross_validation(
            config=config,
            pipeline=pipeline,
            X_train=X_train,
            y_train=y_train,
            model_name=model_name)

        results[model_name] = {
            "cv": cv_results,
            "best_params": best_params
        }

        current_rmse = cv_results["summary"]["RMSE"]
        if current_rmse < best_rmse:
            best_rmse = current_rmse
            best_model_name = model_name
            best_pipeline = clone(pipeline)
            best_pipeline.fit(X_train, y_train)

    if best_pipeline is None or best_model_name is None:
        raise ValueError("Модели не обучены")

    print(f"\n=== Best Regression Model: {best_model_name} with CV RMSE {best_rmse:.4f} ===")

    test_metrics, y_pred = evaluate_best_model(
        pipeline=best_pipeline,
        X_test=X_test,
        y_test=y_test,
    )

    results[best_model_name]["test"] = test_metrics

    visualise_regression_models(
        results=results,
        plot_dir=artifact_path,
        y_test=y_test,
        y_pred=y_pred,
        best_model_name=best_model_name
    )

    return {
        "best_pipeline": best_pipeline,
        "best_model_name": best_model_name,
        "results": results,
    }


def save_artifacts(
        artifact_path: Path,
        best_results: dict[str, Any],
) -> None:
    best_model_name = best_results["best_model_name"].lower()

    with open(artifact_path / "metrics.json", "w") as f:
        json.dump(make_json_serializable(best_results["results"]), f, indent=4)

    joblib.dump(best_results["best_pipeline"], artifact_path / f"{best_model_name}_pipeline.joblib")
