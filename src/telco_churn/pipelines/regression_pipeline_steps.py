from typing import Any

import numpy as np
import pandas as pd
from sklearn import clone
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline

from configs.telco_churn_config import Config
from src.telco_churn.models.model_factory import ModelFactory
from src.telco_churn.preprocessor import Preprocessor
from src.telco_churn.training.trainer import RegressionTrainer
from src.telco_churn.training.tuning import MLPRegressorTuner, LGBMRegressorTuner


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
        X_train,
        y_train,
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
):
    kf = KFold(n_splits=config.n_splits_a2, shuffle=True, random_state=config.random_state)
    fold_metrics = []

    for fold_idx, (train_idx, val_idx) in enumerate(kf.split(X_train), start=1):
        X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
        y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]

        fold_pipeline = clone(pipeline)

        trainer = RegressionTrainer(fold_pipeline)
        trainer.fit(X_tr, y_tr)

        metrics = trainer.evaluate(X_val, y_val)
        print(f"Fold {fold_idx}: MAE={metrics['MAE']:.4f}, RMSE={metrics['RMSE']:.4f}")
        fold_metrics.append(metrics)

    cv_metrics = {
        "MAE_mean": np.mean([m["MAE"] for m in fold_metrics]),
        "MAE_std": np.std([m["MAE"] for m in fold_metrics]),
        "RMSE_mean": np.mean([m["RMSE"] for m in fold_metrics]),
        "RMSE_std": np.std([m["RMSE"] for m in fold_metrics]),
    }

    print(
        f"\nAverage metrics for {model_name}: "
        f"MAE={cv_metrics['MAE_mean']:.4f} ± {cv_metrics['MAE_std']:.4f}, "
        f"RMSE={cv_metrics['RMSE_mean']:.4f} ± {cv_metrics['RMSE_std']:.4f}"
    )

    return {
        "summary": cv_metrics,
        "folds": fold_metrics}


def train_and_select_model(
        config: Config,
        X_train: pd.DataFrame,
        y_train: pd.Series,

):
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

        current_rmse = cv_results["summary"]["RMSE_mean"]
        if current_rmse < best_rmse:
            best_rmse = current_rmse
            best_model_name = model_name
            best_pipeline = clone(pipeline)
            best_pipeline.fit(X_train, y_train)

    print(f"\n=== Best Regression Model: {best_model_name} with CV RMSE {best_rmse:.4f} ===")
    return {
        "best_pipeline": best_pipeline,
        "best_model_name": best_model_name,
        "results": results
    }
