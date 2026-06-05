from typing import Dict, Any

import numpy as np
import optuna
from optuna.integration import LightGBMPruningCallback
from lightgbm import early_stopping, log_evaluation
from sklearn import clone
from sklearn.metrics import roc_auc_score, f1_score
from sklearn.pipeline import Pipeline

from configs.telco_churn_config import Config

class LGBMTuner:
    """Подбор гиперпараметров для LightGBM"""

    def __init__(self, config: Config):
        self.config = config
        self.n_trials = config.n_trials
        self.metric_name = "auc"
        self.random_state = config.random_state

    def _suggest_params(self, trial: optuna.Trial) -> Dict[str, Any]:

        params = {
            "model__n_estimators": 1000,
            "model__learning_rate": trial.suggest_float("learning_rate", 0.01, 0.15),
            "model__num_leaves": trial.suggest_int("num_leaves", 20, 80),
            "model__max_depth": trial.suggest_int("max_depth", 3, 10),
            "model__min_child_samples": trial.suggest_int("min_child_samples", 10, 100),
            "model__subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "model__colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "model__reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10, log=True),
            "model__reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10, log=True),
            "model__scale_pos_weight": trial.suggest_float("scale_pos_weight", 1, 10),
            "model__random_state": self.random_state,
            "model__verbosity": -1,
        }

        return params


    def _objective(self, trial: optuna.Trial, pipeline: Pipeline, X_train: np.ndarray, y_train: np.ndarray,
                   X_val: np.ndarray, y_val: np.ndarray) -> float:

        params = self._suggest_params(trial)
        preprocessor = clone(pipeline.named_steps["preprocessor"])
        X_train_processed = preprocessor.fit_transform(X_train)
        X_val_processed = preprocessor.transform(X_val)
        model = clone(pipeline.named_steps["model"])
        model.set_params(**{
            k.replace("model__", ""): v
            for k, v in params.items()
        })
        model.fit(
            X_train_processed,
            y_train,
            eval_set=[(X_val_processed, y_val)],
            eval_metric=self.metric_name,
            callbacks=[
                LightGBMPruningCallback(trial, self.metric_name),
                # Добавляем стандартную раннюю остановку через callback
                early_stopping(stopping_rounds=50),  # стандартная ранняя остановка
                log_evaluation(period=0),
            ])
        proba = model.predict_proba(X_val_processed)[:, 1]

        return roc_auc_score(y_val, proba)

    def tune(self, pipeline: Pipeline, X_train: np.ndarray, y_train: np.ndarray,
             X_val: np.ndarray, y_val: np.ndarray) -> Dict[str, Any]:
        sampler = optuna.samplers.TPESampler(seed=self.random_state)
        study = optuna.create_study(direction="maximize", sampler=sampler, study_name="lgbm_classification_tuning")
        study.optimize(
            lambda trial: self._objective(trial, pipeline, X_train, y_train, X_val, y_val),
                       n_trials=self.n_trials)

        best_params = {f"model__{k}": v for k, v in study.best_params.items()}
        best_params["model__n_estimators"] = 1000
        best_params["model__random_state"] = self.random_state
        best_params["model__verbosity"] = -1
        print("Best LGBM params:", best_params)
        return best_params

class KNNTuner:
    """Подбор гиперпараметров для KNN с использованием Optuna.
    Подбирает n_neighbors, weights, p, algorithm, leaf_size.
    """
    def __init__(self, config: Config):
        self.config = config
        self.n_trials = config.n_trials
        self.random_state = config.random_state

    def _suggest_params(self, trial: optuna.Trial) -> Dict[str, Any]:
        params = {
            "model__n_neighbors": trial.suggest_int("n_neighbors", 3, 20),
            "model__weights": trial.suggest_categorical("weights", ["uniform", "distance"]),
            "model__p": trial.suggest_categorical("p", [1, 2]),  # 1 = Manhattan, 2 = Euclidean
            "model__algorithm": trial.suggest_categorical("algorithm", ["auto", "ball_tree", "kd_tree", "brute"]),
            "model__leaf_size": trial.suggest_int("leaf_size", 20, 50)
        }
        return params

    def _objective(self, trial: optuna.Trial, pipeline: Pipeline, X_train: np.ndarray, y_train: np.ndarray,
                   X_val: np.ndarray, y_val: np.ndarray) -> float:
        params = self._suggest_params(trial)
        trial_pipeline = clone(pipeline)
        trial_pipeline.set_params(**params)
        trial_pipeline.fit(X_train, y_train)
        y_proba = trial_pipeline.predict_proba(X_val)[:, 1]
        y_pred = (y_proba >= 0.5).astype(int)

        return f1_score(y_val, y_pred)

    def tune(self, pipeline: Pipeline, X_train: np.ndarray, y_train: np.ndarray,
             X_val: np.ndarray, y_val: np.ndarray) -> Dict[str, Any]:
        sampler = optuna.samplers.TPESampler(seed=self.random_state)
        study = optuna.create_study(direction="maximize", sampler=sampler, study_name="knn_classification_tuning")
        study.optimize(lambda trial: self._objective(trial, pipeline, X_train, y_train, X_val, y_val),
                       n_trials=self.n_trials)
        best_params = {f"model__{k}": v for k, v in study.best_params.items()}
        print("Best KNN params:", best_params)
        return best_params
