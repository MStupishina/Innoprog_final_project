from typing import Dict, Any

import numpy as np
import optuna
import pandas as pd
from optuna_integration.lightgbm import LightGBMPruningCallback
from lightgbm import early_stopping, log_evaluation
from sklearn import clone
from sklearn.metrics import roc_auc_score, mean_squared_error
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.pipeline import Pipeline

from configs.telco_churn_config import Config


class LGBMTuner:
    """Подбор гиперпараметров для LightGBM"""

    def __init__(self, config: Config):
        self.config = config
        self.n_trials = config.n_trials
        self.random_state = config.random_state
        self.n_splits = config.n_splits_clasification

    def _suggest_params(self, trial: optuna.Trial) -> Dict[str, Any]:
        params = {
            "model__n_estimators": self.config.lgbm_n_estimators,
            "model__learning_rate": trial.suggest_float("learning_rate", 0.01, 0.15),
            "model__num_leaves": trial.suggest_int("num_leaves", 20, 80),
            "model__min_child_samples": trial.suggest_int("min_child_samples", 10, 50),
            "model__subsample": trial.suggest_float("subsample", 0.7, 1.0),
            "model__colsample_bytree": trial.suggest_float("colsample_bytree", 0.7, 1.0),
            "model__reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 5, log=True),
            "model__reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 5, log=True),
            "model__random_state": self.random_state,
            "model__verbosity": -1,
        }

        return params

    def _objective(self, trial: optuna.Trial, pipeline: Pipeline, X: pd.DataFrame, y: pd.Series) -> float:
        params = self._suggest_params(trial)
        cv = StratifiedKFold(n_splits=self.n_splits, shuffle=True, random_state=self.random_state)
        fold_scores = []

        for train_idx, val_idx in cv.split(X, y):
            X_train_fold = X.iloc[train_idx]
            X_val_fold = X.iloc[val_idx]

            y_train_fold = y.iloc[train_idx]
            y_val_fold = y.iloc[val_idx]

            # Новый preprocessor для каждого fold
            preprocessor = clone(pipeline.named_steps["preprocessor"])

            X_train_processed = preprocessor.fit_transform(X_train_fold)
            X_val_processed = preprocessor.transform(X_val_fold)

            # Новая модель для каждого fold
            model = clone(pipeline.named_steps["model"])

            model.set_params(**{
                k.replace("model__", ""): v
                for k, v in params.items()
            })

            model.fit(
                X_train_processed,
                y_train_fold,
                eval_set=[(X_val_processed, y_val_fold)],
                eval_metric="auc",
                callbacks=[
                    LightGBMPruningCallback(trial, "auc"),
                    # Добавляем стандартную раннюю остановку через callback
                    early_stopping(stopping_rounds=50),
                    log_evaluation(period=0),
                ])

            y_proba = model.predict_proba(X_val_processed)[:, 1]
            score = roc_auc_score(y_val_fold, y_proba)
            fold_scores.append(score)

        return np.mean(fold_scores)

    def tune(self, pipeline: Pipeline, X: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
        sampler = optuna.samplers.TPESampler(seed=self.random_state)
        study = optuna.create_study(direction="maximize", sampler=sampler, study_name="lgbm_classification_tuning")
        study.optimize(
            lambda trial: self._objective(trial, pipeline, X, y),
            n_trials=self.n_trials)

        best_params = {f"model__{k}": v for k, v in study.best_params.items()}
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
        self.n_splits = config.n_splits_clasification

    def _suggest_params(self, trial: optuna.Trial) -> Dict[str, Any]:
        params = {
            "model__n_neighbors": trial.suggest_int("n_neighbors", 3, 20),
            "model__weights": trial.suggest_categorical("weights", ["uniform", "distance"]),
            "model__p": trial.suggest_categorical("p", [1, 2]),  # 1 = Manhattan, 2 = Euclidean
            "model__algorithm": trial.suggest_categorical("algorithm", ["auto"]),
        }
        return params

    def _objective(self, trial: optuna.Trial, pipeline: Pipeline, X: np.ndarray, y: np.ndarray) -> float:
        params = self._suggest_params(trial)

        cv = StratifiedKFold(n_splits=self.n_splits, shuffle=True, random_state=self.random_state)
        fold_scores = []

        for train_idx, val_idx in cv.split(X, y):
            X_train_fold = X.iloc[train_idx]
            X_val_fold = X.iloc[val_idx]

            y_train_fold = y.iloc[train_idx]
            y_val_fold = y.iloc[val_idx]

            trial_pipeline = clone(pipeline)
            trial_pipeline.set_params(**params)
            trial_pipeline.fit(X_train_fold, y_train_fold)

            y_proba = trial_pipeline.predict_proba(X_val_fold)[:, 1]

            score = roc_auc_score(y_val_fold, y_proba)
            fold_scores.append(score)

        return np.mean(fold_scores)

    def tune(self, pipeline: Pipeline, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        sampler = optuna.samplers.TPESampler(seed=self.random_state)
        study = optuna.create_study(direction="maximize", sampler=sampler, study_name="knn_classification_tuning")
        study.optimize(lambda trial: self._objective(trial, pipeline, X, y),
                       n_trials=self.n_trials)
        best_params = {f"model__{k}": v for k, v in study.best_params.items()}
        print("Best KNN params:", best_params)
        return best_params


class LGBMRegressorTuner:
    """Подбор гиперпараметров для LGBMRegressor с использованием Optuna и k-fold CV. Оптимизируем RMSE"""

    def __init__(self, config: Config):
        self.config = config
        self.n_trials = config.n_trials
        self.n_splits = config.n_splits_regression
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
            "model__random_state": self.random_state,
        }
        return params

    def _objective(self, trial: optuna.Trial, pipeline: Pipeline, X: pd.DataFrame, y: pd.Series) -> float:
        params = self._suggest_params(trial)
        cv = KFold(n_splits=self.n_splits, shuffle=True, random_state=self.random_state)
        scores = []

        for train_idx, val_idx in cv.split(X):
            X_train_fold = X.iloc[train_idx]
            X_val_fold = X.iloc[val_idx]
            y_train_fold = y.iloc[train_idx]
            y_val_fold = y.iloc[val_idx]

            # Препроцессор внутри фолда
            preprocessor = clone(pipeline.named_steps["preprocessor"])
            X_train_processed = preprocessor.fit_transform(X_train_fold)
            X_val_processed = preprocessor.transform(X_val_fold)

            model = clone(pipeline.named_steps["model"])

            model.set_params(**{
                k.replace("model__", ""): v
                for k, v in params.items()
            })

            model.fit(
                X_train_processed, y_train_fold,
                eval_set=[(X_val_processed, y_val_fold)],
                eval_metric="rmse",
                callbacks=[early_stopping(stopping_rounds=50),
                           log_evaluation(period=0)],
            )
            y_pred = model.predict(X_val_processed)
            mse = mean_squared_error(y_val_fold, y_pred)  # по умолчанию squared=True
            score = np.sqrt(mse)  # RMSE
            scores.append(score)

        return np.mean(scores)

    def tune(self, pipeline: Pipeline, X: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
        study = optuna.create_study(direction="minimize")  # минимизируем RMSE
        study.optimize(lambda trial: self._objective(trial, pipeline, X, y), n_trials=self.n_trials)
        best_params = {f"model__{k}": v for k, v in study.best_params.items()}
        print("Best LGBM params:", best_params)
        return best_params

class MLPRegressorTuner:
    """Подбор гиперпараметров для MLPRegressor с использованием k-fold CV. Оптимизируем RMSE."""

    def __init__(self, config: Config):
        self.config = config
        self.n_trials = config.n_trials
        self.n_splits = config.n_splits_regression
        self.random_state = config.random_state

    def _suggest_params(self, trial: optuna.Trial) -> Dict[str, Any]:
        # 1. Сначала просим Optuna выбрать количество слоев
        n_layers = trial.suggest_int("n_layers", 1, 3)

        # 2. Генерируем количество нейронов для каждого слоя
        hidden_layer_sizes = tuple([
            trial.suggest_int(f"n_units_l{i}", 10, 200) for i in range(n_layers)
        ])

        # 3. Собираем итоговый "чистый" словарь для sklearn
        params = {
            "model__hidden_layer_sizes": hidden_layer_sizes,
            "model__activation": trial.suggest_categorical("activation", ["relu", "tanh"]),
            "model__solver": trial.suggest_categorical("solver", ["adam", "lbfgs"]),
            "model__alpha": trial.suggest_float("alpha", 1e-5, 1e-1, log=True),
            "model__learning_rate_init": trial.suggest_float("learning_rate_init", 1e-4, 1e-1, log=True),
            "model__max_iter": 500,
            "model__random_state": self.random_state,
        }
        return params

    def _objective(self, trial: optuna.Trial, pipeline: Pipeline, X: pd.DataFrame, y: pd.Series) -> float:
        # Генерируем параметры ОДИН РАЗ для всего Trial (до фолдов)
        params = self._suggest_params(trial)

        # Добавляем параметры, специфичные для обучения (не для Optuna)
        params.update({
            "model__early_stopping": True,
            "model__validation_fraction": 0.1,
        })

        cv = KFold(n_splits=self.n_splits, shuffle=True, random_state=self.random_state)
        scores = []

        for train_idx, val_idx in cv.split(X):
            X_train_fold = X.iloc[train_idx]
            X_val_fold = X.iloc[val_idx]
            y_train_fold = y.iloc[train_idx]
            y_val_fold = y.iloc[val_idx]

            trial_pipeline = clone(pipeline)
            trial_pipeline.set_params(**params)

            trial_pipeline.fit(X_train_fold, y_train_fold)
            y_pred = trial_pipeline.predict(X_val_fold)

            mse = mean_squared_error(y_val_fold, y_pred)
            score = np.sqrt(mse)  # RMSE
            scores.append(score)

        return np.mean(scores)

    def tune(self, pipeline: Pipeline, X: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
        study = optuna.create_study(direction="minimize")
        study.optimize(lambda trial: self._objective(trial, pipeline, X, y), n_trials=self.n_trials)

        best_raw = study.best_params.copy()

        clean_params = {
            k.replace("model__", ""): v
            for k, v in best_raw.items()
        }

        n_layers = clean_params.pop("n_layers")
        hidden_layer_sizes = tuple([
            best_raw.pop(f"n_units_l{i}") for i in range(n_layers)
        ])

        # Возвращаем чистый словарь
        best_params = {
            "model__hidden_layer_sizes": hidden_layer_sizes,
            "model__activation": clean_params["activation"],
            "model__solver": clean_params["solver"],
            "model__alpha": clean_params["alpha"],
            "model__learning_rate_init": clean_params["learning_rate_init"],
            "model__max_iter": 1000,  # На финальном прогоне даем больше времени сойтись
            "model__random_state": self.random_state,
        }

        print(f"Best MLPRegressor params: {best_params}")
        return best_params
