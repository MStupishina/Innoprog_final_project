from typing import Dict, Any

import numpy as np
import optuna
from optuna.integration import LightGBMPruningCallback
from lightgbm import LGBMClassifier, early_stopping, log_evaluation
from sklearn.metrics import roc_auc_score, f1_score
from sklearn.neighbors import KNeighborsClassifier

from configs.Telco_churn_config import Config


class LGBMTuner:
    """Подбор гиперпараметров для LightGBM"""

    def __init__(self, config: Config):
        self.config = config
        self.n_trials = config.n_trials
        self.metric_name = "auc"
        self.random_state = config.random_state

    def _suggest_params(self, trial: optuna.Trial) -> Dict[str, Any]:

        params = {
            "n_estimators": 1000,
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.15),
            "num_leaves": trial.suggest_int("num_leaves", 20, 80),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "min_child_samples": trial.suggest_int("min_child_samples", 10, 100),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10, log=True),
            "scale_pos_weight": trial.suggest_float("scale_pos_weight", 1, 10),
            "random_state": self.random_state,
            "verbosity": -1,
        }

        return params


    def _objective(self, trial: optuna.Trial, X_train: np.ndarray, y_train: np.ndarray,
                   X_val: np.ndarray, y_val: np.ndarray) -> float:

        params = self._suggest_params(trial)
        model = LGBMClassifier(**params)
        model.fit(
            X_train,
            y_train,
            eval_set=[(X_val, y_val)],
            eval_metric=self.metric_name,
            callbacks=[
                LightGBMPruningCallback(trial, self.metric_name),
                # Добавляем стандартную раннюю остановку через callback
                early_stopping(stopping_rounds=50),  # стандартная ранняя остановка
                log_evaluation(period=0),
            ])
        proba = model.predict_proba(X_val)[:, 1]
        score = roc_auc_score(y_val, proba)
        return score

    def tune(self, X_train: np.ndarray, y_train: np.ndarray,
             X_val: np.ndarray, y_val: np.ndarray) -> Dict[str, Any]:
        sampler = optuna.samplers.TPESampler(seed=self.random_state)
        study = optuna.create_study(direction="maximize", sampler=sampler, study_name="lgbm_classification_tuning")
        study.optimize(
            lambda trial: self._objective(trial, X_train, y_train, X_val, y_val),
                       n_trials=self.n_trials)

        best_params = self._suggest_params(study.best_trial)
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
            "n_neighbors": trial.suggest_int("n_neighbors", 3, 20),
            "weights": trial.suggest_categorical("weights", ["uniform", "distance"]),
            "p": trial.suggest_categorical("p", [1, 2]),  # 1 = Manhattan, 2 = Euclidean
            "algorithm": trial.suggest_categorical("algorithm", ["auto", "ball_tree", "kd_tree", "brute"]),
            "leaf_size": trial.suggest_int("leaf_size", 20, 50)
        }
        return params

    def _objective(self, trial: optuna.Trial, X_train: np.ndarray, y_train: np.ndarray,
                   X_val: np.ndarray, y_val: np.ndarray) -> float:
        params = self._suggest_params(trial)
        model = KNeighborsClassifier(**params)
        model.fit(X_train, y_train)
        y_proba = model.predict_proba(X_val)[:, 1]
        y_pred = (y_proba >= 0.5).astype(int)
        score = f1_score(y_val, y_pred)
        return score

    def tune(self, X_train: np.ndarray, y_train: np.ndarray,
             X_val: np.ndarray, y_val: np.ndarray) -> Dict[str, Any]:
        sampler = optuna.samplers.TPESampler(seed=self.random_state)
        study = optuna.create_study(direction="maximize", sampler=sampler, study_name="knn_classification_tuning")
        study.optimize(lambda trial: self._objective(trial, X_train, y_train, X_val, y_val),
                       n_trials=self.n_trials)
        print("Best KNN params:", study.best_params)
        return study.best_params
