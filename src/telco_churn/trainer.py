from typing import Dict, Any

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score,
                             precision_score,
                             recall_score,
                             f1_score,
                             roc_auc_score,
                             confusion_matrix,
                             average_precision_score, mean_absolute_error, mean_squared_error)

class BaseTrainer:
    """Базовый класс для обучения пайпланов"""
    def __init__(self, pipeline):
        self.pipeline = pipeline

    def fit(self, X: pd.DataFrame, y: pd.Series):
        self.pipeline.fit(X, y)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.pipeline.predict(X)

    def save(self, path: str):
        joblib.dump(self.pipeline, path)

    @classmethod
    def load(cls, path: str):
        pipeline = joblib.load(path)
        return cls(pipeline)

class ClassificationTrainer(BaseTrainer):
    """Класс для обучения пайплайнов классификации"""

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return self.pipeline.predict_proba(X)[:, 1]

    def evaluate(self,
        y_true: pd.Series,
        y_proba: np.ndarray, threshold = 0.5) -> Dict[str, Any]:
        y_pred = (y_proba >= threshold).astype(int)
        metrics = {
            "accuracy": accuracy_score(y_true, y_pred),
            "precision": precision_score(y_true, y_pred, zero_division=0),
            "recall": recall_score(y_true, y_pred, zero_division=0),
            "f1": f1_score(y_true, y_pred, zero_division=0),
            "roc_auc": roc_auc_score(y_true, y_proba),
            "pr_auc": average_precision_score(y_true, y_proba),
            "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
        }
        return metrics

class RegressionTrainer:
    """Класс для обучения пайплайнов регрессии"""

    def evaluate(self, y_true: pd.Series, y_pred: np.ndarray) -> Dict[str, Any]:
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        return {"MAE": mae, "RMSE": rmse}