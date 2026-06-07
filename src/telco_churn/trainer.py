from typing import Dict, Any

import numpy as np
import pandas as pd
from sklearn.exceptions import NotFittedError
from sklearn.metrics import (accuracy_score,
                             precision_score,
                             recall_score,
                             f1_score,
                             roc_auc_score,
                             confusion_matrix,
                             average_precision_score)
from sklearn.utils.validation import check_is_fitted

class ClassificationTrainer:
    """Класс для обучения моделей классификации"""
    def __init__(self, pipeline):
        self.pipeline = pipeline

    def _check_fitted(self):
        try:
            check_is_fitted(self.pipeline.named_steps["model"])
        except NotFittedError:
            raise RuntimeError("Модель еще не обучена")

    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        self.pipeline.fit(X, y)

    def predict(self, X: pd.DataFrame) -> pd.Series:
        self._check_fitted()
        return self.pipeline.predict(X)

    def predict_proba(self, X: pd.DataFrame) -> pd.Series:
        self._check_fitted()
        return self.pipeline.predict_proba(X)[:, 1]

    def evaluate(self, y_true: pd.Series, y_pred: pd.Series, y_proba: pd.Series) -> Dict[str, Any]:
        metrics = {
            "accuracy": accuracy_score(y_true, y_pred),
            "precision": precision_score(y_true, y_pred, zero_division=0),
            "recall": recall_score(y_true, y_pred, zero_division=0),
            "f1": f1_score(y_true, y_pred, zero_division=0),
            "roc_auc": roc_auc_score(y_true, y_proba),
            "confusion_matrix": confusion_matrix(y_true, y_pred),
            "pr_auc": average_precision_score(y_true, y_proba)
        }
        return metrics
