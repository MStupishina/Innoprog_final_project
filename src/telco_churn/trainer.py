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

from src.telco_churn.preprocessor import PreprocessorClassification


class ClassificationTrainer:
    """Класс для обучения моделей классификации"""
    def __init__(self, model,  preprocessor: PreprocessorClassification):
        self.model = model
        self.preprocessor = preprocessor
        self.is_fitted = False

    def _check_fitted(self):
        try:
            check_is_fitted(self.model)
        except NotFittedError:
            raise RuntimeError("Модель еще не обучена")

    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        self.model.fit(X, y)
        self.is_fitted = True

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        self._check_fitted()
        return self.model.predict(X)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        self._check_fitted()
        return self.model.predict_proba(X)[:, 1]

    def evaluate(self, y_true: np.ndarray, y_pred: np.ndarray, y_proba: np.ndarray) -> Dict[str, Any]:
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
