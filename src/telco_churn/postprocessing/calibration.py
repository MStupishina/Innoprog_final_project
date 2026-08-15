import numpy as np
import pandas as pd
from sklearn.exceptions import NotFittedError
from sklearn.utils.validation import check_is_fitted
from sklearn.calibration import CalibratedClassifierCV
from sklearn.frozen import FrozenEstimator
from typing import Union


class ProbabilityCalibrator:
    """
    Класс для калибровки вероятностей модели.
    Использует Platt scaling (sigmoid) или Isotonic regression ('isotonic').
    """

    def __init__(self, method: str = "sigmoid"):
        """:param method: 'sigmoid' или 'isotonic'"""
        if method not in ["sigmoid", "isotonic"]:
            raise ValueError("Метод должен быть 'sigmoid' или 'isotonic'")
        self.method = method
        self.calibrator = None

    def fit(self, model, X_val: Union[np.ndarray, pd.DataFrame], y_val: Union[np.ndarray, pd.Series]):
        """Обучает калибратор на валидационном наборе"""
        if not hasattr(model, "predict_proba"):
            raise ValueError("Модель должна поддерживать predict_proba")

        try:
            check_is_fitted(model)
        except NotFittedError:
            raise ValueError(
                "Модель должна быть обучена до калибровки"
            )

        self.calibrator = CalibratedClassifierCV(
            estimator = FrozenEstimator(model),
            method = self.method
        )
        self.calibrator.fit(X_val, y_val)
        return self

    def predict_proba(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """Возвращает откалиброванные вероятности классов"""
        if self.calibrator is None:
            raise RuntimeError("Калибратор еще не обучен")
        return self.calibrator.predict_proba(X)[:, 1]

    def predict(self, X: Union[np.ndarray, pd.DataFrame], threshold: float = 0.5) -> np.ndarray:
        """Делает предсказания 0/1 по откалиброванным вероятностям"""
        proba = self.predict_proba(X)

        return (proba >= threshold).astype(int)