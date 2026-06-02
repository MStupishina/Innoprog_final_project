import numpy as np
from typing import Tuple
from sklearn.metrics import f1_score

class ThresholdTuner:
    """Подбор оптимального порога классификации"""

    def __init__(self, metric=f1_score):
        self.metric = metric
        self.best_threshold = 0.5
        self.best_score = 0.0

    def tune(
        self,
        y_true: np.ndarray,
        y_proba: np.ndarray,
        thresholds: np.ndarray = None,
    ) -> Tuple[float, float]:

        y_proba = np.asarray(y_proba)

        if y_proba.ndim == 2:
            y_proba = y_proba[:, 1]

        if y_proba.ndim != 1:
            raise ValueError("y_proba должен быть одномерным массивом вероятностей")

        if thresholds is None:
            thresholds = np.linspace(0.01, 0.99, 99)
        else:
            thresholds = np.asarray(thresholds)

        best_threshold = 0.5
        best_score = -1

        for threshold in thresholds:
            y_pred = (y_proba >= threshold).astype(int)

            score = self.metric(y_true, y_pred)

            if score > best_score or (
                score == best_score and threshold > best_threshold
            ):
                best_score = score
                best_threshold = threshold

        self.best_threshold = best_threshold
        self.best_score = best_score

        return best_threshold, best_score