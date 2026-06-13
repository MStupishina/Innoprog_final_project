import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn import clone
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline

from configs.telco_churn_config import Config
from src.telco_churn.classification.calibration import ProbabilityCalibrator
from src.telco_churn.dataset.encode_target import encode_target
from src.telco_churn.preprocessor import PreprocessorClassification


class OOFPChurnGenerator:
    """
    Генерирует честные вероятности оттока (p_churn) без data leakage через Out-Of-Fold предсказания.
    Для train_val: каждый пример получает p_churn от модели, которая его НЕ видела при обучении (OOF по StratifiedKFold).
    Для test: p_churn от финальной модели, обученной на всём train_val.
    Результат сохраняется в config.p_churn_data_path для использования дальше.
    """

    def __init__(self, config: Config):
        self.config = config

    def generate_and_save(
        self,
        train_val_df: pd.DataFrame,
        test_df: pd.DataFrame,
        final_pipeline: Pipeline,        # лучший пайплайн
        final_calibrator: ProbabilityCalibrator | None,      # калибратор или None (для LogReg)
    ) -> None:
        """
        Запускает OOF-цикл, собирает датасет с p_churn и сохраняет его.
        """
        oof_proba = self._compute_oof(train_val_df, final_pipeline)
        test_proba = self._compute_test(test_df, final_pipeline, final_calibrator)

        # Собираем итоговый датасет
        tv = train_val_df.copy().reset_index(drop=True)
        tv["p_churn"] = oof_proba

        te = test_df.copy().reset_index(drop=True)
        te["p_churn"] = test_proba

        df_all = pd.concat([tv, te]).reset_index(drop=True)

        output_path = self.config.p_churn_data_path
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df_all.to_csv(output_path, index=False)
        print(f"[OOF] Сохранено {len(df_all)} строк с p_churn → {output_path}")

    def _compute_oof(
        self,
        train_val_df: pd.DataFrame,
        best_pipeline: Pipeline,
    ) -> np.ndarray:
        """
        Считает OOF вероятности для train_val через StratifiedKFold.
        Препроцессор обучается заново на каждом фолде — без утечки.
        """

        X_tv = train_val_df.drop(columns=[self.config.target_column_classification])
        y_tv = train_val_df[self.config.target_column_classification]
        y_tv = encode_target(y_tv, self.config.yes_no_map)

        oof_proba = np.zeros(len(train_val_df))

        model = best_pipeline.named_steps["model"]

        needs_calibration = isinstance(model, (LGBMClassifier, KNeighborsClassifier))

        skf = StratifiedKFold(
            n_splits=self.config.oof_n_splits, shuffle=True,
            random_state=self.config.random_state
        )

        for fold, (tr_idx, val_idx) in enumerate(skf.split(X_tv, y_tv)):
            fold_model = clone(best_pipeline.named_steps["model"])
            X_tr_raw = X_tv.iloc[tr_idx]
            y_tr_raw = y_tv.iloc[tr_idx]
            X_val_raw = X_tv.iloc[val_idx]

            # Препроцессор обучается только на данных этого фолда
            fold_pipeline = Pipeline([
                ("preprocessor", PreprocessorClassification(self.config)),
                ("model", fold_model)])

            if needs_calibration:
                # Делим fold train на train/cal для калибровки
                X_tr_m, X_cal, y_tr_m, y_cal = train_test_split(
                    X_tr_raw, y_tr_raw,
                    test_size=self.config.oof_cal_size,
                    stratify=y_tr_raw,
                    random_state=self.config.random_state
                )
                fold_pipeline.fit(X_tr_m, y_tr_m)

                fold_calib = ProbabilityCalibrator(method="sigmoid")
                fold_calib.fit(fold_pipeline, X_cal, y_cal)
                oof_proba[val_idx] = self._get_positive_class_proba(fold_calib, X_val_raw)
            else:
                fold_pipeline.fit(X_tr_raw, y_tr_raw)
                oof_proba[val_idx] = self._get_positive_class_proba(fold_pipeline, X_val_raw)

            print(f"[OOF] Fold {fold + 1}/{self.config.oof_n_splits} готов")

        return oof_proba

    def _compute_test(
        self,
        test_df: pd.DataFrame,
        best_pipeline: Pipeline,
        best_calibrator,
    ) -> np.ndarray:
        """
        Считает p_churn для test финальной моделью.
        Финальная модель обучена на train_val — test она не видела.
        """
        X_test = test_df.drop(columns=[self.config.target_column_classification])

        if best_calibrator is not None:
            return self._get_positive_class_proba(best_calibrator, X_test)

        return self._get_positive_class_proba(best_pipeline, X_test)

    def _get_positive_class_proba(self, predictor, X):
        proba = predictor.predict_proba(X)
        if proba.ndim == 2:
            return proba[:, 1]
        return proba
