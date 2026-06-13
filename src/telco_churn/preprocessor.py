import pandas as pd
import warnings

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.utils.validation import check_is_fitted

from configs.telco_churn_config import Config


class PreprocessorClassification(BaseEstimator, TransformerMixin):
    """Класс для preprocessing'a данных """

    def __init__(self, config: Config):
        self.config = config
        self.binary_columns = config.binary_columns
        self.category_columns = config.category_columns
        self.numeric_columns = config.numeric_columns
        self.feature_columns = self.binary_columns + self.category_columns + self.numeric_columns
        self.preprocessor = None

    def _validate_columns(self, X: pd.DataFrame) -> None:
        """Проверка полноты признаков"""
        extra_columns = set(X.columns) - set(self.feature_columns)
        missing_columns = set(self.feature_columns) - set(X.columns)

        ignored_columns = {"customerID", "TotalCharges"}  # TotalCharges=monthlyCharges*tenure
        unexpected_extra = extra_columns - ignored_columns
        if unexpected_extra:
            warnings.warn(f"Лишние признаки не будут учтены: {extra_columns}")
        if missing_columns:
            raise ValueError(f"Не хватает признаков: {missing_columns}")

    def _map_yes_no(self, X: pd.DataFrame) -> pd.DataFrame:
        """Преобразование для Yes/No"""
        for column in self.binary_columns:
            if column in X.columns:
                mapped = X[column].map(self.config.yes_no_map)
                if mapped.isna().any():
                    invalid_values = X.loc[mapped.isna(), column].unique()
                    raise ValueError(
                        f"Неизвестные значения в колонке '{column}': {invalid_values}")

                X[column] = mapped.astype(float)
        return X

    def _prepare_input(self, X: pd.DataFrame) -> pd.DataFrame:
        """Общая подготовка данных перед fit/transform"""
        X = X.copy()
        self._validate_columns(X)
        valid_features = [col for col in self.feature_columns if col in X.columns]
        X = X[valid_features]
        X = self._map_yes_no(X)
        return X

    def _build_dataframe(self, X_array, X_index) -> pd.DataFrame:
        """Собирает DataFrame после трансформации"""
        all_cols = self.preprocessor.get_feature_names_out()

        return pd.DataFrame(
            X_array,
            columns=all_cols,
            index=X_index
        )

    def fit(self, X: pd.DataFrame, y=None) -> "PreprocessorClassification":
        X = self._prepare_input(X)

        self.preprocessor = ColumnTransformer(
            transformers=[
                ('numeric', StandardScaler(), self.numeric_columns),
                ('categorical', OneHotEncoder(
                    drop='first', handle_unknown='ignore', sparse_output=False),
                 self.category_columns),
                ('binary', 'passthrough', self.binary_columns)
            ]
        )

        self.preprocessor.fit(X)
        self.is_fitted_ = True
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        check_is_fitted(self, "is_fitted_")

        X = self._prepare_input(X)
        X_array = self.preprocessor.transform(X)
        return self._build_dataframe(
            X_array,
            X.index
        )
