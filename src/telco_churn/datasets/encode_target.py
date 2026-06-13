import pandas as pd

def encode_target_classification(y: pd.Series, mapping: dict) -> pd.Series:
    """Преобразуем Yes/No в 0/1 для целевой переменной"""
    mapped = y.map(mapping)
    if mapped.isna().any():
        invalid = y.loc[mapped.isna()].unique()
        raise ValueError(f"Неизвестные значения target: {invalid}")
    return mapped.astype(int)

def encode_target_regression(self, y: pd.Series) -> pd.Series:
    return y.astype(float)