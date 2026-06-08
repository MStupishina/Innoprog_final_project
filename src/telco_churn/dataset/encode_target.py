import pandas as pd

def encode_target(y: pd.Series, mapping: dict) -> pd.Series:
    """Преобразуем Yes/No в 0/1 для целевой переменной"""
    mapped = y.map(mapping)
    if mapped.isna().any():
        invalid = y.loc[mapped.isna()].unique()
        raise ValueError(f"Неизвестные значения target: {invalid}")
    return mapped.astype(int)