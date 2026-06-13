import numpy as np
import pandas as pd
from pathlib import Path


def load_data(file_path: str | Path) -> pd.DataFrame:
    """
    Загружает данные из CSV или Excel файла в DataFrame и выводит информацию о размере данных.
    """
    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(f"Файл не найден: {file_path}")

    if file_path.suffix.lower() == ".csv":
        df = pd.read_csv(file_path)
    elif file_path.suffix.lower() in [".xlsx", ".xls"]:
        df = pd.read_excel(file_path)
    else:
        raise ValueError(f"Не поддерживаемый формат файла: {file_path.suffix}")

    print(f"[INFO] Загружен файл: {file_path} | Размер: {df.shape}")
    return df


def save_data(df: pd.DataFrame, file_path: str | Path, index: bool = False) -> None:
    """
    Сохраняет DataFrame в CSV или Excel файл и выводит информацию о сохранении.
    """
    file_path = Path(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)  # создаёт папки, если их нет

    if file_path.suffix.lower() == ".csv":
        df.to_csv(file_path, index=index)
    elif file_path.suffix.lower() in [".xlsx", ".xls"]:
        df.to_excel(file_path, index=index)
    else:
        raise ValueError(f"Не поддерживаемый формат файла для сохранения: {file_path.suffix}")

    print(f"[INFO] Данные сохранены в: {file_path} | Размер: {df.shape}")


def make_json_serializable(obj):
    """
    Преобразует объекты (np.ndarray, pd.Series) в типы, которые JSON умеет сериализовать.
    """
    if isinstance(obj, dict):
        return {k: make_json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [make_json_serializable(v) for v in obj]
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, pd.Series):
        return obj.tolist()
    else:
        return obj
