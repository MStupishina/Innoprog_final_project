import pandas as pd
from typing import Tuple
from sklearn.model_selection import train_test_split

from configs.telco_churn_config import Config
from src.telco_churn.utils import load_data, save_data


class DatasetLoaderClassification:
    """Класс для загрузки и разбиения датасета на train/val/test со стратификацией для классфикации"""

    def __init__(self, config: Config):
        self.config = config

    def load_and_split_data(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Загрузка и разбиение данных на train/val/test со стратификацией        """
        df = load_data(self.config.raw_data_path)
        y = df[self.config.target_column_classification]
        train_val_df, test_df = train_test_split(
            df, test_size=self.config.test_size, stratify=y, random_state=self.config.random_state)
        y_train_val = train_val_df[self.config.target_column_classification]
        train_df, val_df = train_test_split(
            train_val_df,
            test_size=self.config.val_size,
            stratify=y_train_val,
            random_state=self.config.random_state)
        return train_df, val_df, test_df, train_val_df

    def save_splits(self,
                    train_df: pd.DataFrame,
                    val_df: pd.DataFrame,
                    test_df: pd.DataFrame,
                    ) -> None:
        """Сохранение сплитов в CSV файлы с использованием utils.save_data"""
        output_dir = self.config.processed_data_dir
        output_dir.mkdir(parents=True, exist_ok=True)

        save_data(train_df, output_dir / "train.csv")
        save_data(val_df, output_dir / "val.csv")
        save_data(test_df, output_dir / "test.csv")


class DatasetLoaderRegression:
    def __init__(self, config: Config):
        self.config = config

    def load_data_with_value(self) -> pd.DataFrame:
        """Загружает данные и вычисляет customer_value"""
        df = load_data(self.config.p_churn_data_path)
        df[self.config.target_column_value] = (
                df["MonthlyCharges"] * 12 * (1 - df["p_churn"])
        )
        return df

    def load_and_split_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Делит на train/test без val_set"""
        df = self.load_data_with_value()

        # Train/Test split
        train_df, test_df = train_test_split(
            df,
            test_size=self.config.test_size,
            random_state=self.config.random_state
        )
        return train_df, test_df
