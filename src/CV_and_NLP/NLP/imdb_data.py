from datasets import load_dataset
from sklearn.model_selection import train_test_split


def load_imdb_splits(config):
    """Возвращает одинаковые train/val/test для обеих моделей."""

    print("Loading IMDb dataset...")
    dataset = load_dataset("stanfordnlp/imdb", cache_dir=str(config.imdb_cache))
    # Официальный train
    train_df = dataset["train"].to_pandas()
    # Одна и та же подвыборка для обеих моделей
    sample_size = config.B4["sample_size"]
    if sample_size and sample_size < len(train_df):
        train_df = train_df.sample(
            n=sample_size,
            random_state=config.seed,
        )
    # Одинаковое стратифицированное разбиение
    train_df, val_df = train_test_split(
        train_df,
        test_size=config.B4["val_size"],
        random_state=config.seed,
        stratify=train_df["label"],
    )
    # Сбрасываем индексы после разбиения
    train_df = train_df.reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)
    # Официальный test не изменяем и не используем для подбора модели
    test_df = dataset["test"].to_pandas().reset_index(drop=True)
    print(
        f"Train: {len(train_df)}, "
        f"Val: {len(val_df)}, "
        f"Test: {len(test_df)}"
    )
    return train_df, val_df, test_df
