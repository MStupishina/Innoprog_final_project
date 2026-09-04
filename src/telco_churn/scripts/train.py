from src.telco_churn.scripts.train_pipeline_classification import (
    main as train_classification,
)
from src.telco_churn.scripts.train_pipeline_regression import (
    main as train_regression,
)
from src.telco_churn.scripts.train_pipeline_clustering import (
    main as train_clustering,
)

def train(stage: str = "all") -> None:
    """Обучения для выбранного шага"""
    if stage == "classification":
        print("\n=== Training classification ===")
        train_classification()

    elif stage == "regression":
        print("\n=== Training regression ===")
        train_regression()

    elif stage == "clustering":
        print("\n=== Training clustering ===")
        train_clustering()

    elif stage == "all":
        print("\n=== Training classification ===")
        train_classification()

        print("\n=== Training regression ===")
        train_regression()

        print("\n=== Training clustering ===")
        train_clustering()

        print("\n✓ Part A успешно выполнена")
        print("Артефакты сохранены в artifacts/")

