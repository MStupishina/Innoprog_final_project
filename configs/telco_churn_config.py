from dataclasses import dataclass, field
from pathlib import Path
from typing import List

BASE_DIR = Path(__file__).resolve().parent.parent


@dataclass
class Config:
    # Data
    raw_data_path: Path = BASE_DIR / "data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv"
    processed_data_dir: Path = BASE_DIR / "data/processed"
    artifacts_dir: Path = BASE_DIR / "artifacts"
    p_churn_data_path: Path = BASE_DIR / "data/processed/with_p_churn.csv"

    target_column_classification: str = "Churn"

    test_size: float = 0.2
    val_size: float = 0.25  # доля validation от train_val
    random_state: int = 42

    # Features
    binary_columns: List[str] = field(default_factory=lambda: [
        "gender",
        "Partner",
        "Dependents",
        "PhoneService",
        "PaperlessBilling",
    ])

    category_columns: List[str] = field(default_factory=lambda: [
        "MultipleLines",
        "InternetService",
        "OnlineSecurity",
        "OnlineBackup",
        "DeviceProtection",
        "TechSupport",
        "StreamingTV",
        "StreamingMovies",
        "Contract",
        "PaymentMethod",
    ])

    numeric_columns: List[str] = field(default_factory=lambda: [
        "tenure",
        "MonthlyCharges",
        "SeniorCitizen",  # по смыслу булевый признак, но уже сохранен числом
    ])

    yes_no_map: dict = None

    def __post_init__(self):
        if self.yes_no_map is None:
            self.yes_no_map = {"Yes": 1, "No": 0, "Female": 1, "Male": 0}

    # Models_classification
    models_classification: List[str] = field(default_factory=lambda: [
        "logistic_regression", "lightgbm", "knn"])

    # Baseline logistic value_regression
    lr_C: float = 1.0
    lr_max_iter: int = 1000
    lr_solver: str = "lbfgs"

    # lightgbmClassification
    lgbm_n_estimators: int = 500
    lgbm_learning_rate: float = 0.05
    lgbm_num_leaves: int = 31

    # KNN
    knn_n_neighbors: int = 5

    # Models_regression
    models_regression: List[str] = field(default_factory=lambda: ["ridge", "lightgbm", "mlp"])

    # Baseline Ridge
    ridge_alpha: float = 1.0

    # LightGBM Regression
    lgbm_r_n_estimators: int = 500
    lgbm_r_learning_rate: float = 0.05
    lgbm_r_num_leaves: int = 31

    # MLP
    mlp_hidden_layer_sizes: tuple = (64, 32)
    mlp_max_iter: int = 500

    # Tunning
    n_trials: int = 10
    n_splits_clasification: int = 5

    # OOF settings
    oof_n_splits: int = 5  # число фолдов в OOF
    oof_cal_size: float = 0.2  # доля от fold train под калибровку

    # Calibrator
    calibrated_models = {"lightgbm", "knn"}
