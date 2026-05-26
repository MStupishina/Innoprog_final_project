from dataclasses import dataclass, field
from pathlib import Path
from typing import List

BASE_DIR = Path(__file__).resolve().parent

@dataclass
class Config:
    # Data
    raw_data_path: Path = BASE_DIR/"data/raw"
    processed_data_dir: Path = BASE_DIR/"data/processed"


    target_column: str = "Churn"

    test_size: float = 0.2
    val_size: float = 0.25 # доля validation от train_val
    random_state: int = 42

    # Features
    binary_columns: List[str] = field(default_factory=lambda: [
        "gender",
        "Partner",
        "Dependents",
        "PhoneService",
        "PaperlessBilling",
    ])

    category_columns: List[str] = field(default_factory=lambda:[
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

    numeric_columns: List[str] = field(default_factory=lambda:[
        "tenure",
        "MonthlyCharges",
        "SeniorCitizen", # по смыслу булевый признак, но уже сохранен числом
    ])

    yes_no_map: dict = None

    def __post_init__(self):
        if self.yes_no_map is None:
            self.yes_no_map = {"Yes": 1, "No": 0, "Female": 1, "Male": 0}
