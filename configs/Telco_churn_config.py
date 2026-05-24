from dataclasses import dataclass
from pathlib import Path

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
