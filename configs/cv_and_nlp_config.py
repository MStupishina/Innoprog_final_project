from dataclasses import dataclass
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

@dataclass
class Config:
    epochs = 10
    batch_size = 32
    learning_rate = 1e-4
    data_dir = BASE_DIR / "data"
    artifacts_dir = BASE_DIR / "artifacts"