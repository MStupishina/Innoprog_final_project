from dataclasses import dataclass
from pathlib import Path

import torch

BASE_DIR = Path(__file__).resolve().parent.parent


@dataclass
class Config:
    # Данные
    data_dir: Path = BASE_DIR / "data/datasets"
    voc_dir: Path = data_dir
    imdb_cache: Path = data_dir / "imdb"

    # Артефакты (веса, метрики, графики)
    artifacts_dir: Path = BASE_DIR / "artifacts"
    artifacts_B1: Path = artifacts_dir / "cv_classification"
    artifacts_B2: Path = artifacts_dir / "detection"
    artifacts_B3: Path = artifacts_dir / "segmentation"
    artifacts_B4: Path = artifacts_dir / "nlp"

    # Устройство
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Общие
    seed = 42

    VOC_CLASSES = [
            "aeroplane", "bicycle", "bird", "boat", "bottle",
            "bus", "car", "cat", "chair", "cow",
            "diningtable", "dog", "horse", "motorbike", "person",
            "pottedplant", "sheep", "sofa", "train", "tvmonitor",
        ]


    # B1 параметры
    B1 = {
        "image_size": 224,
        "batch_size": 32,
        "num_workers": 2,
        "num_classes": 20,  # VOC: 20 объектов (без фона)
        "num_epochs_baseline": 25,
        "num_epochs_transfer": 15,
        "lr_baseline": 0.001,
        "lr_transfer": 0.0001,
        "step_size": 7,
        "lr_gamma": 0.1,
        "dropout": 0.5,
        "weight_decay": 1e-4,
        "threshold": 0.5,
        "classes":VOC_CLASSES,
        "patience": 5,
        "train_size": 0.7,
        "val_size": 0.15,
        "test_size": 0.15,
    }

    # B2 параметры
    B2 = {
        "model": "yolov8n.pt",  # nano — быстро, для прототипа
        "imgsz": 640,
        "batch": 16,
        "epochs": 30,
        "lr0": 0.01,
        "lrf": 0.01,  # final lr = lr0 * lrf
        "patience": 10,  # early stopping
        "conf_threshold": 0.25,  # для инференса
        "iou_threshold": 0.45,  # NMS
    }

