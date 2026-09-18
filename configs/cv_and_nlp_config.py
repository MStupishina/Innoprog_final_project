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


    # B1 Классификация изображений параметры
    B1 = {
        "image_size": 224,
        "batch_size": 32,
        "num_workers": 2,
        "num_classes": len(VOC_CLASSES),  # VOC: 20 объектов (без фона)
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

    # B2 Детекция объектов (YOLOv8) параметры
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

    # B3: Сегментация (U-Net) параметры
    B3 = {
        "in_channels": 3,
        "out_channels": 21,  # 20 классов + фон
        "features": [64, 128, 256, 512],  # каналы на каждом уровне U-Net
        "img_size": 256,
        "batch_size": 8,
        "num_epochs": 30,
        "lr": 0.001,
        "weight_decay": 1e-4,
        "ignore_index": 255,  # границы объектов в VOC (игнорируем)
        "num_workers": 2,
    }
    # B4: NLP — анализ тональности
    B4 = {
        # TF-IDF + LogReg
        "tfidf_max_features": 50000,
        "tfidf_ngram_range": (1, 2),
        "tfidf_sublinear_tf": True,
        # DistilBERT
        "transformer_model": "distilbert-base-uncased",
        "max_length": 256,
        "transformer_batch_size": 16,
        "transformer_epochs": 3,
        "transformer_lr": 2e-5,
        "warmup_steps": 0,
        "weight_decay": 0.01,
        # Данные
        "sample_size": 10000,  # подвыборка IMDb для скорости
        "test_size": 0.2,
    }
