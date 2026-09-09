import json
from pathlib import Path

import torch
from PIL import Image

from configs.cv_and_nlp_config import Config
from src.CV_and_NLP.picture_classification.model import get_resnet18
from src.CV_and_NLP.picture_classification.dataset import get_classification_transforms


def load_model(config: Config, model_path):
    model = get_resnet18(config=config, freeze_backbone=False)
    checkpoint = torch.load(
        model_path,
        map_location=config.device,
        weights_only=False
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(config.device)
    model.eval()

    return model


def load_threshold(config: Config, model_path):
    """Загружает threshold для модели, если отдельный threshold.json не найден,
    использует threshold из Config """
    model_dir = Path(model_path).parent
    threshold_path = model_dir / "threshold.json"

    if threshold_path.exists():
        with open(threshold_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return float(data["threshold"])

    return float(config.B1["threshold"])


@torch.no_grad()
def predict_image(model, image_path, config: Config, threshold=None):
    transform = get_classification_transforms(train=False, config=config)
    if threshold is None:
        threshold = config.B1["threshold"]
    image = Image.open(image_path).convert("RGB")
    image = transform(image)
    image = image.unsqueeze(0)
    image = image.to(config.device)
    logits = model(image)
    probs = torch.sigmoid(logits)[0]
    predictions = probs > threshold
    classes = []

    for idx, pred in enumerate(predictions):
        if pred:
            classes.append(config.B1["classes"][idx])

    return classes, probs.cpu().numpy()
