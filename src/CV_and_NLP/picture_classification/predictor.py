import torch
from PIL import Image

from configs.cv_and_nlp_config import Config
from src.CV_and_NLP.picture_classification.model import get_resnet18
from src.CV_and_NLP.picture_classification.dataset import get_classification_transforms


def load_model(config: Config, model_path):
    model = get_resnet18(config=config, freeze_backbone=False)
    checkpoint = torch.load(model_path, map_location=config.device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(config.device)
    model.eval()

    return model


@torch.no_grad()
def predict_image(model, image_path, config: Config):
    transform = get_classification_transforms(train=False,config=config)
    image = Image.open(image_path).convert("RGB")
    image = transform(image)
    image = image.unsqueeze(0)
    image = image.to(config.device)
    logits = model(image)
    probs = torch.sigmoid(logits)[0]
    predictions = (probs > config.B1["threshold"])
    classes = []

    for idx, pred in enumerate(predictions):
        if pred:
            classes.append(config.B1["classes"][idx])

    return classes, probs.cpu().numpy()
