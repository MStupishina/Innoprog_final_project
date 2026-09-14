"""
B3: Инференс модели семантической сегментации.

Скрипт:
1. Загружает обученную U-Net.
2. Загружает одно изображение.
3. Выполняет preprocessing, аналогичный validation.
4. Получает предсказанную карту классов.
5. Сохраняет предсказанную маску.
6. Сохраняет визуализацию: оригинал → prediction.

Пример запуска:
python -m src.CV_and_NLP.segmentation.predict --image path/to/image.jpg
"""
import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from matplotlib import pyplot as plt
from torchvision import transforms
from torchvision.transforms import functional as TF

from configs.cv_and_nlp_config import Config
from src.CV_and_NLP.segmentation.dataset import VOC_PALETTE
from src.CV_and_NLP.segmentation.model import UNet


def load_model(model_path: Path, device: str) -> torch.nn.Module:
    """Загружает обученную U-Net."""
    model = UNet().to(device)
    state_dict = torch.load(
        model_path,
        map_location=device,
        weights_only=True,
    )
    model.load_state_dict(state_dict)
    model.eval()

    return model


def preprocess_image(image: Image.Image) -> torch.Tensor:
    """Preprocessing изображения, аналогичный validation в dataset.py."""
    image = image.convert("RGB")
    image = TF.resize(
        image,
        [Config.B3["img_size"], Config.B3["img_size"]],
        interpolation=transforms.InterpolationMode.BILINEAR,
    )
    image = TF.to_tensor(image)
    image = TF.normalize(
        image,
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    )

    return image.unsqueeze(0)


@torch.no_grad()
def predict(model: torch.nn.Module, image_tensor: torch.Tensor, device: str):
    """Получает карту классов для изображения."""

    image_tensor = image_tensor.to(device)
    output = model(image_tensor)
    prediction = output.argmax(dim=1).squeeze(0)

    return prediction.cpu().numpy()


def mask_to_rgb(mask: np.ndarray) -> np.ndarray:
    """Преобразует карту классов в RGB-маску VOC."""

    palette = np.array(VOC_PALETTE, dtype=np.uint8)
    rgb_mask = np.zeros((*mask.shape, 3), dtype=np.uint8)
    valid = (mask >= 0) & (mask < len(palette))
    rgb_mask[valid] = palette[mask[valid]]

    return rgb_mask


def save_prediction(
        original_image: Image.Image,
        prediction: np.ndarray,
        output_dir: Path,
        image_name: str,
):
    """Сохраняет RGB-маску и визуализацию результата."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # RGB-маска
    prediction_rgb = mask_to_rgb(prediction)
    mask_path = output_dir / f"{image_name}_mask.png"
    Image.fromarray(prediction_rgb).save(mask_path)

    # Визуализация
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(original_image)
    axes[0].set_title("Original")
    axes[0].axis("off")
    axes[1].imshow(prediction_rgb)
    axes[1].set_title("Prediction")
    axes[1].axis("off")
    plt.tight_layout()
    visualization_path = output_dir / f"{image_name}_prediction.png"
    plt.savefig(visualization_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    return mask_path, visualization_path


def main():
    parser = argparse.ArgumentParser(description="B3 semantic segmentation inference")
    parser.add_argument(
        "--image",
        type=Path,
        required=True,
        help="Path to input image",
    )

    parser.add_argument(
        "--model",
        type=Path,
        default=Config.artifacts_B3 / "best_model.pt",
        help="Path to trained model",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Config.artifacts_B3 / "predictions",
        help="Directory for prediction results",
    )

    args = parser.parse_args()

    # Проверки
    if not args.image.exists():
        raise FileNotFoundError(
            f"Изображение не найдено: {args.image}"
        )

    if not args.model.exists():
        raise FileNotFoundError(
            f"Модель не найдена: {args.model}"
        )

    device = Config.device

    print(f"Device: {device}")
    print(f"Image:  {args.image}")
    print(f"Model:  {args.model}")

    # Загружаем изображение
    original_image = Image.open(args.image).convert("RGB")

    # Загружаем модель
    model = load_model(
        model_path=args.model,
        device=device,
    )

    # Preprocessing
    image_tensor = preprocess_image(original_image)

    # Prediction
    prediction = predict(
        model=model,
        image_tensor=image_tensor,
        device=device,
    )

    # Сохраняем результат
    image_name = args.image.stem
    mask_path, visualization_path = save_prediction(
        original_image=original_image,
        prediction=prediction,
        output_dir=args.output_dir,
        image_name=image_name,
    )

    print(f"Prediction mask: {mask_path}")
    print(f"Visualization:   {visualization_path}")


if __name__ == "__main__":
    main()
