"""
B3: Анализ ошибок семантической сегментации VOC 2012.

Скрипт:
1. Загружает лучшую U-Net модель.
2. Выполняет предсказания на validation split.
3. Считает mIoU и IoU по каждому классу.
4. Находит лучшие и худшие примеры.
5. Сохраняет визуализации:
   Original → Ground Truth → Prediction → Error.
6. Строит confusion matrix по пикселям.
7. Сохраняет результаты анализа в JSON.

Артефакты:
artifacts/segmentation/error_analysis/
├── error_analysis.json
├── per_class_iou.json
├── confusion_matrix.png
├── best_samples/
│   ├── 01_*.png
│   └── ...
└── worst_samples/
    ├── 01_*.png
    └── ...
"""
import json
from pathlib import Path

import numpy as np
import torch
from matplotlib import pyplot as plt

from configs.cv_and_nlp_config import Config
from src.CV_and_NLP.segmentation.dataset import VOCSegmentationDataset
from src.CV_and_NLP.segmentation.model import UNet


# ============================================================
# Загрузка модели
# ============================================================

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


# ============================================================
# Метрики
# ============================================================

def compute_class_iou(
        prediction: torch.Tensor,
        target: torch.Tensor,
        num_classes: int,
        ignore_index: int,
) -> list:
    """
    Считает IoU отдельно для каждого класса.
    Если класс отсутствует и в GT, и в prediction, его IoU возвращается как None.
    """

    prediction = prediction.flatten()
    target = target.flatten()
    valid = target != ignore_index
    prediction = prediction[valid]
    target = target[valid]
    class_ious = []
    for class_id in range(num_classes):
        pred_class = prediction == class_id
        target_class = target == class_id
        intersection = (pred_class & target_class).sum().item()
        union = (pred_class | target_class).sum().item()

        if union == 0:
            class_ious.append(None)
        else:
            class_ious.append(intersection / union)

    return class_ious


def compute_mean_iou(class_ious: list) -> float:
    """Средний IoU только по присутствующим классам."""

    valid_ious = [
        iou for iou in class_ious
        if iou is not None
    ]
    if not valid_ious:
        return 0.0

    return float(np.mean(valid_ious))


# ============================================================
# Подготовка изображения
# ============================================================

def denormalize_image(image: torch.Tensor) -> np.ndarray:
    """Возвращает изображение из ImageNet-normalized tensor."""

    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = image.cpu().numpy().transpose(1, 2, 0)
    image = image * std + mean

    return np.clip(image, 0.0, 1.0)


# ============================================================
# Визуализация примера
# ============================================================

def save_sample(
        image: torch.Tensor,
        target: torch.Tensor,
        prediction: torch.Tensor,
        image_id: str,
        miou: float,
        output_path: Path,
) -> None:
    """Сохраняет: Original | Ground Truth | Prediction | Error"""

    image = denormalize_image(image)
    target = target.cpu().numpy()
    prediction = prediction.cpu().numpy()
    # Ошибочные пиксели
    error = prediction != target
    # Ignore pixels не считаем ошибками
    error[target == Config.B3["ignore_index"]] = False

    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    axes[0].imshow(image)
    axes[0].set_title("Original")
    axes[1].imshow(
        target,
        cmap="tab20",
        vmin=0,
        vmax=Config.B3["out_channels"] - 1,
    )
    axes[1].set_title("Ground Truth")
    axes[2].imshow(
        prediction,
        cmap="tab20",
        vmin=0,
        vmax=Config.B3["out_channels"] - 1,
    )
    axes[2].set_title("Prediction")
    axes[3].imshow(error, cmap="gray")
    axes[3].set_title("Error")
    for ax in axes:
        ax.axis("off")
    fig.suptitle(f"{image_id} | mIoU = {miou:.4f}", fontsize=12)
    plt.tight_layout()
    plt.savefig(output_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


# ============================================================
# Confusion matrix
# ============================================================

def update_confusion_matrix(
        confusion_matrix: np.ndarray,
        prediction: torch.Tensor,
        target: torch.Tensor,
        num_classes: int,
        ignore_index: int,
) -> None:
    """Обновляет confusion matrix по пикселям. Строки: Ground Truth. Столбцы: Prediction"""

    prediction = prediction.flatten()
    target = target.flatten()
    valid = target != ignore_index
    prediction = prediction[valid]
    target = target[valid]
    indices = target * num_classes + prediction
    counts = torch.bincount(indices, minlength=num_classes * num_classes)
    confusion_matrix += (counts.reshape(num_classes, num_classes).cpu().numpy())


def save_confusion_matrix(
        confusion_matrix: np.ndarray,
        class_names: list[str],
        output_path: Path,
) -> None:
    """Сохраняет confusion matrix."""

    fig, ax = plt.subplots(figsize=(12, 10))
    ax.imshow(confusion_matrix)
    ax.set_title("Segmentation Confusion Matrix")
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("Ground truth class")
    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(
        class_names,
        rotation=90,
        fontsize=7,
    )
    ax.set_yticklabels(class_names, fontsize=7)
    plt.tight_layout()
    plt.savefig(
        output_path,
        dpi=150,
        bbox_inches="tight",
    )
    plt.close(fig)


# ============================================================
# Основной анализ
# ============================================================

@torch.no_grad()
def run_error_analysis() -> None:
    config = Config()
    device = config.device
    # --------------------------------------------------------
    # Пути
    # --------------------------------------------------------
    artifact_dir = config.artifacts_B3

    analysis_dir = artifact_dir / "error_analysis"
    best_dir = analysis_dir / "best_samples"
    worst_dir = analysis_dir / "worst_samples"
    analysis_dir.mkdir(parents=True, exist_ok=True)
    best_dir.mkdir(parents=True, exist_ok=True)
    worst_dir.mkdir(parents=True, exist_ok=True)
    model_path = artifact_dir / "best_model.pt"
    if not model_path.exists():
        raise FileNotFoundError(
            f"Не найдена модель: {model_path}"
        )

    # --------------------------------------------------------
    # Модель
    # --------------------------------------------------------

    model = load_model(
        model_path=model_path,
        device=device,
    )

    # --------------------------------------------------------
    # Validation dataset
    # --------------------------------------------------------

    dataset = VOCSegmentationDataset(
        root=config.voc_dir,
        image_set="val",
        image_size=config.B3["img_size"],
        augment=False,
    )
    print(f"Device: {device}")
    print(f"Validation samples: {len(dataset)}")
    # --------------------------------------------------------
    # Параметры
    # --------------------------------------------------------

    num_classes = config.B3["out_channels"]
    ignore_index = config.B3["ignore_index"]
    class_names = [
        "background",
        *config.VOC_CLASSES,
    ]

    # --------------------------------------------------------
    # Накопители
    # --------------------------------------------------------

    confusion_matrix = np.zeros((num_classes, num_classes), dtype=np.int64)

    # IoU каждого класса по изображениям
    class_iou_values = [[] for _ in range(num_classes)]
    samples = []

    # ========================================================
    # Inference
    # ========================================================

    for idx in range(len(dataset)):
        image, target = dataset[idx]
        image_batch = image.unsqueeze(0).to(device)
        output = model(image_batch)
        prediction = output.argmax(dim=1).squeeze(0).cpu()

        # ----------------------------------------------------
        # IoU текущего изображения
        # ----------------------------------------------------

        class_ious = compute_class_iou(
            prediction=prediction,
            target=target,
            num_classes=num_classes,
            ignore_index=ignore_index,
        )
        image_miou = compute_mean_iou(class_ious)

        # ----------------------------------------------------
        # Сохраняем IoU классов
        # ----------------------------------------------------
        for class_id, iou in enumerate(class_ious):
            if iou is not None:
                class_iou_values[class_id].append(iou)

        # ----------------------------------------------------
        # Confusion matrix
        # ----------------------------------------------------

        update_confusion_matrix(
            confusion_matrix=confusion_matrix,
            prediction=prediction,
            target=target,
            num_classes=num_classes,
            ignore_index=ignore_index,
        )

        # ----------------------------------------------------
        # Информация о примере
        # ----------------------------------------------------

        image_id = dataset.image_ids[idx]
        samples.append(
            {
                "image_id": image_id,
                "miou": image_miou,
                "image": image,
                "target": target,
                "prediction": prediction,
            }
        )

    # ========================================================
    # Сортируем изображения по mIoU
    # ========================================================

    samples.sort(key=lambda sample: sample["miou"])

    # ========================================================
    # Худшие примеры
    # ========================================================

    num_worst = min(10, len(samples))
    for rank, sample in enumerate(samples[:num_worst], start=1):
        output_path = (
                worst_dir
                / f"{rank:02d}_{sample['image_id']}.png"
        )

        save_sample(
            image=sample["image"],
            target=sample["target"],
            prediction=sample["prediction"],
            image_id=sample["image_id"],
            miou=sample["miou"],
            output_path=output_path,
        )

    # ========================================================
    # Лучшие примеры
    # ========================================================

    num_best = min(5, len(samples))
    best_samples = list(reversed(samples[-num_best:]))
    for rank, sample in enumerate(best_samples, start=1):
        output_path = (
                best_dir
                / f"{rank:02d}_{sample['image_id']}.png"
        )

        save_sample(
            image=sample["image"],
            target=sample["target"],
            prediction=sample["prediction"],
            image_id=sample["image_id"],
            miou=sample["miou"],
            output_path=output_path,
        )

    # ========================================================
    # Per-class IoU
    # ========================================================
    per_class_iou = {}
    for class_id, class_name in enumerate(class_names):
        values = class_iou_values[class_id]
        if values:
            per_class_iou[class_name] = float(np.mean(values))
        else:
            per_class_iou[class_name] = None

    # ========================================================
    # Overall mIoU
    # ========================================================
    valid_ious = [iou for iou in per_class_iou.values() if iou is not None]
    mean_iou = float(np.mean(valid_ious)) if valid_ious else 0.0

    # ========================================================
    # Confusion matrix
    # ========================================================

    save_confusion_matrix(
        confusion_matrix=confusion_matrix,
        class_names=class_names,
        output_path=analysis_dir / "confusion_matrix.png",
    )

    # ========================================================
    # JSON
    # ========================================================
    worst_results = [
        {
            "image_id": sample["image_id"],
            "miou": round(sample["miou"], 4),
        }
        for sample in samples[:num_worst]
    ]

    best_results = [
        {
            "image_id": sample["image_id"],
            "miou": round(sample["miou"], 4),
        }
        for sample in best_samples
    ]

    results = {
        "split": "val",
        "num_samples": len(dataset),
        "mean_iou": round(mean_iou, 4),
        "per_class_iou": {
            class_name: None if iou is None else round(iou, 4)
            for class_name, iou
            in per_class_iou.items()
        },
        "worst_samples": worst_results,
        "best_samples": best_results,
    }

    with open(analysis_dir / "error_analysis.json", "w", encoding="utf-8") as file:
        json.dump(results, file, indent=2, ensure_ascii=False)

    with open(analysis_dir / "per_class_iou.json", "w", encoding="utf-8") as file:
        json.dump(results["per_class_iou"], file, indent=2, ensure_ascii=False)

    # ========================================================
    # Вывод
    # ========================================================

    print("\n" + "=" * 60)
    print("B3 ERROR ANALYSIS")
    print("=" * 60)
    print(f"Split:       val")
    print(f"Samples:     {len(dataset)}")
    print(f"mIoU:        {mean_iou:.4f}")
    print("\nPer-class IoU:")
    for class_name, iou in (results["per_class_iou"].items()):
        if iou is None:
            print(f"{class_name:15s}: N/A")
        else:
            print(f"{class_name:15s}: {iou:.4f}")
    print("\nWorst samples:")
    for sample in worst_results:
        print(f"{sample['image_id']:15s} mIoU={sample['miou']:.4f}")
    print(f"\nResults saved to: {analysis_dir}")


if __name__ == "__main__":
    run_error_analysis()
