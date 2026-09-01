import json

import numpy as np
import torch
from matplotlib import pyplot as plt
from sklearn.metrics import f1_score

from configs.cv_and_nlp_config import Config
from src.CV_and_NLP.picture_classification.dataset import get_dataloaders
from src.CV_and_NLP.picture_classification.model import get_resnet18


def load_model(config: Config):
    """Загружает обученную ResNet18 из checkpoint."""
    model = get_resnet18(config=config, freeze_backbone=True, ).to(config.device)

    checkpoint_path = (config.artifacts_B1 / "resnet18" / "best_model.pt")

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Не найден checkpoint:\n{checkpoint_path}")

    checkpoint = torch.load(
        checkpoint_path,
        map_location=config.device,
        weights_only=False,
    )

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    print(f"Checkpoint загружен: {checkpoint_path}")
    print(f"Best epoch: {checkpoint.get('best_epoch')}")
    print(f"Best val F1 при обучении: "
          f"{checkpoint.get('best_val_f1', 'N/A')}")

    return model


@torch.no_grad()
def get_validation_predictions(model, val_loader, device):
    """Получает вероятности и истинные labels на validation set."""
    all_probs = []
    all_targets = []

    for images, targets in val_loader:
        images = images.to(device, non_blocking=True)

        outputs = model(images)
        probs = torch.sigmoid(outputs)

        all_probs.append(probs.cpu())
        all_targets.append(targets.cpu())

    probs = torch.cat(all_probs).numpy()
    targets = torch.cat(all_targets).numpy()

    return probs, targets


def find_best_threshold(
        probs,
        targets,
        min_threshold=0.10,
        max_threshold=0.90,
        step=0.01,
):
    """Подбирает один общий threshold для всех 20 классов по максимальному micro-F1."""

    thresholds = np.arange(
        min_threshold,
        max_threshold + step / 2,
        step,
    )

    results = []

    for threshold in thresholds:
        predictions = (probs >= threshold).astype(int)

        f1 = f1_score(
            targets,
            predictions,
            average="micro",
            zero_division=0,
        )

        results.append({
            "threshold": float(threshold),
            "micro_f1": float(f1),
        })

    best_result = max(
        results,
        key=lambda x: x["micro_f1"],
    )

    return best_result, results


def save_results(
        config: Config,
        best_result,
        results,
):
    save_dir = config.artifacts_B1 / "resnet18"
    save_dir.mkdir(parents=True, exist_ok=True)

    output = {
        "model": "resnet18",
        "metric": "micro_f1",
        "threshold": best_result["threshold"],
        "val_micro_f1": best_result["micro_f1"],
        "search_range": {
            "min": 0.10,
            "max": 0.90,
            "step": 0.01,
        },
        "results": results,
    }

    output_path = save_dir / "threshold.json"

    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\nThreshold сохранён:")
    print(output_path)

    return output_path


def plot_results(config: Config, results):
    """Строит график зависимости F1 от threshold."""

    thresholds = [x["threshold"] for x in results]

    f1_values = [x["micro_f1"] for x in results]

    best_index = int(np.argmax(f1_values))

    plt.figure(figsize=(8, 5))
    plt.plot(thresholds, f1_values)
    plt.scatter(
        thresholds[best_index],
        f1_values[best_index],
        s=60,
        label="Best threshold",
    )

    plt.xlabel("Threshold")
    plt.ylabel("Validation micro-F1")
    plt.title("ResNet18: threshold tuning")
    plt.grid(True)
    plt.legend()
    save_path = (config.artifacts_B1 / "resnet18" / "threshold_vs_f1.png")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"График сохранён:")
    print(save_path)


def main():
    config = Config()
    print("ResNet18 — подбор threshold")
    print(f"Device: {config.device}")
    # ---------------------------------------------------------
    # Загружаем validation dataset
    # ---------------------------------------------------------
    _, val_loader, _ = get_dataloaders(config)
    # ---------------------------------------------------------
    # Загружаем обученную ResNet18
    # ---------------------------------------------------------
    model = load_model(config)
    # ---------------------------------------------------------
    # Получаем вероятности на validation
    # ---------------------------------------------------------
    probs, targets = get_validation_predictions(
        model,
        val_loader,
        config.device,
    )
    print("\nValidation predictions:")
    print(f"Samples: {len(targets)}")
    print(f"Classes: {targets.shape[1]}")
    # ---------------------------------------------------------
    # Подбираем threshold
    # ---------------------------------------------------------
    best_result, results = find_best_threshold(
        probs,
        targets,
        min_threshold=0.10,
        max_threshold=0.90,
        step=0.01,
    )
    print("\nРЕЗУЛЬТАТ")
    print(
        f"Лучший threshold: "
        f"{best_result['threshold']:.2f}"
    )

    print(
        f"Validation micro-F1: "
        f"{best_result['micro_f1']:.4f}"
    )
    # ---------------------------------------------------------
    # Сохраняем
    # ---------------------------------------------------------
    save_results(
        config,
        best_result,
        results,
    )
    plot_results(config,results)

if __name__ == "__main__":
    main()
