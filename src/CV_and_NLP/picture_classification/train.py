import argparse
import copy
import json

import torch
from matplotlib import pyplot as plt
from sklearn.metrics import f1_score, roc_auc_score
from torch import nn, optim
from torch.optim.lr_scheduler import StepLR
from tqdm import tqdm

from configs.cv_and_nlp_config import Config
from src.CV_and_NLP.picture_classification.dataset import get_dataloaders
from src.CV_and_NLP.picture_classification.model import BaselineCNN, get_resnet18
from src.CV_and_NLP.utils import set_seed


# ===================================================================
# Утилиты
# ===================================================================
def f1_calculate(preds, targets):
    return f1_score(
        targets.cpu().numpy(),
        preds.cpu().numpy(),
        average="micro",
        zero_division=0
    )


def save_metrics(metrics, path):
    with open(path, "w") as f:
        json.dump(metrics, f, indent=2)


def plot_curves(history, save_path, title_prefix=""):
    """Рисует графики loss и accuracy (train vs val)."""
    epochs = range(1, len(history["train_loss"]) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.plot(epochs, history["train_loss"], label="Train")
    ax1.plot(epochs, history["val_loss"], label="Val")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title(f"{title_prefix}Loss")
    ax1.legend()
    ax1.grid(True)

    ax2.plot(epochs, history["train_f1"], label="Train")
    ax2.plot(epochs, history["val_f1"], label="Val")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("F1")
    ax2.set_title(f"{title_prefix}F1")
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    plt.savefig(save_path, dpi=100)
    plt.close()


# ===================================================================
# Один проход обучения
# ===================================================================
def train_one_epoch(model, loader, criterion, optimizer, scaler, device, config: Config):
    model.train()
    running_loss = 0.0
    all_preds = []
    all_targets = []
    all_probs = []

    for images, targets in tqdm(loader, desc="Train", leave=False):
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        optimizer.zero_grad()
        with torch.amp.autocast(
                device_type=device,
                enabled=(device == "cuda"),
        ):
            outputs = model(images)
            loss = criterion(outputs, targets)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        probs = torch.sigmoid(outputs)
        preds = (probs > config.B1["threshold"])
        all_probs.append(probs.cpu())
        all_preds.append(preds.cpu())
        all_targets.append(targets.cpu())
        running_loss += loss.item() * images.size(0)

    all_preds = torch.cat(all_preds)
    all_targets = torch.cat(all_targets)
    all_probs = torch.cat(all_probs)
    epoch_f1 = f1_calculate(all_preds, all_targets)
    try:
        epoch_auc = roc_auc_score(
            all_targets.numpy(),
            all_probs.numpy(),
            average="macro",
        )
    except ValueError:
        epoch_auc = 0.0

    return running_loss / len(loader.dataset), epoch_f1, epoch_auc


@torch.no_grad()
def validate(model, loader, criterion, device, config: Config):
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_targets = []
    all_probs = []

    for images, targets in tqdm(loader, desc="Val", leave=False):
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        with torch.amp.autocast(
                device_type=device,
                enabled=(device == "cuda"),
        ):
            outputs = model(images)
            loss = criterion(outputs, targets)
        running_loss += loss.item() * images.size(0)
        probs = torch.sigmoid(outputs)
        preds = (probs > config.B1["threshold"])
        all_probs.append(probs.cpu())
        all_preds.append(preds.cpu())
        all_targets.append(targets.cpu())

    all_preds = torch.cat(all_preds)
    all_targets = torch.cat(all_targets)
    all_probs = torch.cat(all_probs)
    epoch_f1 = f1_calculate(all_preds, all_targets)
    try:
        epoch_auc = roc_auc_score(
            all_targets.numpy(),
            all_probs.numpy(),
            average="macro",
        )
    except ValueError:
        epoch_auc = 0.0

    return running_loss / len(loader.dataset), epoch_f1, epoch_auc


# ===================================================================
# Полный цикл обучения
# ===================================================================
def train_model(model, train_loader, val_loader, num_epochs, lr,
                step_size, lr_gamma, weight_decay, model_name, device, config: Config):
    """Обучает модель, сохраняет лучшие веса, метрики, графики."""

    criterion = nn.BCEWithLogitsLoss()
    trainable_parameters = filter(lambda p: p.requires_grad, model.parameters())

    if model_name == "resnet18":
        optimizer = optim.AdamW(
            trainable_parameters,
            lr=lr,
            weight_decay=weight_decay,
        )
    else:
        optimizer = optim.Adam(
            trainable_parameters,
            lr=lr,
            weight_decay=weight_decay,
        )
    scheduler = StepLR(optimizer, step_size=step_size, gamma=lr_gamma)
    scaler = torch.amp.GradScaler("cuda", enabled=(device == "cuda"))
    save_dir = config.artifacts_B1 / model_name
    save_dir.mkdir(parents=True, exist_ok=True)
    config_dict = {
        "device": config.device,
        "seed": config.seed,
        "data_dir": str(config.data_dir),
        "voc_dir": str(config.voc_dir),
        "artifacts_dir": str(config.artifacts_B1),
        "B1": config.B1,
    }

    with open(save_dir / "config.json", "w") as f:
        json.dump(config_dict, f, indent=2)

    history = {"train_loss": [], "val_loss": [], "train_f1": [], "val_f1": [], "train_auc": [], "val_auc": []}
    best_val_f1 = -1.0
    best_val_auc = 0.0
    best_epoch = 0
    best_state = copy.deepcopy(model.state_dict())
    epochs_without_improvement = 0

    for epoch in range(1, num_epochs + 1):
        print(f"\n{'=' * 50}")
        print(f"{model_name} — Epoch {epoch}/{num_epochs}")
        print(f"LR: {scheduler.get_last_lr()[0]:.2e}")

        train_loss, train_f1, train_auc = train_one_epoch(model, train_loader, criterion, optimizer, scaler, device, config)
        val_loss, val_f1, val_auc = validate(model, val_loader, criterion, device, config)

        history["train_loss"].append(round(train_loss, 4))
        history["val_loss"].append(round(val_loss, 4))
        history["train_f1"].append(round(train_f1, 4))
        history["val_f1"].append(round(val_f1, 4))
        history["train_auc"].append(round(train_auc, 4))
        history["val_auc"].append(round(val_auc, 4))

        print(f"Train Loss: {train_loss:.4f} | Train F1: {train_f1:.4f} | Train ROC-AUC: {train_auc:.4f}")
        print(f"Val Loss: {val_loss:.4f} | Val F1: {val_f1:.4f} | Val ROC-AUC: {val_auc:.4f}")

        scheduler.step()

        # Сохраняем лучшую модель
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_val_auc = val_auc
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            print(f"Новая лучшая модель (val_f1={best_val_f1:.4f})")
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= config.B1["patience"]:
            print("Сработала ранняя остановка")
            break

    # Сохраняем финальную модель и метрики
    model.load_state_dict(best_state)
    torch.save(
        {
            "best_epoch": best_epoch,
            "best_val_f1": best_val_f1,
            "best_val_auc": best_val_auc,
            "model_state_dict": model.state_dict(),
        },
        save_dir / "best_model.pt"
    )

    metrics = {
        "best_epoch": int(best_epoch),
        "best_val_f1": float(best_val_f1),
        "best_val_auc": float(best_val_auc),
        "history": history,
    }
    save_metrics(metrics, save_dir / "metrics.json")
    plot_curves(history, save_dir / "learning_curves.png", f"{model_name} — ")

    print(f"\n{'=' * 50}")
    print(f"{model_name} обучена. Лучший val_f1 = {best_val_f1:.4f}")
    print(f"Артефакты сохранены в {save_dir}")

    return history, best_val_f1


def main():
    config = Config()
    set_seed(config.seed)

    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", action="store_true", help="Train BaselineCNN only")
    parser.add_argument("--transfer", action="store_true", help="Train ResNet18 only")
    args = parser.parse_args()

    # Если ничего не указано — обучаем обе
    train_baseline = args.baseline or not args.transfer
    train_transfer = args.transfer or not args.baseline

    print(f"Device: {config.device}")
    train_loader, val_loader, test_loader = get_dataloaders(config)

    results = {}

    if train_baseline:
        print("\n" + "=" * 60)
        print("Обучение BaselineCNN")
        print("=" * 60)
        model = BaselineCNN(config).to(config.device)
        _, best_f1 = train_model(
            model, train_loader, val_loader,
            num_epochs=config.B1["num_epochs_baseline"],
            lr=config.B1["lr_baseline"],
            step_size=config.B1["step_size"],
            lr_gamma=config.B1["lr_gamma"],
            weight_decay=config.B1["weight_decay"],
            model_name="baseline_cnn",
            device=config.device,
            config=config,
        )
        results["baseline_cnn"] = {"best_val_f1": best_f1}

        criterion = nn.BCEWithLogitsLoss()
        checkpoint = torch.load(
            config.artifacts_B1 / "baseline_cnn" / "best_model.pt",
            map_location=config.device,
        )
        model.load_state_dict(checkpoint["model_state_dict"])
        test_loss, test_f1, test_auc = validate(
            model,
            test_loader,
            criterion,
            config.device,
            config,
        )
        results["baseline_cnn"]["test_f1"] = test_f1
        metrics_path = config.artifacts_B1 / "baseline_cnn" / "metrics.json"
        with open(metrics_path, "r") as f:
            metrics = json.load(f)
        metrics["test_loss"] = float(test_loss)
        metrics["test_f1"] = float(test_f1)
        metrics["test_auc"] = float(test_auc)
        save_metrics(metrics, metrics_path)

    if train_transfer:
        print("\n" + "=" * 60)
        print("Обучение ResNet18 (Transfer Learning)")
        print("=" * 60)
        model = get_resnet18(config=config, freeze_backbone=True).to(config.device)
        _, best_f1 = train_model(
            model, train_loader, val_loader,
            num_epochs=config.B1["num_epochs_transfer"],
            lr=config.B1["lr_transfer"],
            step_size=config.B1["step_size"],
            lr_gamma=config.B1["lr_gamma"],
            weight_decay=config.B1["weight_decay"],
            model_name="resnet18",
            device=config.device,
            config=config,
        )
        results["resnet18"] = {"best_val_f1": best_f1}

        criterion = nn.BCEWithLogitsLoss()
        checkpoint = torch.load(
            config.artifacts_B1 / "resnet18" / "best_model.pt",
            map_location=config.device,
        )
        model.load_state_dict(checkpoint["model_state_dict"])
        test_loss, test_f1, test_auc = validate(
            model,
            test_loader,
            criterion,
            config.device,
            config,
        )
        results["resnet18"]["test_f1"] = test_f1
        metrics_path = config.artifacts_B1 / "resnet18" / "metrics.json"
        with open(metrics_path, "r") as f:
            metrics = json.load(f)
        metrics["test_loss"] = float(test_loss)
        metrics["test_f1"] = float(test_f1)
        metrics["test_auc"] = float(test_auc)
        save_metrics(metrics, metrics_path)

    # Сводка
    print("\n" + "=" * 60)
    print("Свод результатов")
    print("=" * 60)
    for name, r in results.items():
        print(f"{name}: best val_f1={r['best_val_f1']:.4f} | test_f1={r['test_f1']:.4f}")

if __name__ == "__main__":
    main()
