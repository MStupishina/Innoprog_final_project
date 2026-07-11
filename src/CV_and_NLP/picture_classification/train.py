import argparse
import json

import torch
from matplotlib import pyplot as plt
from sklearn.metrics import f1_score
from torch import nn, optim
from torch.optim.lr_scheduler import StepLR
from tqdm import tqdm

from configs.cv_and_nlp_config import Config
from src.CV_and_NLP.picture_classification.dataset import get_dataloaders
from src.CV_and_NLP.picture_classification.model import BaselineCNN, get_resnet18


# ===================================================================
# Утилиты
# ===================================================================
def f1_multi_label(outputs, targets):
    probs = torch.sigmoid(outputs)
    preds = (probs > 0.5).cpu().numpy()
    targets = targets.cpu().numpy()

    return f1_score(
        targets,
        preds,
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
def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    running_f1 = 0.0

    for images, targets in tqdm(loader, desc="Train", leave=False):
        images, targets = images.to(device), targets.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        running_f1 += f1_multi_label(outputs, targets) * images.size(0)

    return running_loss / len(loader.dataset), running_f1 / len(loader.dataset)


@torch.no_grad()
def validate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    running_f1 = 0.0

    for images, targets in tqdm(loader, desc="Val", leave=False):
        images, targets = images.to(device), targets.to(device)
        outputs = model(images)
        loss = criterion(outputs, targets)

        running_loss += loss.item() * images.size(0)
        running_f1 += f1_multi_label(outputs, targets) * images.size(0)

    return running_loss / len(loader.dataset), running_f1 / len(loader.dataset)


# ===================================================================
# Полный цикл обучения
# ===================================================================
def train_model(model, train_loader, val_loader, num_epochs, lr,
                step_size, lr_gamma, weight_decay, model_name, device, config: Config):
    """Обучает модель, сохраняет лучшие веса, метрики, графики."""

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = StepLR(optimizer, step_size=step_size, gamma=lr_gamma)

    save_dir = config.artifacts_B1 / model_name
    save_dir.mkdir(parents=True, exist_ok=True)

    history = {"train_loss": [], "val_loss": [], "train_f1": [], "val_f1": []}
    best_val_f1 = 0.0

    for epoch in range(1, num_epochs + 1):
        print(f"\n{'=' * 50}")
        print(f"{model_name} — Epoch {epoch}/{num_epochs}")
        print(f"LR: {scheduler.get_last_lr()[0]:.2e}")

        train_loss, train_f1 = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_f1 = validate(model, val_loader, criterion, device)

        history["train_loss"].append(round(train_loss, 4))
        history["val_loss"].append(round(val_loss, 4))
        history["train_f1"].append(round(train_f1, 4))
        history["val_f1"].append(round(val_f1, 4))

        print(f"Train Loss: {train_loss:.4f} | Train F1: {train_f1:.4f}")
        print(f"Val Loss:   {val_loss:.4f} | Val F1:   {val_f1:.4f}")

        scheduler.step()

        # Сохраняем лучшую модель
        if val_f1 > best_val_f1:
            best_val_acc = val_f1
            torch.save(model.state_dict(), save_dir / "best_model.pt")
            print(f"  ✓ Saved best model (val_f1={best_val_f1:.4f})")

    # Сохраняем финальную модель и метрики
    torch.save(model.state_dict(), save_dir / "last_model.pt")
    save_metrics(history, save_dir/ "metrics.json")
    plot_curves(history, save_dir/ "learning_curves.png", f"{model_name} — ")

    print(f"\n{'=' * 50}")
    print(f"{model_name} done! Best val_f1 = {best_val_f1:.4f}")
    print(f"Artifacts saved to {save_dir}")

    return history, best_val_f1


def main():
    config = Config()

    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", action="store_true", help="Train BaselineCNN only")
    parser.add_argument("--transfer", action="store_true", help="Train ResNet18 only")
    args = parser.parse_args()

    # Если ничего не указано — обучаем обе
    train_baseline = args.baseline or not args.transfer
    train_transfer = args.transfer or not args.baseline

    print(f"Device: {config.device}")
    train_loader, val_loader = get_dataloaders(config)

    results = {}

    if train_baseline:
        print("\n" + "=" * 60)
        print("Training BaselineCNN")
        print("=" * 60)
        model = BaselineCNN(config).to(config.device)
        _, best_acc = train_model(
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
        results["baseline_cnn"] = {"best_val_acc": best_acc}

    if train_transfer:
        print("\n" + "=" * 60)
        print("Training ResNet18 (Transfer Learning)")
        print("=" * 60)
        model = get_resnet18(config=config,freeze_backbone=True).to(config.device)
        _, best_acc = train_model(
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
        results["resnet18"] = {"best_val_acc": best_acc}

    # Сводка
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    for name, r in results.items():
        print(f"  {name}: best val_acc = {r['best_val_acc']:.4f}")


if __name__ == "__main__":
    main()
