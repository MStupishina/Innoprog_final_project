# ===================================================================
# mIoU (mean Intersection over Union) — основная метрика
# ===================================================================
import json

import torch
from matplotlib import pyplot as plt
from torch import nn, optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

from configs.cv_and_nlp_config import Config
from src.CV_and_NLP.segmentation.dataset import get_segmentation_dataloaders
from src.CV_and_NLP.segmentation.model import UNet


def compute_miou(pred, target, num_classes, ignore_index=255):
    """Считает mean IoU по всему набору данных."""
    valid = target != ignore_index
    pred = pred[valid]
    target = target[valid]
    ious = []
    for cls in range(num_classes):
        pred_cls = pred == cls
        target_cls = target == cls

        intersection = (pred_cls & target_cls).sum().float()
        union = (pred_cls | target_cls).sum().float()

        if union == 0:
            continue

        iou = intersection / union
        ious.append(iou.item())

    return sum(ious) / len(ious) if ious else 0.0


# ===================================================================
# Обучение одной эпохи
# ===================================================================
def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0

    for images, masks in tqdm(loader, desc="Train", leave=False):
        images, masks = images.to(device), masks.to(device)

        optimizer.zero_grad()
        outputs = model(images)  # [B, 21, H, W]
        loss = criterion(outputs, masks)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)

    return running_loss / len(loader.dataset)


@torch.no_grad()
def validate(model, loader, criterion, device, num_classes, ignore_index):
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_targets = []

    for images, masks in tqdm(loader, desc="Val", leave=False):
        images, masks = images.to(device), masks.to(device)
        outputs = model(images)
        loss = criterion(outputs, masks)
        running_loss += loss.item() * images.size(0)

        # mIoU
        preds = outputs.argmax(dim=1)  # [B, H, W]
        all_preds.append(preds.cpu())
        all_targets.append(masks.cpu())

    all_preds = torch.cat(all_preds, dim=0)
    all_targets = torch.cat(all_targets, dim=0)
    avg_loss = running_loss / len(loader.dataset)
    avg_miou = compute_miou(
        all_preds,
        all_targets,
        num_classes,
        ignore_index,
    )
    return avg_loss, avg_miou


# ===================================================================
# Полный цикл
# ===================================================================
def main():
    config = Config()
    print(f"Device: {config.device}")

    train_loader, val_loader = get_segmentation_dataloaders()

    model = UNet().to(config.device)
    criterion = nn.CrossEntropyLoss(ignore_index=config.B3["ignore_index"])
    optimizer = optim.Adam(model.parameters(), lr=config.B3["lr"],
                           weight_decay=config.B3["weight_decay"])
    scheduler = CosineAnnealingLR(optimizer, T_max=config.B3["num_epochs"])

    save_dir = config.artifacts_B3
    save_dir.mkdir(parents=True, exist_ok=True)

    history = {"train_loss": [], "val_loss": [], "val_miou": []}
    best_miou = -1.0

    for epoch in range(1, config.B3["num_epochs"] + 1):
        print(f"\n{'=' * 50}")
        print(f"Epoch {epoch}/{config.B3['num_epochs']} | LR: {scheduler.get_last_lr()[0]:.2e}")

        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, config.device)
        val_loss, val_miou = validate(
            model,
            val_loader,
            criterion,
            config.device,
            config.B3["out_channels"],
            config.B3["ignore_index"],
        )

        history["train_loss"].append(round(train_loss, 4))
        history["val_loss"].append(round(val_loss, 4))
        history["val_miou"].append(round(val_miou, 4))

        print(f"Train Loss: {train_loss:.4f}")
        print(f"Val Loss:   {val_loss:.4f} | Val mIoU: {val_miou:.4f}")

        scheduler.step()

        # Сохраняем лучшую
        if val_miou > best_miou:
            best_miou = val_miou
            torch.save(model.state_dict(), save_dir / "best_model.pt")
            print(f"  ✓ Saved (mIoU={best_miou:.4f})")

    # Финальное сохранение
    torch.save(model.state_dict(), save_dir / "last_model.pt")

    # Графики
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    epochs = range(1, len(history["train_loss"]) + 1)
    ax1.plot(epochs, history["train_loss"], label="Train")
    ax1.plot(epochs, history["val_loss"], label="Val")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title("Loss")
    ax1.legend()
    ax1.grid(True)

    ax2.plot(epochs, history["val_miou"], label="Val mIoU", color="green")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("mIoU")
    ax2.set_title("Mean IoU")
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    plt.savefig(save_dir / "learning_curves.png", dpi=100)
    plt.close()

    # Метрики
    metrics = {**history, "best_val_miou": best_miou}
    with open(save_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\n{'=' * 50}")
    print(f"Training complete! Best mIoU: {best_miou:.4f}")
    print(f"Artifacts saved to: {save_dir}")


if __name__ == "__main__":
    main()
