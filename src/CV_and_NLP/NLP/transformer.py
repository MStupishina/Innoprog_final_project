import json

import torch
from datasets import load_dataset
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from transformers import (DistilBertTokenizer, DistilBertForSequenceClassification,
                         get_linear_schedule_with_warmup, AdamW)

from configs.cv_and_nlp_config import Config


class IMDbDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]

        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "label": torch.tensor(label, dtype=torch.long),
        }


# ===================================================================
# Обучение
# ===================================================================
def train_epoch(model, loader, optimizer, scheduler, device):
    model.train()
    total_loss = 0.0

    for batch in tqdm(loader, desc="Train", leave=False):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["label"].to(device)

        optimizer.zero_grad()
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        scheduler.step()

        total_loss += loss.item()

    return total_loss / len(loader)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    all_preds = []
    all_labels = []
    total_loss = 0.0

    for batch in tqdm(loader, desc="Eval", leave=False):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["label"].to(device)

        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        total_loss += outputs.loss.item()

        preds = outputs.logits.argmax(dim=-1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds)
    return total_loss / len(loader), acc, f1, all_preds, all_labels


# ===================================================================
# Main
# ===================================================================
def main():
    config = Config()
    save_dir = config.artifacts_B4 / "distilbert"
    save_dir.mkdir(parents=True, exist_ok=True)

    print(f"Device: {config.device}")

    # ── Загрузка данных ──
    print("Loading IMDb dataset...")
    dataset = load_dataset("stanfordnlp/imdb")
    train_data = dataset["train"]
    test_data = dataset["test"]

    # Подвыборка для скорости
    if config.B4["sample_size"] and config.B4["sample_size"] < len(train_data):
        train_data = train_data.shuffle(seed=config.seed).select(range(config.B4["sample_size"]))

    print(f"Train: {len(train_data)}, Test: {len(test_data)}")

    # ── Tokenizer ──
    tokenizer = DistilBertTokenizer.from_pretrained(config.B4["transformer_model"])

    # ── Датасеты ──
    train_dataset = IMDbDataset(
        train_data["text"], train_data["label"],
        tokenizer, config.B4["max_length"],
    )
    test_dataset = IMDbDataset(
        test_data["text"], test_data["label"],
        tokenizer, config.B4["max_length"],
    )

    train_loader = DataLoader(train_dataset, batch_size=config.B4["transformer_batch_size"],
                              shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=config.B4["transformer_batch_size"],
                             shuffle=False)

    # ── Модель ──
    model = DistilBertForSequenceClassification.from_pretrained(
        config.B4["transformer_model"],
        num_labels=2,
    ).to(config.device)

    # ── Оптимизатор + Scheduler ──
    optimizer = AdamW(model.parameters(), lr=config.B4["transformer_lr"],
                      weight_decay=config.B4["weight_decay"])
    total_steps = len(train_loader) * config.B4["transformer_epochs"]
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=config.B4["warmup_steps"],
        num_training_steps=total_steps,
    )

    # ── Обучение ──
    history = {"train_loss": [], "val_loss": [], "val_acc": [], "val_f1": []}
    best_f1 = 0.0

    for epoch in range(1, config.B4["transformer_epochs"] + 1):
        print(f"\n{'='*50}")
        print(f"Epoch {epoch}/{config.B4['transformer_epochs']}")

        train_loss = train_epoch(model, train_loader, optimizer, scheduler, config.device)
        val_loss, val_acc, val_f1, val_preds, val_labels = evaluate(model, test_loader, config.device)

        history["train_loss"].append(round(train_loss, 4))
        history["val_loss"].append(round(val_loss, 4))
        history["val_acc"].append(round(val_acc, 4))
        history["val_f1"].append(round(val_f1, 4))

        print(f"Train Loss: {train_loss:.4f}")
        print(f"Val Loss:   {val_loss:.4f} | Val Acc: {val_acc:.4f} | Val F1: {val_f1:.4f}")

        if val_f1 > best_f1:
            best_f1 = val_f1
            model.save_pretrained(save_dir)
            tokenizer.save_pretrained(save_dir)
            print(f"  ✓ Saved (F1={best_f1:.4f})")

    # ── Финальные метрики ──
    print(f"\n{'='*50}")
    print(f"DistilBERT Results")
    print(f"{'='*50}")
    print(f"Best F1:        {best_f1:.4f}")
    print(f"Best Accuracy:  {max(history['val_acc']):.4f}")

    metrics = {
        "best_f1": round(best_f1, 4),
        "best_accuracy": round(max(history["val_acc"]), 4),
    }
    with open(save_dir/"metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    # ── Графики ──
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    epochs_x = range(1, len(history["train_loss"]) + 1)

    ax1.plot(epochs_x, history["train_loss"], label="Train")
    ax1.plot(epochs_x, history["val_loss"], label="Val")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title("Loss")
    ax1.legend()
    ax1.grid(True)

    ax2.plot(epochs_x, history["val_acc"], label="Accuracy", marker="o")
    ax2.plot(epochs_x, history["val_f1"], label="F1", marker="s")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Score")
    ax2.set_title("Metrics")
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    plt.savefig(save_dir/ "learning_curves.png", dpi=100)
    plt.close()

    # ── Confusion Matrix ──
    _, _, _, final_preds, final_labels = evaluate(model, test_loader, config.device)
    cm = confusion_matrix(final_labels, final_preds)
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["neg", "pos"])
    ax.set_yticklabels(["neg", "pos"])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion Matrix — DistilBERT")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, cm[i, j], ha="center", va="center",
                    fontsize=16, fontweight="bold",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    plt.colorbar(im, ax=ax)
    plt.tight_layout()
    plt.savefig(save_dir/ "confusion_matrix.png", dpi=100)
    plt.close()

    print(f"\n✓ Artifacts saved to {save_dir}")


if __name__ == "__main__":
    main()
