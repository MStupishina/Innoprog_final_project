from pathlib import Path

import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import (precision_recall_curve, confusion_matrix, recall_score, precision_score, f1_score,
                             average_precision_score, roc_curve, auc)


# === Classification ===
def plot_roc_curve(y_true, y_proba, model_name="models", output_dir=None, show_plot=True):
    """ROC-кривая с AUC"""
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    roc_auc = auc(fpr, tpr)

    fig, ax = plt.subplots(figsize=(7, 5))

    ax.plot(fpr, tpr, linewidth=2, label=f"AUC = {roc_auc:.3f}")
    ax.plot([0, 1], [0, 1], linestyle="--", color="grey")

    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(f"ROC Curve ({model_name})")

    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        path = output_dir / f"{model_name}_roc_curve.png"
        plt.savefig(path, dpi=150, bbox_inches="tight")
        print(f"ROC curve saved to {path}")

    if show_plot:
        plt.show()
    plt.close(fig)


def plot_pr_curve(y_true, y_proba, model_name="models", output_dir=None, show_plot=True):
    """Precision-Recall кривая с PR-AUC"""

    precision, recall, _ = precision_recall_curve(y_true, y_proba)
    pr_auc = average_precision_score(y_true, y_proba)

    fig, ax = plt.subplots(figsize=(7, 5))

    ax.plot(
        recall,
        precision,
        linewidth=2,
        label=f"PR-AUC = {pr_auc:.3f}"
    )

    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(f"Precision-Recall Curve ({model_name})")

    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        path = output_dir / f"{model_name}_pr_curve.png"
        plt.savefig(path, dpi=150, bbox_inches="tight")
        print(f"PR curve saved to {path}")

    if show_plot:
        plt.show()
    plt.close(fig)


def plot_confusion_matrix(y_true, y_pred, model_name="models", class_names=None, output_dir=None, show_plot=True):
    """Confusion matrix в виде heatmap"""
    if class_names is None:
        class_names = ["Class 0", "Class 1"]
    cm = confusion_matrix(y_true, y_pred)

    fig, ax = plt.subplots(figsize=(5, 4))

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        ax=ax
    )

    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(f"Confusion Matrix ({model_name})")

    plt.tight_layout()

    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        path = output_dir / f"{model_name}_confusion_matrix.png"
        plt.savefig(path, dpi=150, bbox_inches="tight")
        print(f"Confusion matrix saved to {path}")

    if show_plot:
        plt.show()
    plt.close(fig)


def plot_threshold_metrics(y_true, y_proba, model_name="models", output_dir=None,
                           thresholds=None, show_plot=True):
    """
    Визуализирует FP, FN, Precision, Recall и F1 по разным порогам.
    Сохраняет график в output_dir, если указан.
    Использует две оси Y: левую для счетчиков (FP/FN), правую для долей (0-1).
    """
    if thresholds is None:
        thresholds = np.linspace(0, 1, 101)

    fps, fns, precisions, recalls, f1s = [], [], [], [], []

    for t in thresholds:
        y_pred = (y_proba >= t).astype(int)
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
        fp = cm[0, 1]
        fn = cm[1, 0]
        fps.append(fp)
        fns.append(fn)
        precisions.append(precision_score(y_true, y_pred, zero_division=0))
        recalls.append(recall_score(y_true, y_pred, zero_division=0))
        f1s.append(f1_score(y_true, y_pred, zero_division=0))

    best_idx = np.argmax(f1s)
    best_threshold = thresholds[best_idx]

    fig, ax1 = plt.subplots(figsize=(8, 6))

    # Левая ось Y (Количество ошибок FP / FN)
    color1 = 'tab:red'
    color2 = 'tab:blue'
    ax1.set_xlabel("Threshold")
    ax1.set_ylabel("Count (FP / FN)", color='black')
    l1, = ax1.plot(thresholds, fps, label="False Positives", color=color1, linewidth=2)
    l2, = ax1.plot(thresholds, fns, label="False Negatives", color=color2, linewidth=2)
    ax1.tick_params(axis='y', labelcolor='black')
    ax1.grid(True, alpha=0.3)

    # Правая ось Y (Метрики 0.0 - 1.0)
    ax2 = ax1.twinx()
    color3 = 'tab:green'
    color4 = 'tab:orange'
    color5 = 'tab:purple'
    ax2.set_ylabel("Score (Precision / Recall / F1)", color='black')
    l3, = ax2.plot(thresholds, precisions, label="Precision", color=color3, linestyle="--", linewidth=2)
    l4, = ax2.plot(thresholds, recalls, label="Recall", color=color4, linestyle="--", linewidth=2)
    l5, = ax2.plot(thresholds, f1s, label="F1 Score", color=color5, linestyle="-.", linewidth=2.5)
    ax2.tick_params(axis='y', labelcolor='black')
    ax2.set_ylim(-0.05, 1.05)  # Фиксируем шкалу от 0 до 1

    # Линия лучшего порога
    l6 = ax1.axvline(x=best_threshold, color="black", linestyle=":", linewidth=2,
                     label=f"Best F1 ({f1s[best_idx]:.2f}) at t={best_threshold:.2f}")

    # Собираем легенду с обеих осей
    lines = [l1, l2, l3, l4, l5, l6]
    labels = [l.get_label() for l in lines]
    ax1.legend(
        lines,
        labels,
        loc='upper center',
        bbox_to_anchor=(0.5, -0.12),
        ncol=3,
        frameon=True
    )

    ax1.set_title(f"Metrics vs Threshold ({model_name})")
    plt.tight_layout()
    # Сохраняем график, если указан output_dir
    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True, parents=True)
        file_path = output_dir / f"{model_name}_threshold_metrics.png"
        plt.savefig(file_path, dpi=150, bbox_inches='tight')
        print(f"Threshold metrics plot saved to {file_path}")

    if show_plot:
        plt.show()
    plt.close(fig)
    return {
        "thresholds": thresholds,
        "fps": fps,
        "fns": fns,
        "precisions": precisions,
        "recalls": recalls,
        "f1s": f1s,
        "best_threshold": best_threshold,
    }

# === Regression ===
