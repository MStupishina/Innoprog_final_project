from pathlib import Path

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.calibration import CalibrationDisplay
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

def plot_calibration_curve(
        y_true,
        y_proba,
        model_name="model",
        output_dir=None,
        show_plot=True
):
    """Reliability diagram / Calibration curve"""

    fig, ax = plt.subplots(figsize=(7, 5))

    CalibrationDisplay.from_predictions(
        y_true=y_true,
        y_prob=y_proba,
        n_bins=10,
        strategy="quantile",
        ax=ax
    )

    ax.set_title(f"Calibration Curve ({model_name})")

    plt.tight_layout()

    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        save_path = output_dir / f"{model_name}_calibration_curve.png"
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Calibration curve saved to {save_path}")

    if show_plot:
        plt.show()

    plt.close()

# === Regression ===
def plot_model_comparison(
        results: dict,
        output_dir: Path = None,
        show_plot: bool = True
):
    """Сравнение моделей по CV метрикам"""

    model_names = list(results.keys())
    mae_scores = [results[m]["cv"]["summary"]["MAE"] for m in model_names]
    rmse_scores = [results[m]["cv"]["summary"]["RMSE"] for m in model_names]

    x = np.arange(len(model_names))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.bar(x - width / 2, mae_scores, width, label="MAE")
    ax.bar(x + width / 2, rmse_scores, width, label="RMSE")

    ax.set_xticks(x)
    ax.set_xticklabels(model_names)

    ax.set_ylabel("Metric value")
    ax.set_title("Model Comparison (CV Metrics)")
    ax.legend()

    plt.tight_layout()

    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        save_path = output_dir / "model_comparison.png"
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved: {save_path}")

    if show_plot:
        plt.show()

    plt.close()


def plot_cv_boxplot(
        results: dict,
        metric_name: str = "RMSE",
        output_dir: Path = None,
        show_plot: bool = True
):
    """Строит boxplot метрики по CV folds для всех моделей"""
    model_names = []
    metric_values = []
    for model_name, model_results in results.items():
        folds = model_results["cv"]["folds"]
        values = [fold[metric_name] for fold in folds]
        model_names.append(model_name)
        metric_values.append(values)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.boxplot(metric_values, tick_labels=model_names)
    ax.set_title(f"{metric_name} Distribution Across CV Folds")
    ax.set_ylabel(metric_name)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        save_path = output_dir / f"cv_{metric_name.lower()}_boxplot.png"
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    if show_plot:
        plt.show()
    plt.close()


def plot_actual_vs_predicted(
        y_true,
        y_pred,
        model_name="model",
        output_dir=None,
        show_plot=True
):
    """Scatter plot: Actual vs Predicted"""
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(y_true, y_pred, alpha=0.5)
    min_val = min(np.min(y_true), np.min(y_pred))
    max_val = max(np.max(y_true), np.max(y_pred))
    ax.plot([min_val, max_val], [min_val, max_val], linestyle="--", linewidth=2)
    ax.set_xlabel("Actual")
    ax.set_ylabel("Predicted")
    ax.set_title(f"{model_name}: Actual vs Predicted")
    plt.tight_layout()
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        save_path = Path(output_dir) / f"{model_name}_actual_vs_predicted.png"
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    if show_plot:
        plt.show()
    plt.close()


def plot_residuals(
        y_true,
        y_pred,
        model_name="model",
        output_dir=None,
        show_plot=True
):
    """Residual plot"""
    residuals = y_true - y_pred
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(y_pred, residuals, alpha=0.5)
    ax.axhline(0, linestyle="--", linewidth=2)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Residual (Actual - Predicted)")
    ax.set_title(f"{model_name}: Residual Plot")
    plt.tight_layout()
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        save_path = Path(output_dir) / f"{model_name}_residuals.png"
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    if show_plot:
        plt.show()
    plt.close()


def plot_error_by_quantiles(
        y_true,
        y_pred,
        n_quantiles=5,
        model_name="model",
        output_dir=None,
        show_plot=True
):
    """MAE по квантилям таргета"""
    df = pd.DataFrame({
        "actual": y_true,
        "predicted": y_pred
    })
    df["abs_error"] = (df["actual"] - df["predicted"]).abs()
    df["quantile"] = pd.qcut(
        df["actual"],
        q=n_quantiles,
        labels=[f"Q{i + 1}" for i in range(n_quantiles)])
    mae_by_quantile = df.groupby("quantile")["abs_error"].mean()
    fig, ax = plt.subplots(figsize=(8, 6))
    mae_by_quantile.plot(kind="bar", ax=ax)
    ax.set_title(f"{model_name}: MAE by Target Quantile")
    ax.set_xlabel("Target Quantile")
    ax.set_ylabel("Mean Absolute Error")
    plt.tight_layout()
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        save_path = Path(output_dir) / f"{model_name}_quantile_mae.png"
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    if show_plot:
        plt.show()
    plt.close()


def plot_feature_importance(
        model,
        feature_names,
        top_n=15,
        model_name="model",
        output_dir=None,
        show_plot=True
):
    """Top-N feature importance"""
    importance_df = pd.DataFrame({
        "feature": feature_names,
        "importance": model.feature_importances_
    })
    importance_df = (
        importance_df
        .sort_values(
            "importance",
            ascending=False
        )
        .head(top_n)
        .sort_values("importance")
    )
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(importance_df["feature"], importance_df["importance"])
    ax.set_title(f"{model_name}: Feature Importance")
    plt.tight_layout()
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        save_path = Path(output_dir) / f"{model_name}_feature_importance.png"
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    if show_plot:
        plt.show()
    plt.close()


# === Clustering ===
def plot_elbow_method(
        X: np.ndarray,
        output_dir: Path,
        random_state: int,
        n_init: int,
        show_plot=True
):
    inertias = []
    k_range = range(2, 8)
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=random_state, n_init=n_init)
        km.fit(X)
        inertias.append(km.inertia_)
    plt.figure(figsize=(8, 4))
    plt.plot(list(k_range), inertias, marker='o')
    plt.xlabel('Число кластеров')
    plt.ylabel('Inertia')
    plt.title('Elbow Method — выбор числа кластеров')
    plt.xticks(list(k_range))
    plt.grid(True)
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_dir / 'elbow.png', dpi=150, bbox_inches='tight')
    if show_plot:
        plt.show()
    plt.close()

def plot_scatter_2d(
        X_2d: np.ndarray,
        labels: np.ndarray,
        title: str,
        filename: str,
        output_dir: Path,
        show_plot=True
):
    """Рисует и сохраняет scatter plot"""
    plt.figure(figsize=(10, 8))
    sns.scatterplot(
        x=X_2d[:, 0], y=X_2d[:, 1],
        hue=labels,
        palette="tab10" if len(np.unique(labels)) <= 10 else "viridis",
        alpha=0.6,
        legend="full"
    )
    plt.title(title)
    plt.xlabel("Component 1")
    plt.ylabel("Component 2")
    plt.grid(True, alpha=0.3)
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_dir / filename, dpi=150, bbox_inches='tight')
    if show_plot:
        plt.show()
    plt.close()

