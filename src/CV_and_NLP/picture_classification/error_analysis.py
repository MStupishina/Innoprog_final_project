"""
Анализ ошибок ResNet18 для B1 — multilabel classification на PASCAL VOC 2012.

Скрипт:
1. Загружает обученную ResNet18.
2. Загружает test split.
3. Использует threshold из threshold.json.
4. Для каждого изображения сравнивает prediction с ground truth.
5. Находит FP и FN.
6. Для каждого типа ошибки выбирает наиболее показательные
   случаи — с вероятностью максимально близкой к threshold.
7. Сохраняет:
   - общий CSV с результатами;
   - статистику ошибок по классам;
   - наиболее показательные FP;
   - наиболее показательные FN.
"""

import argparse
import csv
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from matplotlib import pyplot as plt

from configs.cv_and_nlp_config import Config
from src.CV_and_NLP.picture_classification.dataset import (
    VOCMultiLabelDataset,
    get_classification_transforms,
)
from src.CV_and_NLP.picture_classification.predictor import (
    load_model,
    load_threshold,
)


# ============================================================
# Ground Truth
# ============================================================

def get_ground_truth(dataset, index):
    """Возвращает список классов ground truth."""
    _, annotation = dataset.voc[index]
    objects = annotation["annotation"].get("object", [])
    if not isinstance(objects, list):
        objects = [objects]
    classes = []
    for obj in objects:
        class_name = obj["name"]
        if class_name in dataset.class_to_idx:
            classes.append(class_name)

    return sorted(set(classes))


# ============================================================
# Prediction
# ============================================================

@torch.no_grad()
def predict_single(
        model,
        image,
        config: Config,
        threshold: float,
):
    """Получает вероятности и предсказанные классы."""
    image = image.unsqueeze(0).to(config.device)
    logits = model(image)
    probs = torch.sigmoid(logits)[0].cpu().numpy()
    predictions = probs >= threshold
    predicted_classes = [
        config.B1["classes"][i]
        for i, pred in enumerate(predictions)
        if pred
    ]

    return probs, predicted_classes


# ============================================================
# Анализ одного изображения
# ============================================================

def analyze_image(
        model,
        dataset,
        index,
        config,
        threshold,
):
    """Анализирует одно изображение."""
    image, _ = dataset[index]
    ground_truth = get_ground_truth(dataset, index)
    probs, predictions = predict_single(
        model=model,
        image=image,
        config=config,
        threshold=threshold,
    )
    ground_truth_set = set(ground_truth)
    prediction_set = set(predictions)
    false_positive = sorted(prediction_set - ground_truth_set)
    false_negative = sorted(ground_truth_set - prediction_set)

    return {
        "ground_truth": ground_truth,
        "predictions": predictions,
        "false_positive": false_positive,
        "false_negative": false_negative,
        "probabilities": probs,
        "threshold": threshold,
    }


# ============================================================
# Поиск наиболее показательных ошибок
# ============================================================

def find_fp_candidates(
        results,
        classes,
        threshold,
):
    """Для каждого FP ищет вероятность ошибочно предсказанного класса,
    максимально близкую к threshold. Чем меньше distance_to_threshold,
    тем более пограничная ошибка.
    """
    candidates = []
    for result in results:
        for class_name in result["false_positive"]:
            class_idx = classes.index(class_name)
            probability = float(result["probabilities"][class_idx])
            distance = abs(probability - threshold)
            candidates.append({
                "result": result,
                "class": class_name,
                "probability": probability,
                "distance": distance,
            })
    candidates.sort(key=lambda x: x["distance"])

    return candidates


def find_fn_candidates(
        results,
        classes,
        threshold,
):
    """Для каждого FN ищет вероятность истинного класса,
    максимально близкую к threshold."""

    candidates = []
    for result in results:
        for class_name in result["false_negative"]:
            class_idx = classes.index(class_name)
            probability = float(result["probabilities"][class_idx])
            distance = abs(probability - threshold)
            candidates.append({
                "result": result,
                "class": class_name,
                "probability": probability,
                "distance": distance,
            })
    candidates.sort(key=lambda x: x["distance"])

    return candidates

def build_error_rows(results, classes, threshold):
    """Формирует запись об ошибке"""
    rows = []
    for result in results:
        image_id = result["image_id"]
        ground_truth = result["ground_truth"]
        predictions = result["predictions"]
        # False Positive
        for class_name in result["false_positive"]:
            class_idx = classes.index(class_name)
            probability = float(result["probabilities"][class_idx])
            rows.append({
                "image_id": image_id,
                "error_type": "FP",
                "error_class": class_name,
                "error_probability": probability,
                "threshold": threshold,
                "distance_to_threshold": abs(probability - threshold),
                "ground_truth": ", ".join(ground_truth),
                "predictions": ", ".join(predictions),
            })
        # False Negative
        for class_name in result["false_negative"]:
            class_idx = classes.index(class_name)
            probability = float(result["probabilities"][class_idx])
            rows.append({
                "image_id": image_id,
                "error_type": "FN",
                "error_class": class_name,
                "error_probability": probability,
                "threshold": threshold,
                "distance_to_threshold": abs(probability - threshold),
                "ground_truth": ", ".join(ground_truth),
                "predictions": ", ".join(predictions),
            })

    return rows
# ============================================================
# Сохранение изображения ошибки
# ============================================================

def save_error_image(
        image_path,
        ground_truth,
        predictions,
        probabilities,
        threshold,
        error_class,
        error_probability,
        distance_to_threshold,
        save_path,
        error_type,
):
    """Сохраняет изображение с информацией об ошибке."""
    image = Image.open(image_path).convert("RGB")
    plt.figure(figsize=(9, 7))
    plt.imshow(image)
    plt.axis("off")
    title_lines = [
        f"Error type: {error_type}",
        f"Error class: {error_class}",
        f"Error probability: {error_probability:.3f}",
        f"Threshold: {threshold:.3f}",
        f"Distance: {distance_to_threshold:.3f}",
        "",
        f"Ground Truth: "
        f"{', '.join(ground_truth) if ground_truth else 'none'}",
        f"Prediction: "
        f"{', '.join(predictions) if predictions else 'none'}",
    ]

    plt.title("\n".join(title_lines), fontsize=10)
    plt.tight_layout()
    plt.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close()


# ============================================================
# CSV с результатами
# ============================================================

def save_results(results, save_path):
    """Сохраняет результаты анализа всех изображений."""

    fieldnames = [
        "index",
        "image",
        "ground_truth",
        "predictions",
        "false_positive",
        "false_negative",
    ]

    with open(
            save_path,
            "w",
            newline="",
            encoding="utf-8",
    ) as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for result in results:
            writer.writerow({
                "index": result["index"],
                "image": result["image"],
                "ground_truth": ", ".join(result["ground_truth"]),
                "predictions": ", ".join(result["predictions"]),
                "false_positive": ", ".join(result["false_positive"]),
                "false_negative": ", ".join(result["false_negative"]),
            })


# ============================================================
# Статистика по классам
# ============================================================

def calculate_class_statistics(
        results,
        classes,
):
    """Считает TP / FP / FN для каждого класса."""

    statistics = {}
    for class_name in classes:
        tp = 0
        fp = 0
        fn = 0
        for result in results:
            ground_truth = set(result["ground_truth"])
            predictions = set(result["predictions"])
            if (class_name in ground_truth and class_name in predictions):
                tp += 1
            elif (class_name not in ground_truth and class_name in predictions):
                fp += 1
            elif (class_name in ground_truth and class_name not in predictions):
                fn += 1
        precision = tp / (tp + fp) if tp + fp > 0 else 0.0
        recall = tp / (tp + fn) if tp + fn > 0 else 0.0
        statistics[class_name] = {
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "precision": precision,
            "recall": recall,
        }

    return statistics


def save_class_statistics(
        statistics,
        save_path,
):
    """Сохраняет статистику по классам."""

    fieldnames = [
        "class",
        "tp",
        "fp",
        "fn",
        "precision",
        "recall",
    ]

    with open(
            save_path,
            "w",
            newline="",
            encoding="utf-8",
    ) as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for class_name, values in statistics.items():
            writer.writerow({
                "class": class_name,
                "tp": values["tp"],
                "fp": values["fp"],
                "fn": values["fn"],
                "precision": round(values["precision"], 4),
                "recall": round(values["recall"], 4),
            })


# ============================================================
# Основной анализ
# ============================================================

def run_error_analysis(
        config: Config,
        model_path,
        max_examples=10,
):
    """Полный анализ ошибок на test set."""
    print("=" * 60)
    print("B1 — Error Analysis")
    print("=" * 60)
    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------
    print(f"\nModel: {model_path}")
    model = load_model(config=config, model_path=model_path)
    threshold = load_threshold(config=config, model_path=model_path)
    print(f"Threshold: {threshold:.3f}")
    # --------------------------------------------------------
    # Test split
    # --------------------------------------------------------
    split_dir = (config.artifacts_B1 / "splits")
    test_idx_path = (split_dir / "test_idx.npy")
    if not test_idx_path.exists():
        raise FileNotFoundError(
            f"Не найден test split: "
            f"{test_idx_path}"
        )
    test_indices = np.load(test_idx_path)
    dataset = VOCMultiLabelDataset(
        config=config,
        root=config.voc_dir,
        image_set="trainval",
        transform=get_classification_transforms(train=False, config=config),
    )
    # --------------------------------------------------------
    # Output
    # --------------------------------------------------------
    save_dir = config.artifacts_B1 / "resnet18" / "error_analysis"
    fp_dir = save_dir / "false_positives"
    fn_dir = save_dir / "false_negatives"
    fp_dir.mkdir(parents=True,exist_ok=True)
    fn_dir.mkdir(parents=True,exist_ok=True)
    # --------------------------------------------------------
    # Analyze test set
    # --------------------------------------------------------
    results = []
    print(f"\nTest images: {len(test_indices)}")
    print("Analyzing...")
    for position, dataset_index in enumerate(test_indices):
        dataset_index = int(dataset_index)
        result = analyze_image(
            model=model,
            dataset=dataset,
            index=dataset_index,
            config=config,
            threshold=threshold,
        )
        image_id = dataset.voc.ids[dataset_index]
        result["index"] = dataset_index
        result["image"] = image_id
        results.append(result)
        if (position + 1) % 100 == 0:
            print(f"Processed: {position + 1}/{len(test_indices)}")
    # --------------------------------------------------------
    # Save all results
    # --------------------------------------------------------
    results_path = save_dir / "error_analysis.csv"
    save_results(
        results=results,
        save_path=results_path,
    )
    error_rows = build_error_rows(
        results=results,
        classes=config.B1["classes"],
        threshold=threshold,
    )
    error_df = pd.DataFrame(error_rows)
    error_csv = save_dir / "error_details.csv"
    error_df.to_csv(error_csv, index=False, encoding="utf-8-sig")
    # --------------------------------------------------------
    # Class statistics
    # --------------------------------------------------------
    statistics = calculate_class_statistics(
        results=results,
        classes=config.B1["classes"],
    )
    statistics_path = save_dir/ "class_error_summary.csv"
    save_class_statistics(
        statistics=statistics,
        save_path=statistics_path,
    )
    # --------------------------------------------------------
    # Find representative FP
    # --------------------------------------------------------
    fp_candidates = find_fp_candidates(
        results=results,
        classes=config.B1["classes"],
        threshold=threshold,
    )
    # --------------------------------------------------------
    # Find representative FN
    # --------------------------------------------------------
    fn_candidates = find_fn_candidates(
        results=results,
        classes=config.B1["classes"],
        threshold=threshold,
    )
    # --------------------------------------------------------
    # Save representative FP
    # --------------------------------------------------------
    saved_fp = set()
    for candidate in fp_candidates:
        if len(saved_fp) >= max_examples:
            break
        result = candidate["result"]
        image_id = result["image"]
        # Не сохраняем одно и то же изображение
        # несколько раз, если на нём несколько FP.
        if image_id in saved_fp:
            continue
        saved_fp.add(image_id)
        rank = len(saved_fp)
        image_path = (
                Path(dataset.voc.root)
                / "VOCdevkit"
                / "VOC2012"
                / "JPEGImages"
                / f"{image_id}.jpg"
        )
        save_path = fp_dir / f"{rank:02d}_{image_id}_FP.jpg"
        save_error_image(
            image_path=image_path,
            ground_truth=result["ground_truth"],
            predictions=result["predictions"],
            probabilities=result["probabilities"],
            threshold=threshold,
            error_class=candidate["class"],
            error_probability=candidate["probability"],
            distance_to_threshold=candidate["distance"],
            save_path=save_path,
            error_type="False Positive",
        )
    # --------------------------------------------------------
    # Save representative FN
    # --------------------------------------------------------
    saved_fn = set()
    for candidate in fn_candidates:
        if len(saved_fn) >= max_examples:
            break
        result = candidate["result"]
        image_id = result["image"]
        if image_id in saved_fn:
            continue
        saved_fn.add(image_id)
        rank = len(saved_fn)
        image_path = (
                Path(dataset.voc.root)
                / "VOCdevkit"
                / "VOC2012"
                / "JPEGImages"
                / f"{image_id}.jpg"
        )
        save_path = fn_dir/ f"{rank:02d}_{image_id}_FN.jpg"
        save_error_image(
            image_path=image_path,
            ground_truth=result["ground_truth"],
            predictions=result["predictions"],
            probabilities=result["probabilities"],
            threshold=threshold,
            error_class=candidate["class"],
            error_probability=candidate["probability"],
            distance_to_threshold=candidate["distance"],
            save_path=save_path,
            error_type="False Negative",
        )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    total_fp_images = sum(bool(r["false_positive"]) for r in results)
    total_fn_images = sum(bool(r["false_negative"]) for r in results)
    exact_matches = sum(
        not r["false_positive"]
        and not r["false_negative"]
        for r in results
    )
    print("\n" + "=" * 60)
    print("Error analysis completed")
    print("=" * 60)
    print(f"Test images: {len(results)}")
    print(f"Images with FP: {total_fp_images}")
    print(f"Images with FN: {total_fn_images}")
    print(f"Images without FP/FN: {exact_matches}")
    print(f"\nRepresentative FP saved: {len(saved_fp)}")
    print(f"Representative FN saved: {len(saved_fn)}")
    print("\nSaved:")
    print(f"- {results_path}")
    print(f"- {error_csv}")
    print(f"- {statistics_path}")
    print(f"- {fp_dir}")
    print(f"- {fn_dir}")
# ============================================================
# CLI
# ============================================================
def main():
    parser = argparse.ArgumentParser(
        description="Анализ ошибок для ResNet18"
    )
    parser.add_argument(
        "--model",
        default=(
            "artifacts/cv_classification/"
            "resnet18/best_model.pt"
        ),
        help="Path to model checkpoint",
    )
    parser.add_argument(
        "--max-examples",
        type=int,
        default=10,
        help=(
            "Number of representative "
            "FP/FN images to save"
        ),
    )
    args = parser.parse_args()
    config = Config()
    run_error_analysis(
        config=config,
        model_path=args.model,
        max_examples=args.max_examples,
    )

if __name__ == "__main__":
    main()
