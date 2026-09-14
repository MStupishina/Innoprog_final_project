import cv2
from ultralytics import YOLO

from configs.cv_and_nlp_config import Config

from src.CV_and_NLP.detection.error_analysis import (
    draw_results,
    load_ground_truth,
    match_predictions_to_ground_truth,
    save_error_summary,
)


def get_predictions(result):
    """Конвертирует результаты Ultralytics в список."""
    predictions = []
    if result.boxes is None:
        return predictions

    boxes = result.boxes.xyxy.cpu().numpy()
    classes = result.boxes.cls.cpu().numpy().astype(int)
    confidences = result.boxes.conf.cpu().numpy()

    for box, class_id, confidence in zip(
            boxes,
            classes,
            confidences,
    ):
        predictions.append(
            {
                "class_id": int(class_id),
                "bbox": box.tolist(),
                "confidence": float(confidence),
            }
        )

    return predictions


def run_inference(model, config):
    """Инференс YOLO на изображениях validation."""

    val_images_dir = config.artifacts_B2 / "voc_yolo" / "images" / "val"
    output_dir = config.artifacts_B2 / "predictions" / "all"
    output_dir.mkdir(parents=True, exist_ok=True)

    model.predict(
        source=str(val_images_dir),
        imgsz=config.B2["imgsz"],
        conf=config.B2["conf_threshold"],
        iou=config.B2["iou_threshold"],
        device=config.device,
        save=True,
        project=str(config.artifacts_B2 / "predictions"),
        name="all",
        exist_ok=True,
        verbose=False,
    )

    print(f"Predictions saved to: {output_dir}")


def analyze_validation_errors(
        model,
        config,
        max_examples_per_category=5,
        match_iou=0.5,
):
    """Находит и сохраняет примеры FP/FN."""

    val_images_dir = config.artifacts_B2 / "voc_yolo" / "images" / "val"
    annotations_dir = config.voc_dir / "VOCdevkit" / "VOC2012" / "Annotations"
    output_dir = config.artifacts_B2 / "predictions"

    categories = (
        "correct",
        "false_positive",
        "false_negative",
        "mixed_errors",
    )

    for category in categories:
        output_dir / category.mkdir(parents=True, exist_ok=True)

    class_to_idx = {name: index for index, name in enumerate(config.VOC_CLASSES)}
    image_paths = sorted(val_images_dir.glob("*.jpg"))
    counts = {category: 0 for category in categories}
    saved_examples = {category: 0 for category in categories}

    total_tp = 0
    total_fp = 0
    total_fn = 0

    for image_path in image_paths:
        xml_path = annotations_dir / f"{image_path.stem}.xml"
        if not xml_path.exists():
            continue
        image = cv2.imread(str(image_path))
        if image is None:
            continue
        ground_truth = load_ground_truth(xml_path, class_to_idx)

        results = model.predict(
            source=str(image_path),
            imgsz=config.B2["imgsz"],
            conf=config.B2["conf_threshold"],
            iou=config.B2["iou_threshold"],
            device=config.device,
            verbose=False,
        )

        predictions = get_predictions(results[0])
        tp, fp, fn = (
            match_predictions_to_ground_truth(
                predictions,
                ground_truth,
                iou_threshold=match_iou,
            )
        )
        total_tp += len(tp)
        total_fp += len(fp)
        total_fn += len(fn)
        has_fp = len(fp) > 0
        has_fn = len(fn) > 0

        if has_fp and has_fn:
            category = "mixed_errors"
        elif has_fp:
            category = "false_positive"
        elif has_fn:
            category = "false_negative"
        else:
            category = "correct"

        counts[category] += 1

        if saved_examples[category] < max_examples_per_category:
            annotated = draw_results(
                image=image,
                predictions=predictions,
                ground_truth=ground_truth,
                false_positives=fp,
                false_negatives=fn,
                class_names=config.VOC_CLASSES,
            )

            output_path = output_dir / category / image_path.name
            cv2.imwrite(str(output_path), annotated)
            saved_examples[category] += 1

        if all(saved_examples[category] >= max_examples_per_category for category in categories):
            break

    summary_path = save_error_summary(
        output_dir=output_dir,
        counts=counts,
        total_tp=total_tp,
        total_fp=total_fp,
        total_fn=total_fn,
        match_iou=match_iou,
        confidence_threshold=config.B2[
            "conf_threshold"
        ],
    )

    print("\n=== Error analysis ===")
    print(f"Correct: {counts['correct']}")
    print(f"False Positive: {counts['false_positive']}")
    print(f"False Negative: {counts['false_negative']}")
    print(f"Mixed errors: {counts['mixed_errors']}")
    print(
        f"TP: {total_tp} | "
        f"FP: {total_fp} | "
        f"FN: {total_fn}"
    )

    print(f"Error summary saved to: {summary_path}")


def main():
    config = Config()

    model_path = config.artifacts_B2 / "best.pt"

    if not model_path.exists():
        raise FileNotFoundError(
            f"Model not found: {model_path}. "
            "Run train.py first."
        )

    print(f"Device: {config.device}")
    print(f"Model: {model_path}")

    model = YOLO(str(model_path))
    run_inference(model,config)
    analyze_validation_errors(
        model,
        config,
        max_examples_per_category=5,
        match_iou=0.5,
    )


if __name__ == "__main__":
    main()