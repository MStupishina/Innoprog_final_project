import json
import xml.etree.ElementTree as ET
import cv2


def calculate_iou(box1, box2):
    """Расчет IoU в формате xyxy."""

    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    intersection = (max(0.0, x2 - x1) * max(0.0, y2 - y1))
    area1 = (max(0.0, box1[2] - box1[0]) * max(0.0, box1[3] - box1[1]))
    area2 = (max(0.0, box2[2] - box2[0]) * max(0.0, box2[3] - box2[1]))
    union = area1 + area2 - intersection
    if union <= 0:
        return 0.0
    return intersection / union


def load_ground_truth(xml_path, class_to_idx):
    """Загрузка рамок из VOC XML."""

    tree = ET.parse(xml_path)
    root = tree.getroot()
    ground_truth = []
    for obj in root.findall("object"):
        difficult = obj.find("difficult")
        if difficult is not None and int(difficult.text) == 1:
            continue
        name_node = obj.find("name")

        if name_node is None:
            continue
        class_name = name_node.text
        if class_name not in class_to_idx:
            continue
        bbox = obj.find("bndbox")
        if bbox is None:
            continue
        ground_truth.append(
            {
                "class_id": class_to_idx[class_name],
                "bbox": [
                    float(bbox.find("xmin").text),
                    float(bbox.find("ymin").text),
                    float(bbox.find("xmax").text),
                    float(bbox.find("ymax").text),
                ],
            }
        )

    return ground_truth


def match_predictions_to_ground_truth(
        predictions,
        ground_truth,
        iou_threshold=0.5,
):
    """Сравнивает predictions с реалтными рамками.
    Returns:
        true_positives,
        false_positives,
        false_negatives
    """

    predictions = sorted(
        predictions,
        key=lambda x: x["confidence"],
        reverse=True,
    )
    matched_gt = set()
    true_positives = []
    false_positives = []

    for prediction in predictions:
        best_iou = 0.0
        best_gt_index = None

        for gt_index, gt in enumerate(ground_truth):
            if gt_index in matched_gt:
                continue
            if prediction["class_id"] != gt["class_id"]:
                continue
            iou = calculate_iou(
                prediction["bbox"],
                gt["bbox"],
            )
            if iou > best_iou:
                best_iou = iou
                best_gt_index = gt_index
        if (
                best_gt_index is not None
                and best_iou >= iou_threshold
        ):
            matched_gt.add(best_gt_index)
            true_positives.append(
                {
                    "prediction": prediction,
                    "ground_truth": ground_truth[best_gt_index],
                    "iou": best_iou,
                }
            )
        else:
            false_positives.append(prediction)
    false_negatives = [
        gt
        for index, gt in enumerate(ground_truth)
        if index not in matched_gt
    ]

    return (
        true_positives,
        false_positives,
        false_negatives,
    )


def draw_results(
        image,
        predictions,
        ground_truth,
        false_positives,
        false_negatives,
        class_names,
):
    """Отрисовка GT, предсказаний, FP и FN на изображении."""
    output = image.copy()

    # Ground truth
    for gt in ground_truth:
        x1, y1, x2, y2 = map(int, gt["bbox"])
        class_name = class_names[gt["class_id"]]
        cv2.rectangle(
            output,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            2,
        )

        cv2.putText(
            output,
            f"GT: {class_name}",
            (x1, max(15, y1 - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            2,
        )

    # Predictions
    for prediction in predictions:
        x1, y1, x2, y2 = map(int, prediction["bbox"])
        class_name = class_names[prediction["class_id"]]
        confidence = prediction["confidence"]
        is_fp = any(
            prediction is fp
            for fp in false_positives
        )
        if is_fp:
            color = (0, 0, 255)
            prefix = "FP"
        else:
            color = (255, 0, 0)
            prefix = "Pred"

        cv2.rectangle(
            output,
            (x1, y1),
            (x2, y2),
            color,
            2,
        )

        cv2.putText(
            output,
            f"{prefix}: {class_name} {confidence:.2f}",
            (x1, min(output.shape[0] - 5, y2 + 18)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            2,
        )

    # False negatives
    for fn in false_negatives:
        x1, y1, x2, y2 = map(int, fn["bbox"])
        class_name = class_names[fn["class_id"]]
        cv2.rectangle(
            output,
            (x1, y1),
            (x2, y2),
            (0, 165, 255),
            3,
        )
        cv2.putText(
            output,
            f"FN: {class_name}",
            (x1, max(15, y1 - 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 165, 255),
            2,
        )

    return output


def save_error_summary(
        output_dir,
        counts,
        total_tp,
        total_fp,
        total_fn,
        match_iou,
        confidence_threshold,
):
    """Сохраняет данные об ошибках."""

    summary = {
        "match_iou": match_iou,
        "confidence_threshold": confidence_threshold,
        "images_by_category": counts,
        "total_true_positives": total_tp,
        "total_false_positives": total_fp,
        "total_false_negatives": total_fn,
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = (output_dir / "error_summary.json")

    with open(summary_path, "w", encoding="utf-8", ) as file:
        json.dump(summary, file, indent=4, ensure_ascii=False)

    return summary_path
