import json
import shutil

from ultralytics import YOLO

from configs.cv_and_nlp_config import Config

def main():
    config = Config()
    voc_yaml = config.artifacts_B2 / "voc.yaml"
    best_pt = config.artifacts_B2 / "yolov8_voc" / "weights" / "best.pt"
    artifact_best_pt = config.artifacts_B2 / "best.pt"
    metrics_json = config.artifacts_B2 / "metrics.json"

    print(f"Device: {config.device}")
    print(f"Config: {voc_yaml}")
    if not voc_yaml.exists():
        raise FileNotFoundError(
            f"Dataset YAML not found: {voc_yaml}. "
            "Run prepare_dataset.py first."
        )

    # Загружаем предобученную nano-модель
    # YOLOv8n — самая лёгкая, быстрая, идеально для прототипа
    model = YOLO(config.B2["model"])

    # Обучаем
    model.train(
        data=voc_yaml,
        imgsz=config.B2["imgsz"],
        batch=config.B2["batch"],
        epochs=config.B2["epochs"],
        lr0=config.B2["lr0"],
        lrf=config.B2["lrf"],
        patience=config.B2["patience"],
        device=config.device,
        project=str(config.artifacts_B2),
        name="yolov8_voc",
        exist_ok=True,           # перезаписываем предыдущий запуск
        seed=config.seed,
        verbose=True,
    )

    if not best_pt.exists():
        raise FileNotFoundError(
            f"Best model not found: {best_pt}"
        )
    shutil.copy2(best_pt, artifact_best_pt)
    print("best.pt сохранены")
    best_model = YOLO(best_pt)

    metrics = best_model.val(
        data=voc_yaml,
        split="val",
        imgsz=config.B2["imgsz"],
        batch=config.B2["batch"],
        device=config.device,
    )
    map50 = float(metrics.box.map50)
    map50_95 = float(metrics.box.map)
    metrics_data = {
        "model": config.B2["model"],
        "imgsz": config.B2["imgsz"],
        "epochs": config.B2["epochs"],
        "batch": config.B2["batch"],
        "mAP@0.5": map50,
        "mAP@0.5:0.95": map50_95,
        "best_model": str(artifact_best_pt),
        "split": "val",
        "confidence_threshold": config.B2["conf_threshold"],
        "iou_threshold": config.B2["iou_threshold"],
    }

    with open(metrics_json, "w", encoding="utf-8") as f:
        json.dump(metrics_data, f, indent=4)
    print(f"Metrics saved to: {metrics_json}")
    print(f"mAP@0.5: {map50:.4f}")
    print(f"mAP@0.5:0.95: {map50_95:.4f}")


if __name__ == "__main__":
    main()
