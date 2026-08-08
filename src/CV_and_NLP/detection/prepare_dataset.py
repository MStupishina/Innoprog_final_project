import xml.etree.ElementTree as ET
from shutil import copy2

from tqdm import tqdm

from configs.cv_and_nlp_config import Config

def convert_bbox_voc_to_yolo(xmin, xmax, ymin, ymax, img_w, img_h):
    """
    VOC → YOLO: абсолютные пиксели → нормализованные [0, 1].
    Возвращает (cx, cy, width, height) — нормализованные.
    """
    cx = (xmin + xmax) / 2.0 / img_w  # центр X
    cy = (ymin + ymax) / 2.0 / img_h  # центр Y
    w = (xmax - xmin) / img_w  # ширина
    h = (ymax - ymin) / img_h  # высота
    return cx, cy, w, h


def parse_voc_xml(xml_path, class_to_idx):
    """
    Извлекает из VOC XML: размер изображения + список аннотаций.
    Возвращает (img_w, img_h, [{"class_id": int, "bbox": (cx,cy,w,h)}, ...]).
    Пропускает difficult=1 объекты.
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()

    # Размер изображения
    size = root.find("size")
    if size is None:
        raise ValueError(f"Missing <size> in {xml_path}")
    width = size.find("width")
    height = size.find("height")
    if width is None or height is None:
        raise ValueError(f"Missing image dimensions in {xml_path}")
    img_w = int(width.text)
    img_h = int(height.text)

    annotations = []
    for obj in root.findall("object"):
        # Пропускаем difficult объекты
        difficult = obj.find("difficult")
        if difficult is not None and int(difficult.text) == 1:
            continue

        class_name = obj.find("name").text
        if class_name not in class_to_idx:
            continue  # пропускаем классы вне 20

        bbox = obj.find("bndbox")
        xmin = float(bbox.find("xmin").text)
        xmax = float(bbox.find("xmax").text)
        ymin = float(bbox.find("ymin").text)
        ymax = float(bbox.find("ymax").text)

        xmin = max(0.0, xmin)
        ymin = max(0.0, ymin)
        xmax = min(float(img_w), xmax)
        ymax = min(float(img_h), ymax)
        if xmax <= xmin or ymax <= ymin:
            continue

        cx, cy, w, h = convert_bbox_voc_to_yolo(xmin, xmax, ymin, ymax, img_w, img_h)

        # Проверка: координаты должны быть в [0, 1]
        if not (0 <= cx <= 1 and 0 <= cy <= 1 and 0 < w <= 1 and 0 < h <= 1):
            continue

        annotations.append({
            "class_id": class_to_idx[class_name],
            "bbox": (cx, cy, w, h),
        })

    return annotations


def process_image_set(config: Config, image_set_name: str):
    """
    Обрабатывает один image set (train или val).
    Читает список имён файлов из ImageSets/Main/{name}.txt
    и создаёт .txt аннотации в YOLO-формате.
    """
    class_to_idx = {
        class_name: idx
        for idx, class_name in enumerate(config.VOC_CLASSES)
    }
    # Используем официальный VOC train для обучения и официальный VOC val для валидации/финальной оценки.
    # Читаем список имён
    image_set_path = (config.voc_dir / "ImageSets" / "Main" / f"{image_set_name}.txt")
    if not image_set_path.exists():
        print(f"⚠ Image set not found: {image_set_path}, skipping.")
        return

    with open(image_set_path, encoding="utf-8") as f:
        image_ids = [line.strip() for line in f if line.strip()]

    # Создаём выходные папки
    labels_dir = config.artifacts_B2 / "voc_yolo" / "labels" / image_set_name
    labels_dir.mkdir(parents=True, exist_ok=True)

    ann_dir = config.voc_dir / "Annotations"
    skipped_empty = 0
    total_objects = 0

    for img_id in tqdm(image_ids, desc=f"Converting {image_set_name}"):
        xml_path = ann_dir / f"{img_id}.xml"
        if not xml_path.exists():
            continue

        annotations = parse_voc_xml(xml_path, class_to_idx)

        # Пишем .txt файл (даже пустой — изображение без объектов)
        # YOLO ожидает пустой файл для изображений без аннотаций
        txt_path = labels_dir / f"{img_id}.txt"
        with open(txt_path, "w", encoding="utf-8") as f:
            for ann in annotations:
                class_id = ann["class_id"]
                cx, cy, w, h = ann["bbox"]
                f.write(f"{class_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")
                total_objects += 1

        if not annotations:
            skipped_empty += 1

    print(f"  Images: {len(image_ids)}, Objects: {total_objects}, "
          f"Empty: {skipped_empty}")


def prepare_images(config: Config, image_set_name: str):
    ids_file = (config.voc_dir / f"ImageSets/Main/{image_set_name}.txt")
    with open(ids_file, encoding="utf-8") as f:
        image_ids = [x.strip() for x in f if x.strip()]
    split = image_set_name

    out_dir = config.artifacts_B2 / "voc_yolo/images" / split
    out_dir.mkdir(parents=True, exist_ok=True)

    for img_id in tqdm(image_ids):
        src = (config.voc_dir / f"JPEGImages/{img_id}.jpg")
        dst = out_dir / f"{img_id}.jpg"
        if not src.exists():
            print(f"⚠ Image not found: {src}")
            continue
        copy2(src, dst)

def create_yaml(config):
    """Создаёт voc.yaml — конфиг датасета для YOLOv8"""
    yolo_dir = config.artifacts_B2 / "voc_yolo"
    yaml_path = config.artifacts_B2 / "voc.yaml"
    classes = config.VOC_CLASSES
    yaml_content = f"""path: {yolo_dir}
train: images/train
val: images/val
nc: {len(classes)}
names:
"""
    for i, name in enumerate(classes):
        yaml_content += f"  {i}: {name}\n"
    with open(yaml_path, "w", encoding="utf-8") as f:
        f.write(yaml_content)


def main():
    config = Config()
    print("VOC:", config.voc_dir)
    print("Output:", config.artifacts_B2)

    # Конвертируем train и val
    process_image_set(config, "train")
    process_image_set(config, "val")

    prepare_images(config, "train")
    prepare_images(config, "val")

    create_yaml(config)
    print("\nКонвертация выполнена")


if __name__ == "__main__":
    main()
