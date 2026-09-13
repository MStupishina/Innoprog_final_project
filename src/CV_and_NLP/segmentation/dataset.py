# VOC цветовая палитра (индекс → RGB-цвет)
# Пиксель с этим цветом → метка класса
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.transforms import functional as TF

from configs.cv_and_nlp_config import Config

VOC_PALETTE = [
    (0, 0, 0),  # 0:  background
    (128, 0, 0),  # 1:  aeroplane
    (0, 128, 0),  # 2:  bicycle
    (128, 128, 0),  # 3:  bird
    (0, 0, 128),  # 4:  boat
    (128, 0, 128),  # 5:  bottle
    (0, 128, 128),  # 6:  bus
    (128, 128, 128),  # 7:  car
    (64, 0, 0),  # 8:  cat
    (192, 0, 0),  # 9:  chair
    (64, 128, 0),  # 10: cow
    (192, 128, 0),  # 11: diningtable
    (64, 0, 128),  # 12: dog
    (192, 0, 128),  # 13: horse
    (64, 128, 128),  # 14: motorbike
    (192, 128, 128),  # 15: person
    (0, 64, 0),  # 16: pottedplant
    (128, 64, 0),  # 17: sheep
    (0, 192, 0),  # 18: sofa
    (128, 192, 0),  # 19: train
    (0, 64, 128),  # 20: tvmonitor
]


# Белый (224, 224, 192) → граница объекта → метка 255 (игнорируем)


def mask_to_class(mask_rgb: np.ndarray) -> np.ndarray:
    """
    Преобразует RGB-маску (H×W×3) в классовую (H×W).
    Использует векторизацию numpy (быстро):
    для каждого цвета сравниваем весь массив и присваиваем индекс класса.
    """
    height, width, _ = mask_rgb.shape
    class_mask = np.full((height, width), full_value=Config.B3["ignore_index"], dtype=np.int64)

    for class_idx, color in enumerate(VOC_PALETTE):
        # Сравниваем каждый пиксель с цветом (векторизованно)
        color = np.asarray(color, dtype=np.uint8)
        matches = np.all(mask_rgb == color, axis=-1)
        class_mask[matches] = class_idx

    # Пиксели, которые не совпали ни с одним цветом → 255 (границы)
    # Так мы не теряем "белые" пиксели — они просто не матчатся с палитрой
    return class_mask


class VOCSegmentationDataset(Dataset):
    """
    VOC 2012 сегментация.
    Возвращает (image, mask) где:
    - image: тензор [3, H, W] нормализованный
    - mask:  тензор [H, W] с метками классов (long)
    """

    def __init__(
            self, root: str,
            image_set: str = "train",
            img_size: int = None,
            augment: bool = False
    ):
        self.root = Path(root or Config.voc_dir)
        self.image_set = image_set
        self.img_size = img_size or Config.B3["img_size"]
        self.augment = augment

        self.jpeg_dir = self.root / "JPEGImages"
        self.mask_dir = self.root / "SegmentationClass"

        # Читаем список файлов
        image_set_path = (self.root / "ImageSets" / "Segmentation" / f"{image_set}.txt")
        if not image_set_path.exists():
            raise FileNotFoundError(f"Не найден файл split: {image_set_path}")

        with open(image_set_path, "r", encoding="utf-8") as f:
            self.image_ids = [line.strip() for line in f if line.strip()]
        if not self.image_ids: raise ValueError(f"Split '{image_set}' пустой: {image_set_path}")

        # Трансформации для изображения
        self.img_transform = transforms.Compose([
            transforms.Resize((self.img_size, self.img_size), interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225]),
            ])

    def __len__(self) -> int:
        return len(self.image_ids)

    def __getitem__(self, idx: int):
        image_id = self.image_ids[idx]

        # Загружаем изображение
        image_path = self.jpeg_dir / f"{image_id}.jpg"
        if not image_path.exists():
            raise FileNotFoundError(f"Не найдено изображение: {image_path}")
        image = Image.open(image_path).convert("RGB")

        # Загружаем маску
        mask_path = self.mask_dir / f"{image_id}.png"
        if not mask_path.exists():
            raise FileNotFoundError(f"Не найдена маска: {mask_path}")
        mask = Image.open(mask_path)

        if self.augment:
            # Один случайный flip применяется одновременно к image и mask.
            if torch.rand(1).item() < 0.5:
                image = TF.hflip(image)
                mask = TF.hflip(mask)

        # Для изображения используем bilinear.
        image = TF.resize(
            image,
            [self.img_size, self.img_size],
            interpolation=transforms.InterpolationMode.BILINEAR)
        # Для маски обязательно NEAREST. # Иначе интерполяция создаст несуществующие значения классов.
        mask = TF.resize(
            mask,
            [self.img_size, self.img_size],
            interpolation=transforms.InterpolationMode.NEAREST)

        if self.augment:
            color_jitter = transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2)
            image = color_jitter(image)

        image = TF.to_tensor(image)
        image = TF.normalize(image, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], )

        mask_np = np.asarray(mask, dtype=np.uint8)
        mask_class = mask_to_class(mask_np)
        mask_tensor = torch.from_numpy(mask_class).long()

        return image, mask_tensor


def get_segmentation_dataloaders(
        batch_size: int = None,
        img_size: int = None,
        num_workers: int = None,
):
    """Возвращает train_loader, val_loader для сегментации."""
    batch_size = batch_size or Config.B3["batch_size"]
    img_size = img_size or Config.B3["img_size"]
    num_workers = (Config.B3["num_workers"] if num_workers is None else num_workers)

    train_dataset = VOCSegmentationDataset(
        root=Config.voc_dir,
        image_set="train",
        img_size=img_size,
        augment=True,
    )
    val_dataset = VOCSegmentationDataset(
        root=Config.voc_dir,
        image_set="val",
        img_size=img_size,
        augment=False,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    return train_loader, val_loader


if __name__ == "__main__":
    # Проверка
    train_ldr, val_ldr = get_segmentation_dataloaders(batch_size=2)
    img, mask = next(iter(train_ldr))
    print(f"Image: {img.shape}")  # [2, 3, 256, 256]
    print(f"Mask:  {mask.shape}")  # [2, 256, 256]
    print(f"Unique classes in batch: {torch.unique(mask)}")
