import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms
from torchvision.datasets import VOCDetection
from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit

from configs.cv_and_nlp_config import Config


def get_classification_transforms(train: bool, config: Config) -> transforms.Compose:
    """
    Возвращает трансформации для классификации.
    train=True  → аугментации (только для обучения)
    train=False → только Resize + CenterCrop + нормализация
    """
    if train:
        return transforms.Compose([
            transforms.RandomResizedCrop(config.B1["image_size"]),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2,
                                   saturation=0.2, hue=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])
    else:
        return transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(config.B1["image_size"]),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])


class VOCMultiLabelDataset(Dataset):
    """
    Оборачивает VOCDetection для задачи multi-label классификации.
    VOC изображения могут содержать несколько объектов → несколько классов.
    target[i] = 1 если класс i присутствует на фото, иначе 0.
    """

    def __init__(
            self,
            config: Config,
            root: str,
            year: str = "2012",
            image_set: str = "train", transform=None,
            download: bool = False):
        self.voc = VOCDetection(
            root=root,
            year=year,
            image_set=image_set,
            download=download,
            transform=None,
        )
        self.transform = transform
        self.classes = config.B1["classes"]
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}

    def __len__(self):
        return len(self.voc)

    def __getitem__(self, idx):
        image, annotation = self.voc[idx]

        # Собираем все классы с этого изображения
        target = torch.zeros(len(self.classes), dtype=torch.float32)
        objects = annotation["annotation"].get("object", [])
        if not isinstance(objects, list):
            objects = [objects]

        for obj in objects:
            class_name = obj["name"]
            if class_name in self.class_to_idx:
                target[self.class_to_idx[class_name]] = 1.0

        if self.transform:
            image = self.transform(image)

        return image, target


def get_targets(dataset):
    """Возвращает матрицу multilabel targets для всего датасета.
    Shape:
    [num_images, num_classes]
    """
    targets = []
    for idx in range(len(dataset.voc)):
        _, annotation = dataset.voc[idx]
        target = np.zeros(len(dataset.classes), dtype=np.int8)
        objects = annotation["annotation"].get("object", [])
        if not isinstance(objects, list):
            objects = [objects]
        for obj in objects:
            class_name = obj["name"]
            if class_name in dataset.class_to_idx:
                target[dataset.class_to_idx[class_name]] = 1
        targets.append(target)

    return np.array(targets)


def print_class_distribution(targets, indices, classes, split_name):
    """Выводит распределение классов в одном из split."""

    split_targets = targets[indices]

    print(f"\n{split_name}")
    print("-" * 45)

    for class_name, count in zip(classes, split_targets.sum(axis=0)):
        percent = 100 * count / len(indices)
        print(f"{class_name:15s}: {int(count):4d} ({percent:5.1f}%)")

def save_class_distribution(targets, train_idx, val_idx, test_idx,
                            classes, save_path):
    """Сохраняет распределение классов в train/val/test в CSV."""

    total_counts = targets.sum(axis=0).astype(int)
    train_counts = targets[train_idx].sum(axis=0).astype(int)
    val_counts = targets[val_idx].sum(axis=0).astype(int)
    test_counts = targets[test_idx].sum(axis=0).astype(int)

    df = pd.DataFrame({
        "class": classes,
        "total": total_counts,
        "train": train_counts,
        "val": val_counts,
        "test": test_counts,
        "train_%": np.round(train_counts / len(train_idx) * 100, 2),
        "val_%": np.round(val_counts / len(val_idx) * 100, 2),
        "test_%": np.round(test_counts / len(test_idx) * 100, 2),
        "matches_total": (train_counts + val_counts + test_counts) == total_counts,
    })

    df.to_csv(save_path, index=False)

    print(f"Распределение классов сохранено {save_path}")

def get_dataloaders(config: Config, batch_size: int = None, num_workers: int = None):
    """
    Возвращает train_loader, val_loader, test_loader
    VOC trainval делится на train/val/test через MultilabelStratifiedShuffleSplit:
    70% train
    15% val
    15% test
    """
    batch_size = batch_size or config.B1["batch_size"]
    num_workers = num_workers or config.B1["num_workers"]
    pin_memory = config.device == "cuda"
    persistent_workers = num_workers > 0
    full_dataset = VOCMultiLabelDataset(
        config=config,
        root=config.voc_dir,
        image_set="trainval",
        transform=get_classification_transforms(train=True, config=config),
    )

    targets = get_targets(full_dataset)
    indices = np.arange(len(full_dataset))

    # Разбиение: 70% train, 15% val, 15% test
    test_val_size = config.B1["val_size"] + config.B1["test_size"]
    splitter = MultilabelStratifiedShuffleSplit(n_splits=1, test_size=test_val_size, random_state=config.seed)
    train_idx, temp_idx = next(splitter.split(indices, targets))
    temp_targets = targets[temp_idx]

    # Делим temp на val и test (50/50)
    splitter_test = MultilabelStratifiedShuffleSplit(n_splits=1, test_size=config.B1["test_size"]/test_val_size, random_state=config.seed)
    val_idx_rel, test_idx_rel = next(splitter_test.split(temp_idx, temp_targets))
    val_idx = temp_idx[val_idx_rel]
    test_idx = temp_idx[test_idx_rel]

    split_dir = config.artifacts_B1 / "splits"
    split_dir.mkdir(parents=True, exist_ok=True)

    np.save(split_dir / "train_idx.npy", train_idx)
    np.save(split_dir / "val_idx.npy", val_idx)
    np.save(split_dir / "test_idx.npy", test_idx)
    save_class_distribution(
        targets=targets,
        train_idx=train_idx,
        val_idx=val_idx,
        test_idx=test_idx,
        classes=config.B1["classes"],
        save_path=split_dir / "class_distribution.csv",
    )

    train_dataset_full = VOCMultiLabelDataset(
        config=config,
        root=config.voc_dir,
        image_set="trainval",
        transform=get_classification_transforms(
            train=True,
            config=config
        ),
    )

    eval_dataset = VOCMultiLabelDataset(
        config=config,
        root=config.voc_dir,
        image_set="trainval",
        transform=get_classification_transforms(
            train=False,
            config=config
        ),
    )

    train_dataset = Subset(train_dataset_full, train_idx)
    val_dataset = Subset(eval_dataset, val_idx)
    test_dataset = Subset(eval_dataset, test_idx)

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size,
        shuffle=True, num_workers=num_workers,
        pin_memory=pin_memory, persistent_workers=persistent_workers,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size,
        shuffle=False, num_workers=num_workers,
        pin_memory=pin_memory, persistent_workers=persistent_workers,
    )

    test_loader = DataLoader(
        test_dataset, batch_size=batch_size,
        shuffle=False, num_workers=num_workers,
        pin_memory=pin_memory, persistent_workers=persistent_workers,
    )
    print_class_distribution(targets, train_idx, config.B1["classes"], "TRAIN")
    print_class_distribution(targets, val_idx, config.B1["classes"], "VALIDATION")
    print_class_distribution(targets, test_idx, config.B1["classes"], "TEST")
    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    # Быстрая проверка: загружаем один батч
    config = Config()
    train_loader, val_loader, test_loader = get_dataloaders(config=config, batch_size=4)
    images, targets = next(iter(train_loader))
    print(f"Batch shape: {images.shape}")  # [4, 3, 224, 224]
    print(f"Target shape: {targets.shape}")  # [4, 20]
    print(f"Classes per image: {targets.sum(dim=1)}")
