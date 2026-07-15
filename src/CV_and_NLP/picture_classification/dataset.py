import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.datasets import VOCDetection

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


def get_dataloaders(config: Config, batch_size: int = None, num_workers: int = None):
    """
    Возвращает train_loader, val_loader.
    VOC trainval делится на train/val через random_split.
    """
    batch_size = batch_size or config.B1["batch_size"]
    num_workers = num_workers or config.B1["num_workers"]

    full_dataset = VOCMultiLabelDataset(
        config=config,
        root=config.voc_dir,
        image_set="trainval",
        transform=get_classification_transforms(train=True, config=config),
    )

    test_dataset = VOCMultiLabelDataset(
        config=config,
        root=config.voc_dir,
        image_set="test",
        transform=get_classification_transforms(train=False,config=config),
    )
    # Разбиение: 80% train, 20% val
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(config.seed),
    )

    # На валидации — другие трансформации
    val_dataset.dataset = VOCMultiLabelDataset(
        config=config,
        root=config.voc_dir,
        image_set="trainval",
        transform=get_classification_transforms(train=False, config=config),
    )
    # Переназначаем индексы после замены dataset
    val_subset_indices = val_dataset.indices
    val_dataset = torch.utils.data.Subset(val_dataset.dataset, val_subset_indices)

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size,
        shuffle=True, num_workers=num_workers,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size,
        shuffle=False, num_workers=num_workers,
    )

    test_loader = DataLoader(
        test_dataset, batch_size=batch_size,
        shuffle=False, num_workers=num_workers,
    )

    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    # Быстрая проверка: загружаем один батч
    config = Config()
    train_loader, val_loader = get_dataloaders(config=config, batch_size=4)
    images, targets = next(iter(train_loader))
    print(f"Batch shape: {images.shape}")  # [4, 3, 224, 224]
    print(f"Target shape: {targets.shape}")  # [4, 20]
    print(f"Classes per image: {targets.sum(dim=1)}")
