import torch
from torch import nn
from torchvision.models import resnet18, ResNet18_Weights

from configs.cv_and_nlp_config import Config


# ===================================================================
# Baseline CNN — архитектура с нуля
# ===================================================================
# 4 свёрточных блока:
#   Conv2d → ReLU → MaxPool2d (уменьшает пространство вдвое)
# Число каналов растёт:  3 → 32 → 64 → 128 → 256
# Пространственный размер: 224 → 112 → 56 → 28 → 14
# Затем: AdaptiveAvgPool2d(4,4) → Flatten → Linear(256*4*4, 256) → Dropout → Linear(256, 20)

class BaselineCNN(nn.Module):
    def __init__(self, config: Config, num_classes: int = None, dropout: float = None):
        super().__init__()
        num_classes = num_classes or config.B1["num_classes"]
        dropout = dropout or config.B1["dropout"]

        self.features = nn.Sequential(
            # Блок 1: 3 → 32, 224 → 112
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            # Блок 2: 32 → 64, 112 → 56
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            # Блок 3: 64 → 128, 56 → 28
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            # Блок 4: 128 → 256, 28 → 14
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
        )

        self.avgpool = nn.AdaptiveAvgPool2d((4, 4))

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 4 * 4, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = self.classifier(x)
        return x  # logits, без сигмоиды (она внутри BCEWithLogitsLoss)


# ===================================================================
# Transfer Learning: ResNet18 с ImageNet
# ===================================================================
def get_resnet18(config: Config, num_classes: int = None, freeze_backbone: bool = True):
    """
    ResNet18, предобученная на ImageNet (1000 классов).
    Заменяем последний fc-слой на 20 классов VOC.
    freeze_backbone=True → замораживаем все слои кроме fc
    (быстрый fine-tuning, основная стратегия).
    """
    num_classes = num_classes or config.B1["num_classes"]

    model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)

    if freeze_backbone:
        # Замораживаем ВСЕ параметры
        for param in model.parameters():
            param.requires_grad = False
        # Размораживаем только последний классификатор
        for param in model.fc.parameters():
            param.requires_grad = True

    # Заменяем голову на 20 классов
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)

    return model


# ===================================================================
# Быстрая проверка
# ===================================================================
if __name__ == "__main__":
    device = "cpu"
    config = Config()
    # Baseline
    baseline = BaselineCNN(config).to(device)
    x = torch.randn(2, 3, 224, 224).to(device)
    out = baseline(x)
    print(f"BaselineCNN output: {out.shape}")    # [2, 20]

    # ResNet18
    resnet = get_resnet18(config).to(device)
    out = resnet(x)
    print(f"ResNet18 output:    {out.shape}")    # [2, 20]

    # Считаем параметры
    baseline_params = sum(p.numel() for p in baseline.parameters()) / 1e6
    resnet_params = sum(p.numel() for p in resnet.parameters()) / 1e6
    resnet_trainable = sum(p.numel() for p in resnet.parameters() if p.requires_grad) / 1e6
    print(f"BaselineCNN: {baseline_params:.1f}M params")
    print(f"ResNet18:    {resnet_params:.1f}M params ({resnet_trainable:.1f}M trainable)")
