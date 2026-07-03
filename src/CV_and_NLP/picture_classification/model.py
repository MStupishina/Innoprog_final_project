import torch
import torch.nn as nn
from torchvision import models

class BaselineCNN(nn.Module):
    """
    Простая сверточная нейросеть (CNN) с нуля в качестве бейзлайна
    """

    def __init__(self, num_classes=20):
        super(BaselineCNN, self).__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Усредняем карты признаков до размера 4x4
            nn.AdaptiveAvgPool2d((4, 4))
        )

        self.classifier = nn.Sequential(
            # 128 каналов * 4 * 4 (размер после пулинга)
            nn.Linear(128 * 4 * 4, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


class ResNetTransferLearning(nn.Module):
    """
    Модель для Transfer Learning на основе ResNet18
    """

    def __init__(self, num_classes=20, use_pretrained=True):
        super(ResNetTransferLearning, self).__init__()

        weights = models.ResNet18_Weights.DEFAULT if use_pretrained else None
        self.backbone = models.resnet18(weights=weights)

        # ЗАМОРОЗКА ВЕСОВ
        # Запрещаем обучать всю базу ResNet, чтобы не сломать предобученные фильтры
        for param in self.backbone.parameters():
            param.requires_grad = False

        in_features = self.backbone.fc.in_features
        # Этот слой создается с requires_grad=True по умолчанию
        self.backbone.fc = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.backbone(x)


def get_model(model_type='transfer', num_classes=20, use_pretrained=True):
    if model_type == 'baseline':
        print("Инициализация: Простая Baseline CNN с нуля (4x4 AdaptivePool)")
        return BaselineCNN(num_classes=num_classes)
    elif model_type == 'transfer':
        print("Инициализация: ResNet18 (Transfer Learning - Замороженные веса)")
        return ResNetTransferLearning(num_classes=num_classes, use_pretrained=use_pretrained)
    else:
        raise ValueError("Unknown model_type. Choose 'baseline' or 'transfer'.")
