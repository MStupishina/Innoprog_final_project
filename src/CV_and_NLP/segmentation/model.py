import torch
from torch import nn

from configs.cv_and_nlp_config import Config


class DoubleConv(nn.Module):
    """Два свёрточных слоя подряд: Conv → BN → ReLU → Conv → BN → ReLU."""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.double_conv(x)


class UNet(nn.Module):
    """
    U-Net архитектура.

    Encoder (слева):    DoubleConv → MaxPool → ...
    Bottleneck (дно):    DoubleConv
    Decoder (справа):    ConvTranspose2d → concat(skip) → DoubleConv → ...

    Skip connections: выход каждого уровня энкодера подаётся
    на соответствующий уровень декодера (конкатенация по каналам).
    Именно это сохраняет пространственную информацию о границах.
    """

    def __init__(
            self,
            in_channels: int = None,
            out_channels: int = None,
            features: list[int] | None = None
    ):
        super().__init__()
        in_channels = (Config.B3["in_channels"] if in_channels is None else in_channels)
        out_channels = (Config.B3["out_channels"] if out_channels is None else out_channels)
        features = (Config.B3["features"] if features is None else features)  # напр. [64, 128, 256, 512]

        self.encoder = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Encoder: Downsampling
        in_ch = in_channels
        for feature in features:
            self.encoder.append(DoubleConv(in_ch, feature))
            in_ch = feature

        # Bottleneck
        self.bottleneck = DoubleConv(features[-1], features[-1] * 2)

        # Decoder: Upsampling
        self.up_convs = nn.ModuleList()
        self.decoder = nn.ModuleList()

        for feature in reversed(features):
            # ConvTranspose2d: удваиваем пространственный размер
            self.up_convs.append(
                nn.ConvTranspose2d(feature * 2, feature, kernel_size=2, stride=2)
            )
            # После конкатенации с skip → DoubleConv
            self.decoder.append(DoubleConv(feature * 2, feature))

        # Финальная свёртка → карта классов
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x):
        # Encoder: сохраняем skip connections
        skip_connections = []
        for enc in self.encoder:
            x = enc(x)
            skip_connections.append(x)
            x = self.pool(x)

        # Bottleneck
        x = self.bottleneck(x)

        # Decoder с skip connections
        skip_connections = skip_connections[::-1]  # разворачиваем (от глубоких к мелким)

        for idx in range(len(self.decoder)):
            x = self.up_convs[idx](x)  # Upsample
            skip = skip_connections[idx]

            # Если размеры не совпадают (из-за нечётных размеров), обрезаем
            if x.shape[2:] != skip.shape[2:]:
                x = nn.functional.interpolate(x, size=skip.shape[2:], mode="bilinear", align_corners=False)

            x = torch.cat([skip, x], dim=1)  # Конкатенация со skip
            x = self.decoder[idx](x)  # DoubleConv

        # Финальная свёртка
        x = self.final_conv(x)
        return x  # [B, num_classes, H, W]


if __name__ == "__main__":
    # Быстрая проверка
    device = "cpu"
    model = UNet().to(device)
    x = torch.randn(2, 3, 256, 256).to(device)
    out = model(x)
    print(f"Input:  {x.shape}")  # [2, 3, 256, 256]
    print(f"Output: {out.shape}")  # [2, 21, 256, 256]

    params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"Parameters: {params:.1f}M")
