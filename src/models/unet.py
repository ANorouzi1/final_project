import torch
import torch.nn as nn
import torch.nn.functional as F

from .base_model import BaseModel

"""
U-Net architecture.

this code is moslty taken from "https://github.com/milesial/pytorch-unet" with some modifications to fit the dual-head design of this project.
"""



class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class Down(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(nn.MaxPool2d(2), DoubleConv(in_channels, out_channels))

    def forward(self, x):
        return self.block(x)


class Up(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels, bilinear=True):
        super().__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
            self.conv = DoubleConv(in_channels + skip_channels, out_channels)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels // 2 + skip_channels, out_channels)

    def forward(self, x, skip):
        x = self.up(x)
        diff_y = skip.size(2) - x.size(2)
        diff_x = skip.size(3) - x.size(3)
        x = F.pad(x, [diff_x // 2, diff_x - diff_x // 2, diff_y // 2, diff_y - diff_y // 2])
        return self.conv(torch.cat([skip, x], dim=1))


class DualHeadUNet(BaseModel):
    """U-Net backbone with mask and signed-distance prediction heads."""

    def __init__(
        self,
        in_channels=3,
        base_channels=32,
        num_classes=1,
        bilinear=True,
        predict_distance=True,
    ):
        super().__init__()
        self.predict_distance = predict_distance
        c = base_channels
        self.inc = DoubleConv(in_channels, c)
        self.down1 = Down(c, c * 2)
        self.down2 = Down(c * 2, c * 4)
        self.down3 = Down(c * 4, c * 8)
        self.down4 = Down(c * 8, c * 16)
        self.up1 = Up(c * 16, c * 8, c * 8, bilinear=bilinear)
        self.up2 = Up(c * 8, c * 4, c * 4, bilinear=bilinear)
        self.up3 = Up(c * 4, c * 2, c * 2, bilinear=bilinear)
        self.up4 = Up(c * 2, c, c, bilinear=bilinear)
        self.mask_head = nn.Conv2d(c, num_classes, kernel_size=1)
        if self.predict_distance:
            self.distance_head = nn.Sequential(
                nn.Conv2d(c, c, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(c),
                nn.ReLU(inplace=True),
                nn.Conv2d(c, 1, kernel_size=1),
            )
        else:
            self.distance_head = None

    def _forward_features(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        return x

    def forward(self, x):
        features = self._forward_features(x)
        outputs = {"mask_logits": self.mask_head(features)}
        if self.distance_head is not None:
            outputs["distance_logits"] = self.distance_head(features)
        return outputs


class FrameFieldUNet(DualHeadUNet):
    """Three-head U-Net with mask, signed-distance, and frame-field outputs."""

    def __init__(self, in_channels=3, base_channels=32, num_classes=1, bilinear=True):
        super().__init__(
            in_channels=in_channels,
            base_channels=base_channels,
            num_classes=num_classes,
            bilinear=bilinear,
        )
        c = base_channels
        self.frame_field_head = nn.Sequential(
            nn.Conv2d(c, c, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(c),
            nn.ReLU(inplace=True),
            nn.Conv2d(c, 4, kernel_size=1),
        )

    def forward(self, x):
        features = self._forward_features(x)
        return {
            "mask_logits": self.mask_head(features),
            "distance_logits": self.distance_head(features),
            "frame_field": self.frame_field_head(features),
        }


class MaskOnlyUNet(BaseModel):
    """U-Net baseline with the same backbone but no auxiliary distance head."""

    def __init__(self, in_channels=3, base_channels=32, num_classes=1, bilinear=True):
        super().__init__()
        c = base_channels
        self.inc = DoubleConv(in_channels, c)
        self.down1 = Down(c, c * 2)
        self.down2 = Down(c * 2, c * 4)
        self.down3 = Down(c * 4, c * 8)
        self.down4 = Down(c * 8, c * 16)
        self.up1 = Up(c * 16, c * 8, c * 8, bilinear=bilinear)
        self.up2 = Up(c * 8, c * 4, c * 4, bilinear=bilinear)
        self.up3 = Up(c * 4, c * 2, c * 2, bilinear=bilinear)
        self.up4 = Up(c * 2, c, c, bilinear=bilinear)
        self.mask_head = nn.Conv2d(c, num_classes, kernel_size=1)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        return {"mask_logits": self.mask_head(x)}
