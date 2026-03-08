import torch
import torch.nn as nn
import torchvision


class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class UpBlock(nn.Module):
    def __init__(self, in_ch, skip_ch, out_ch):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2)
        self.conv = ConvBlock(out_ch + skip_ch, out_ch)

    def forward(self, x, skip):
        x = self.up(x)
        if x.shape[-2:] != skip.shape[-2:]:
            x = nn.functional.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


class Stage4SegmentationUNet(nn.Module):
    """Lightweight U-Net for LV segmentation (Stage 4)."""

    def __init__(self, in_channels=3, base_channels=32):
        super().__init__()
        self.enc1 = ConvBlock(in_channels, base_channels)
        self.enc2 = ConvBlock(base_channels, base_channels * 2)
        self.enc3 = ConvBlock(base_channels * 2, base_channels * 4)
        self.enc4 = ConvBlock(base_channels * 4, base_channels * 8)

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.bottleneck = ConvBlock(base_channels * 8, base_channels * 16)

        self.up4 = UpBlock(base_channels * 16, base_channels * 8, base_channels * 8)
        self.up3 = UpBlock(base_channels * 8, base_channels * 4, base_channels * 4)
        self.up2 = UpBlock(base_channels * 4, base_channels * 2, base_channels * 2)
        self.up1 = UpBlock(base_channels * 2, base_channels, base_channels)

        self.head = nn.Conv2d(base_channels, 1, kernel_size=1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))

        b = self.bottleneck(self.pool(e4))

        d4 = self.up4(b, e4)
        d3 = self.up3(d4, e3)
        d2 = self.up2(d3, e2)
        d1 = self.up1(d2, e1)

        return self.head(d1)


def _load_torchvision_segmentation(model_name, pretrained):
    """
    Build torchvision segmentation model with broad compatibility across torchvision versions.
    """
    model_fn = torchvision.models.segmentation.__dict__[model_name]

    # Newer torchvision API
    if pretrained:
        try:
            model = model_fn(weights="DEFAULT")
            return model
        except Exception:
            pass

    # Legacy API fallback
    try:
        model = model_fn(pretrained=pretrained, aux_loss=False)
    except TypeError:
        model = model_fn(pretrained=pretrained)
    return model


def _replace_classifier_head_with_binary(model):
    if not hasattr(model, "classifier"):
        raise ValueError("Selected segmentation model has no classifier head")

    classifier = model.classifier
    if isinstance(classifier, nn.Sequential):
        if not hasattr(classifier[-1], "in_channels"):
            raise ValueError("Could not infer classifier in_channels for binary head")
        in_channels = classifier[-1].in_channels
        kernel_size = getattr(classifier[-1], "kernel_size", 1)
        classifier[-1] = nn.Conv2d(in_channels, 1, kernel_size=kernel_size)
        model.classifier = classifier
        return model

    # Conservative fallback for uncommon classifier structures
    for name, mod in reversed(list(classifier.named_modules())):
        if isinstance(mod, nn.Conv2d):
            parent = classifier
            parts = name.split(".")
            for part in parts[:-1]:
                parent = parent[int(part)] if part.isdigit() else getattr(parent, part)
            leaf = parts[-1]
            new_conv = nn.Conv2d(mod.in_channels, 1, kernel_size=mod.kernel_size)
            if leaf.isdigit():
                parent[int(leaf)] = new_conv
            else:
                setattr(parent, leaf, new_conv)
            return model

    raise ValueError("Could not replace segmentation classifier head")


def build_stage4_segmentation_model(model_name="deeplabv3_resnet50", pretrained=False, in_channels=3, base_channels=32):
    """
    Build Stage-4 segmentation model.

    - `unet`: custom lightweight U-Net
    - torchvision segmentation models (e.g., deeplabv3_resnet50, fcn_resnet50)
      with binary output head, aligned with EchoNet segmentation baseline style.
    """
    model_name = str(model_name).lower()

    if model_name == "unet":
        return Stage4SegmentationUNet(in_channels=in_channels, base_channels=base_channels)

    if model_name not in torchvision.models.segmentation.__dict__:
        available = sorted(
            name
            for name, fn in torchvision.models.segmentation.__dict__.items()
            if name.islower() and not name.startswith("__") and callable(fn)
        )
        raise ValueError(f"Unknown model_name '{model_name}'. Available: {available + ['unet']}")

    model = _load_torchvision_segmentation(model_name=model_name, pretrained=bool(pretrained))
    model = _replace_classifier_head_with_binary(model)
    return model
