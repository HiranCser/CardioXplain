import torch.nn as nn
import torchvision.models.video as models
from torchvision.models.video import R2Plus1D_18_Weights


class Stage1FeatureExtractor(nn.Module):
    """Stage 1: spatial feature extraction from echo clips."""

    def __init__(self, weights=R2Plus1D_18_Weights.DEFAULT, preserve_temporal_stride=True):
        super().__init__()
        try:
            backbone = models.r2plus1d_18(weights=weights)
        except Exception:
            # Fallback for offline/runtime environments where pretrained weights
            # are unavailable; checkpoint loading will still populate weights.
            backbone = models.r2plus1d_18(weights=None)

        if bool(preserve_temporal_stride):
            # Preserve more temporal detail for phase localization:
            # default backbone uses temporal stride=2 in layer4 block0, which makes T' too small.
            layer4_block0 = backbone.layer4[0]
            layer4_block0.conv1[0][3].stride = (1, 1, 1)
            layer4_block0.downsample[0].stride = (1, 2, 2)

        self.feature_extractor = nn.Sequential(
            backbone.stem,
            backbone.layer1,
            backbone.layer2,
            backbone.layer3,
            backbone.layer4,
        )
        self.spatial_pool = nn.AdaptiveAvgPool3d((None, 1, 1))

    def forward(self, x):
        # x: (B, C, T, H, W)
        feats = self.feature_extractor(x)
        feats = self.spatial_pool(feats)
        # (B, 512, T', 1, 1) -> (B, 512, T')
        feats = feats.squeeze(-1).squeeze(-1)
        return feats
