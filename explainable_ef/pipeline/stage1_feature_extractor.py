import torch.nn as nn
import torchvision.models.video as models
from torchvision.models.video import R2Plus1D_18_Weights


class Stage1FeatureExtractor(nn.Module):
    """Stage 1: spatial feature extraction from echo clips."""

    def __init__(self, weights=R2Plus1D_18_Weights.DEFAULT):
        super().__init__()
        backbone = models.r2plus1d_18(weights=weights)
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
