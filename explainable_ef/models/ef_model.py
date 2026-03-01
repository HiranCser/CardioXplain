import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models.video as models
from torchvision.models.video import R2Plus1D_18_Weights


class EFModel(nn.Module):
    def __init__(self, num_frames=32):
        super().__init__()
        self.num_frames = num_frames

        backbone = models.r2plus1d_18(
            weights=R2Plus1D_18_Weights.DEFAULT
        )
        self.phase_classifier = nn.Linear(512, 3)  # 3 classes
        self.feature_extractor = nn.Sequential(
            backbone.stem,
            backbone.layer1,
            backbone.layer2,
            backbone.layer3,
            backbone.layer4
        )

        self.spatial_pool = nn.AdaptiveAvgPool3d((None, 1, 1))

        self.temporal_attention = nn.Linear(512, 1)
        self.regressor = nn.Linear(512, 1)

    def forward(self, x):
        # x: (B, C, T, H, W)

        feats = self.feature_extractor(x)
        # (B, 512, T', H', W')

        feats = self.spatial_pool(feats)
        # (B, 512, T', 1, 1)

        feats = feats.squeeze(-1).squeeze(-1)
        # (B, 512, T')

        # 🔥 Upsample temporal dimension back to original frame count
        feats = F.interpolate(
            feats,
            size=self.num_frames,
            mode="linear",
            align_corners=False
        )
        # (B, 512, T)

        feats = feats.permute(0, 2, 1)
        # (B, T, 512)

        attn_scores = self.temporal_attention(feats)
        attn_weights = torch.softmax(attn_scores, dim=1)
        # (B, T, 1)

        weighted_feats = feats * attn_weights
        pooled = weighted_feats.sum(dim=1)
        # (B, 512)

        ef = self.regressor(pooled)
        phase_logits = self.phase_classifier(feats)  # (B, T, 3)
        return ef.squeeze(1), attn_weights.squeeze(-1), phase_logits