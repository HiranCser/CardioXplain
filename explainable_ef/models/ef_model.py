import torch
import torch.nn as nn
import torchvision.models.video as models
from torchvision.models.video import R2Plus1D_18_Weights



class EFModel(nn.Module):
    def __init__(self):
        super().__init__()

        backbone = models.r2plus1d_18(
            weights=R2Plus1D_18_Weights.KINETICS400_V1
        )


        
        self.feature_extractor = nn.Sequential(
            backbone.stem,
            backbone.layer1,
            backbone.layer2,
            backbone.layer3,
            backbone.layer4
        )

        self.global_pool = nn.AdaptiveAvgPool3d((None, 1, 1))  # keep time
        self.temporal_attention = nn.Linear(512, 1)
        self.regressor = nn.Linear(512, 1)

    def forward(self, x):
        # x: (B, C, T, H, W)

        feats = self.feature_extractor(x)
        # shape: (B, 512, T', H', W')

        feats = self.global_pool(feats)
        # shape: (B, 512, T', 1, 1)

        feats = feats.squeeze(-1).squeeze(-1)  
        # shape: (B, 512, T')

        feats = feats.permute(0, 2, 1)  
        # shape: (B, T', 512)

        # Temporal attention
        attn_scores = self.temporal_attention(feats)  
        # (B, T', 1)

        attn_weights = torch.softmax(attn_scores, dim=1)
        # normalized over time

        weighted_feats = feats * attn_weights
        pooled = weighted_feats.sum(dim=1)
        # (B, 512)

        ef = self.regressor(pooled)

        return ef.squeeze(1), attn_weights.squeeze(-1)