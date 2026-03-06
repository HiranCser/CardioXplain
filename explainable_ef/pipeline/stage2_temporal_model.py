import torch
import torch.nn as nn
import torch.nn.functional as F


class Stage2TemporalModel(nn.Module):
    """Stage 2: temporal alignment and attention-based temporal pooling."""

    def __init__(self, num_frames=32, feature_dim=512):
        super().__init__()
        self.num_frames = num_frames
        self.temporal_attention = nn.Linear(feature_dim, 1)

    def forward(self, stage1_features):
        # stage1_features: (B, 512, T')
        temporal_feats = F.interpolate(
            stage1_features,
            size=self.num_frames,
            mode="linear",
            align_corners=False,
        )
        # (B, 512, T) -> (B, T, 512)
        temporal_feats = temporal_feats.permute(0, 2, 1)

        attn_scores = self.temporal_attention(temporal_feats)
        attn_weights = torch.softmax(attn_scores, dim=1)

        weighted_feats = temporal_feats * attn_weights
        pooled_feats = weighted_feats.sum(dim=1)
        return temporal_feats, pooled_feats, attn_weights.squeeze(-1)
