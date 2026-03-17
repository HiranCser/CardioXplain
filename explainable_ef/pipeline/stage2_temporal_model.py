import torch
import torch.nn as nn
import torch.nn.functional as F


class Stage2TemporalModel(nn.Module):
    """Stage 2: temporal alignment and attention-based temporal pooling."""

    def __init__(self, num_frames=32, feature_dim=512, hidden_dim=256, dropout=0.1):
        super().__init__()
        self.num_frames = num_frames
        self.temporal_pos_embed = nn.Parameter(torch.zeros(1, num_frames, feature_dim))

        self.temporal_context3 = nn.Sequential(
            nn.Conv1d(feature_dim, feature_dim, kernel_size=3, padding=1, groups=feature_dim, bias=False),
            nn.BatchNorm1d(feature_dim),
            nn.GELU(),
        )
        self.temporal_context5 = nn.Sequential(
            nn.Conv1d(feature_dim, feature_dim, kernel_size=5, padding=2, groups=feature_dim, bias=False),
            nn.BatchNorm1d(feature_dim),
            nn.GELU(),
        )
        self.temporal_mix = nn.Sequential(
            nn.Conv1d(feature_dim * 3, feature_dim, kernel_size=1, bias=False),
            nn.BatchNorm1d(feature_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.temporal_norm = nn.LayerNorm(feature_dim)

        # Keep legacy parameter names for checkpoint compatibility; treat as ED/ES heads.
        self.temporal_attention = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Linear(feature_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )
        self.temporal_attention_es = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Linear(feature_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )
        self.attn_temperature_ed = nn.Parameter(torch.tensor(1.0))
        self.attn_temperature_es = nn.Parameter(torch.tensor(1.0))

    def forward(self, stage1_features):
        # stage1_features: (B, 512, T')
        temporal_feats = F.interpolate(
            stage1_features,
            size=self.num_frames,
            mode="linear",
            align_corners=False,
        )

        context3 = self.temporal_context3(temporal_feats)
        context5 = self.temporal_context5(temporal_feats)
        context = self.temporal_mix(torch.cat([temporal_feats, context3, context5], dim=1))

        # (B, 512, T) -> (B, T, 512)
        temporal_feats = temporal_feats.permute(0, 2, 1)
        context = context.permute(0, 2, 1)
        temporal_feats = self.temporal_norm(temporal_feats + context + self.temporal_pos_embed)

        temp_ed = self.attn_temperature_ed.clamp(0.25, 4.0)
        temp_es = self.attn_temperature_es.clamp(0.25, 4.0)
        attn_scores_ed = self.temporal_attention(temporal_feats) / temp_ed
        attn_scores_es = self.temporal_attention_es(temporal_feats) / temp_es
        attn_weights_ed = torch.softmax(attn_scores_ed, dim=1)
        attn_weights_es = torch.softmax(attn_scores_es, dim=1)
        attn_weights_pair = torch.cat([attn_weights_ed, attn_weights_es], dim=-1)

        # Pool with mean attention so EF uses a single fused temporal context.
        attn_weights_mean = attn_weights_pair.mean(dim=-1, keepdim=True)
        weighted_feats = temporal_feats * attn_weights_mean
        pooled_feats = weighted_feats.sum(dim=1)
        return temporal_feats, pooled_feats, attn_weights_pair
