import torch
import torch.nn as nn


class Stage3PhaseDetector(nn.Module):
    """Stage 3: per-frame phase classification and ED/ES index extraction."""

    def __init__(self, feature_dim=512, num_classes=3, dropout=0.1):
        super().__init__()
        self.temporal_encoder = nn.Sequential(
            nn.Conv1d(feature_dim, feature_dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(feature_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(feature_dim, feature_dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(feature_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.phase_classifier = nn.Conv1d(feature_dim, num_classes, kernel_size=1)

    def forward(self, temporal_features):
        # temporal_features: (B, T, 512)
        x = temporal_features.transpose(1, 2)  # (B, 512, T)
        x = self.temporal_encoder(x)
        return self.phase_classifier(x).transpose(1, 2)  # (B, T, 3)

    @staticmethod
    def predict_indices(phase_logits):
        """Return ED and ES frame indices from per-frame logits."""
        pred_ed_idx = torch.argmax(phase_logits[:, :, 1], dim=1)
        pred_es_idx = torch.argmax(phase_logits[:, :, 2], dim=1)
        return pred_ed_idx, pred_es_idx
