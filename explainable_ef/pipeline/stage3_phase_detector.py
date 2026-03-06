import torch
import torch.nn as nn


class Stage3PhaseDetector(nn.Module):
    """Stage 3: per-frame phase classification and ED/ES index extraction."""

    def __init__(self, feature_dim=512, num_classes=3):
        super().__init__()
        self.phase_classifier = nn.Linear(feature_dim, num_classes)

    def forward(self, temporal_features):
        # temporal_features: (B, T, 512)
        return self.phase_classifier(temporal_features)

    @staticmethod
    def predict_indices(phase_logits):
        """Return ED and ES frame indices from per-frame logits."""
        pred_ed_idx = torch.argmax(phase_logits[:, :, 1], dim=1)
        pred_es_idx = torch.argmax(phase_logits[:, :, 2], dim=1)
        return pred_ed_idx, pred_es_idx
