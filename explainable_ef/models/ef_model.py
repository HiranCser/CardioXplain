import torch.nn as nn

from pipeline.orchestrator import EchoPipeline


class EFModel(nn.Module):
    """Compatibility wrapper that keeps existing training scripts unchanged."""

    def __init__(self, num_frames=32, use_pretrained_backbone=True):
        super().__init__()
        self.pipeline = EchoPipeline(
            num_frames=num_frames,
            feature_dim=512,
            use_pretrained_backbone=use_pretrained_backbone,
        )

    def forward(self, x, **kwargs):
        return self.pipeline(x, **kwargs)
