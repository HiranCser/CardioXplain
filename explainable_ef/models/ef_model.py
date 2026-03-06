import torch.nn as nn

from pipeline.orchestrator import EchoPipeline


class EFModel(nn.Module):
    """Compatibility wrapper that keeps existing training scripts unchanged."""

    def __init__(self, num_frames=32):
        super().__init__()
        self.pipeline = EchoPipeline(num_frames=num_frames, feature_dim=512)

    def forward(self, x, **kwargs):
        return self.pipeline(x, **kwargs)
