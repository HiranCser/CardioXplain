import torch.nn as nn

from pipeline.orchestrator import EchoPipeline


def infer_ef_head_arch(state_dict):
    """Detect the EF head layout used by older checkpoints."""
    if isinstance(state_dict, dict) and "pipeline.ef_regressor.weight" in state_dict:
        return "legacy_linear"
    return "mlp"


class EFModel(nn.Module):
    """Compatibility wrapper that keeps existing training scripts unchanged."""

    def __init__(self, num_frames=32, ef_head_arch="mlp"):
        super().__init__()
        self.pipeline = EchoPipeline(num_frames=num_frames, feature_dim=512, ef_head_arch=ef_head_arch)

    def forward(self, x, **kwargs):
        return self.pipeline(x, **kwargs)
