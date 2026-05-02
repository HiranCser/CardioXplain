import torch
import torch.nn as nn

from pipeline.orchestrator import EchoPipeline


class EFModel(nn.Module):
    """Compatibility wrapper that keeps existing training scripts unchanged."""

    def __init__(self, num_frames=32, preserve_temporal_stride=True):
        super().__init__()
        self.pipeline = EchoPipeline(
            num_frames=num_frames,
            feature_dim=512,
            preserve_temporal_stride=preserve_temporal_stride,
        )

    def forward(self, x, **kwargs):
        return self.pipeline(x, **kwargs)


def resolve_stage1_preserve_temporal_stride(checkpoint_dict, default=True):
    if not isinstance(checkpoint_dict, dict):
        return bool(default)

    args = checkpoint_dict.get("args", {})
    if isinstance(args, dict) and "stage1_preserve_temporal_stride" in args:
        return bool(args["stage1_preserve_temporal_stride"])

    runtime_config = checkpoint_dict.get("runtime_config", {})
    if isinstance(runtime_config, dict) and "STAGE1_PRESERVE_TEMPORAL_STRIDE" in runtime_config:
        return bool(runtime_config["STAGE1_PRESERVE_TEMPORAL_STRIDE"])

    return bool(default)


def load_ef_model_from_checkpoint(checkpoint_path, num_frames, device, default_preserve_temporal_stride=True):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    checkpoint_dict = checkpoint if isinstance(checkpoint, dict) else {}
    state_dict = checkpoint_dict["model_state_dict"] if "model_state_dict" in checkpoint_dict else checkpoint

    preserve_temporal_stride = resolve_stage1_preserve_temporal_stride(
        checkpoint_dict,
        default=default_preserve_temporal_stride,
    )
    model = EFModel(
        num_frames=int(num_frames),
        preserve_temporal_stride=preserve_temporal_stride,
    ).to(device)

    model_state = model.state_dict()
    filtered_state_dict = {
        key: value
        for key, value in state_dict.items()
        if key in model_state and tuple(value.shape) == tuple(model_state[key].shape)
    }
    incompatible = model.load_state_dict(filtered_state_dict, strict=False)
    model.eval()
    return model, incompatible, checkpoint_dict
