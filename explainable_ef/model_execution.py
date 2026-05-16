import argparse
from datetime import datetime
import logging
import math
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

import config
from data.dataset import EchoDataset
from models.ef_model import EFModel
from pipeline.stage3_phase_detector import Stage3PhaseDetector


SMOKE_DEFAULTS = {
    "MAX_VIDEOS": 24,
    "EPOCHS": 2,
    "BATCH_SIZE": 4,
    "NUM_FRAMES": 16,
    "CHECKPOINT_PATH": "best_model_smoke.pth",
}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Train and evaluate EF model (Stage 1-3).")
    parser.add_argument("--smoke", action="store_true", help="Run a tiny smoke test configuration.")
    parser.add_argument("--data-dir", type=str, default=None, help="Override config.DATA_DIR")
    parser.add_argument("--max-videos", type=int, default=None, help="Override config.MAX_VIDEOS")
    parser.add_argument("--epochs", type=int, default=None, help="Override config.EPOCHS")
    parser.add_argument("--learning-rate", "--lr", dest="learning_rate", type=float, default=None, help="Override config.LEARNING_RATE")
    parser.add_argument("--batch-size", type=int, default=None, help="Override config.BATCH_SIZE")
    parser.add_argument("--num-frames", type=int, default=None, help="Override config.NUM_FRAMES")
    parser.add_argument("--checkpoint", type=str, default=None, help="Override config.CHECKPOINT_PATH")
    parser.add_argument("--workers", type=int, default=None, help="Override config.NUM_WORKERS")
    parser.add_argument("--validate-every", type=int, default=None, help="Override config.VALIDATE_EVERY")
    parser.add_argument("--prefetch-factor", type=int, default=None, help="Override config.PREFETCH_FACTOR")
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=None, help="Enable/disable mixed precision")
    parser.add_argument("--pin-memory", action=argparse.BooleanOptionalAction, default=None, help="Enable/disable DataLoader pin_memory")
    parser.add_argument("--persistent-workers", action=argparse.BooleanOptionalAction, default=None, help="Enable/disable DataLoader persistent_workers")
    parser.add_argument("--tf32", action=argparse.BooleanOptionalAction, default=None, help="Enable/disable TF32 matmul/cuDNN")
    parser.add_argument("--benchmark", action=argparse.BooleanOptionalAction, default=None, help="Enable/disable cuDNN benchmark")
    parser.add_argument("--normalize-input", action=argparse.BooleanOptionalAction, default=None, help="Enable/disable Kinetics input normalization")
    parser.add_argument("--phase-loss-weight", type=float, default=None, help="Override config.PHASE_LOSS_WEIGHT")
    parser.add_argument("--phase-label-smoothing", type=float, default=None, help="Override phase index CE label smoothing")
    parser.add_argument("--phase-only", action=argparse.BooleanOptionalAction, default=None, help="Enable/disable phase-only training (no EF loss)")
    parser.add_argument("--phase-only-warmup-epochs", type=int, default=None, help="Temporarily disable EF loss for N initial joint-training epochs")
    parser.add_argument("--phase-backbone-freeze-epochs", type=int, default=None, help="Override config.PHASE_BACKBONE_FREEZE_EPOCHS")
    parser.add_argument("--backbone-lr-mult", type=float, default=None, help="Override config.BACKBONE_LR_MULT")
    parser.add_argument("--phase-soft-sigma", type=float, default=None, help="Override config.PHASE_SOFT_SIGMA")
    parser.add_argument("--phase-soft-radius", type=int, default=None, help="Override config.PHASE_SOFT_RADIUS")
    parser.add_argument("--phase-hard-index-weight", type=float, default=None, help="Override config.PHASE_HARD_INDEX_WEIGHT")
    parser.add_argument("--phase-frame-ce-weight", type=float, default=None, help="Override config.PHASE_FRAME_CE_WEIGHT")
    parser.add_argument("--phase-frame-radius", type=int, default=None, help="Override config.PHASE_FRAME_RADIUS")
    parser.add_argument("--phase-attn-align-weight", type=float, default=None, help="Override config.PHASE_ATTN_ALIGN_WEIGHT")
    parser.add_argument("--phase-attn-align-sigma", type=float, default=None, help="Override config.PHASE_ATTN_ALIGN_SIGMA")
    parser.add_argument("--phase-attn-align-radius", type=int, default=None, help="Override config.PHASE_ATTN_ALIGN_RADIUS")
    parser.add_argument("--phase-attn-index-weight", type=float, default=None, help="Override config.PHASE_ATTN_INDEX_WEIGHT")
    parser.add_argument("--phase-attn-order-weight", type=float, default=None, help="Override config.PHASE_ATTN_ORDER_WEIGHT")
    parser.add_argument("--phase-attn-entropy-weight", type=float, default=None, help="Override config.PHASE_ATTN_ENTROPY_WEIGHT")
    parser.add_argument("--phase-attn-min-gap", type=int, default=None, help="Override config.PHASE_ATTN_MIN_GAP")
    parser.add_argument("--phase-pair-index-weight", type=float, default=None, help="Override config.PHASE_PAIR_INDEX_WEIGHT")
    parser.add_argument("--phase-pair-order-weight", type=float, default=None, help="Override config.PHASE_PAIR_ORDER_WEIGHT")
    parser.add_argument("--phase-pair-min-gap", type=int, default=None, help="Override config.PHASE_PAIR_MIN_GAP")
    parser.add_argument("--phase-unfreeze-lr-mult", type=float, default=None, help="Override config.PHASE_UNFREEZE_LR_MULT")
    parser.add_argument("--weight-decay", type=float, default=None, help="Override config.WEIGHT_DECAY")
    parser.add_argument("--max-grad-norm", type=float, default=None, help="Override config.MAX_GRAD_NORM")
    parser.add_argument("--clip-period", type=int, default=None, help="Override config.CLIP_PERIOD")
    parser.add_argument("--clip-eval-mode", type=str, choices=["center", "all"], default=None, help="Override config.CLIP_EVAL_MODE")
    parser.add_argument("--warm-start-checkpoint", action=argparse.BooleanOptionalAction, default=True, help="Warm-start model weights from existing checkpoint path if available")
    parser.add_argument("--protect-best-checkpoint", action=argparse.BooleanOptionalAction, default=True, help="Do not overwrite checkpoint unless monitor improves over existing checkpoint")
    parser.add_argument("--train-stage123", action=argparse.BooleanOptionalAction, default=False, help="Train Stage1+Stage2+Stage3 end-to-end with joint EF+phase supervision")
    return parser.parse_args(argv)


def apply_runtime_overrides(args, logger):
    overrides = {}

    if args.smoke:
        overrides.update(SMOKE_DEFAULTS)

    if args.data_dir is not None:
        overrides["DATA_DIR"] = args.data_dir
    if args.max_videos is not None:
        overrides["MAX_VIDEOS"] = args.max_videos
    if args.epochs is not None:
        overrides["EPOCHS"] = args.epochs
    if args.learning_rate is not None:
        overrides["LEARNING_RATE"] = args.learning_rate
    if args.batch_size is not None:
        overrides["BATCH_SIZE"] = args.batch_size
    if args.num_frames is not None:
        overrides["NUM_FRAMES"] = args.num_frames
    if args.checkpoint is not None:
        overrides["CHECKPOINT_PATH"] = args.checkpoint
    if args.workers is not None:
        overrides["NUM_WORKERS"] = args.workers
    if args.validate_every is not None:
        overrides["VALIDATE_EVERY"] = args.validate_every
    if args.prefetch_factor is not None:
        overrides["PREFETCH_FACTOR"] = args.prefetch_factor
    if args.amp is not None:
        overrides["USE_MIXED_PRECISION"] = args.amp
    if args.pin_memory is not None:
        overrides["PIN_MEMORY"] = args.pin_memory
    if args.persistent_workers is not None:
        overrides["PERSISTENT_WORKERS"] = args.persistent_workers
    if args.tf32 is not None:
        overrides["ENABLE_TF32"] = args.tf32
    if args.benchmark is not None:
        overrides["CUDNN_BENCHMARK"] = args.benchmark
    if args.normalize_input is not None:
        overrides["NORMALIZE_INPUT"] = args.normalize_input
    if args.phase_loss_weight is not None:
        overrides["PHASE_LOSS_WEIGHT"] = args.phase_loss_weight
    if args.phase_label_smoothing is not None:
        overrides["PHASE_LABEL_SMOOTHING"] = args.phase_label_smoothing
    if args.phase_only is not None:
        overrides["PHASE_ONLY"] = args.phase_only
    if args.phase_only_warmup_epochs is not None:
        overrides["PHASE_ONLY_WARMUP_EPOCHS"] = args.phase_only_warmup_epochs
    if args.phase_backbone_freeze_epochs is not None:
        overrides["PHASE_BACKBONE_FREEZE_EPOCHS"] = args.phase_backbone_freeze_epochs
    if args.backbone_lr_mult is not None:
        overrides["BACKBONE_LR_MULT"] = args.backbone_lr_mult
    if args.phase_soft_sigma is not None:
        overrides["PHASE_SOFT_SIGMA"] = args.phase_soft_sigma
    if args.phase_soft_radius is not None:
        overrides["PHASE_SOFT_RADIUS"] = args.phase_soft_radius
    if args.phase_hard_index_weight is not None:
        overrides["PHASE_HARD_INDEX_WEIGHT"] = args.phase_hard_index_weight
    if args.phase_frame_ce_weight is not None:
        overrides["PHASE_FRAME_CE_WEIGHT"] = args.phase_frame_ce_weight
    if args.phase_frame_radius is not None:
        overrides["PHASE_FRAME_RADIUS"] = args.phase_frame_radius
    if args.phase_attn_align_weight is not None:
        overrides["PHASE_ATTN_ALIGN_WEIGHT"] = args.phase_attn_align_weight
    if args.phase_attn_align_sigma is not None:
        overrides["PHASE_ATTN_ALIGN_SIGMA"] = args.phase_attn_align_sigma
    if args.phase_attn_align_radius is not None:
        overrides["PHASE_ATTN_ALIGN_RADIUS"] = args.phase_attn_align_radius
    if args.phase_attn_index_weight is not None:
        overrides["PHASE_ATTN_INDEX_WEIGHT"] = args.phase_attn_index_weight
    if args.phase_attn_order_weight is not None:
        overrides["PHASE_ATTN_ORDER_WEIGHT"] = args.phase_attn_order_weight
    if args.phase_attn_entropy_weight is not None:
        overrides["PHASE_ATTN_ENTROPY_WEIGHT"] = args.phase_attn_entropy_weight
    if args.phase_attn_min_gap is not None:
        overrides["PHASE_ATTN_MIN_GAP"] = args.phase_attn_min_gap
    if args.phase_pair_index_weight is not None:
        overrides["PHASE_PAIR_INDEX_WEIGHT"] = args.phase_pair_index_weight
    if args.phase_pair_order_weight is not None:
        overrides["PHASE_PAIR_ORDER_WEIGHT"] = args.phase_pair_order_weight
    if args.phase_pair_min_gap is not None:
        overrides["PHASE_PAIR_MIN_GAP"] = args.phase_pair_min_gap
    if args.phase_unfreeze_lr_mult is not None:
        overrides["PHASE_UNFREEZE_LR_MULT"] = args.phase_unfreeze_lr_mult
    if args.weight_decay is not None:
        overrides["WEIGHT_DECAY"] = args.weight_decay
    if args.max_grad_norm is not None:
        overrides["MAX_GRAD_NORM"] = args.max_grad_norm
    if args.clip_period is not None:
        overrides["CLIP_PERIOD"] = args.clip_period
    if args.clip_eval_mode is not None:
        overrides["CLIP_EVAL_MODE"] = args.clip_eval_mode

    if bool(getattr(args, "train_stage123", False)):
        logger.info("Enabled Stage1-3 end-to-end training profile (joint EF + phase)")
        # Stage1/2/3 should train jointly without disabling EF regression.
        # Respect explicit CLI values when present.
        if args.phase_only is None:
            overrides["PHASE_ONLY"] = False
        if args.phase_loss_weight is None:
            overrides["PHASE_LOSS_WEIGHT"] = 2.0
        if args.phase_only_warmup_epochs is None:
            overrides["PHASE_ONLY_WARMUP_EPOCHS"] = 5
        if args.phase_soft_sigma is None:
            overrides["PHASE_SOFT_SIGMA"] = 3.0
        if args.phase_soft_radius is None:
            overrides["PHASE_SOFT_RADIUS"] = 9
        if args.phase_hard_index_weight is None:
            overrides["PHASE_HARD_INDEX_WEIGHT"] = 0.05
        if args.phase_frame_ce_weight is None:
            overrides["PHASE_FRAME_CE_WEIGHT"] = 0.15
        if args.phase_frame_radius is None:
            overrides["PHASE_FRAME_RADIUS"] = 3
        if args.phase_attn_align_weight is None:
            overrides["PHASE_ATTN_ALIGN_WEIGHT"] = 0.70
        if args.phase_attn_align_sigma is None:
            overrides["PHASE_ATTN_ALIGN_SIGMA"] = 3.0
        if args.phase_attn_align_radius is None:
            overrides["PHASE_ATTN_ALIGN_RADIUS"] = 9
        if args.phase_attn_index_weight is None:
            overrides["PHASE_ATTN_INDEX_WEIGHT"] = 0.40
        if args.phase_attn_order_weight is None:
            overrides["PHASE_ATTN_ORDER_WEIGHT"] = 0.20
        if args.phase_attn_entropy_weight is None:
            overrides["PHASE_ATTN_ENTROPY_WEIGHT"] = 0.03
        if args.phase_attn_min_gap is None:
            overrides["PHASE_ATTN_MIN_GAP"] = 15
        if args.phase_pair_index_weight is None:
            overrides["PHASE_PAIR_INDEX_WEIGHT"] = 0.40
        if args.phase_pair_order_weight is None:
            overrides["PHASE_PAIR_ORDER_WEIGHT"] = 0.25
        if args.phase_pair_min_gap is None:
            overrides["PHASE_PAIR_MIN_GAP"] = 15
        overrides["PHASE_BACKBONE_FREEZE_EPOCHS"] = 0

    for key, value in overrides.items():
        setattr(config, key, value)
        logger.info("Runtime override: %s=%s", key, value)

    if hasattr(config, "CLIP_PERIOD"):
        clip_period = int(getattr(config, "CLIP_PERIOD", 1))
        if clip_period < 1:
            setattr(config, "CLIP_PERIOD", 1)
            logger.warning("Clamped CLIP_PERIOD from %d to 1", clip_period)

    attn_align_key = "PHASE_ATTN_ALIGN_WEIGHT"
    if hasattr(config, attn_align_key):
        attn_align_value = float(getattr(config, attn_align_key, 0.0))
        if attn_align_value < 0.0:
            setattr(config, attn_align_key, 0.0)
            logger.warning("Clamped %s from %.3f to 0.000", attn_align_key, attn_align_value)
        elif attn_align_value > 2.0:
            setattr(config, attn_align_key, 2.0)
            logger.warning("Clamped %s from %.3f to 2.000", attn_align_key, attn_align_value)

    for loss_key in ("PHASE_ATTN_INDEX_WEIGHT", "PHASE_ATTN_ORDER_WEIGHT", "PHASE_ATTN_ENTROPY_WEIGHT", "PHASE_PAIR_INDEX_WEIGHT", "PHASE_PAIR_ORDER_WEIGHT"):
        if hasattr(config, loss_key):
            loss_value = float(getattr(config, loss_key, 0.0))
            if loss_value < 0.0:
                setattr(config, loss_key, 0.0)
                logger.warning("Clamped %s from %.3f to 0.000", loss_key, loss_value)
            elif loss_value > 2.0:
                setattr(config, loss_key, 2.0)
                logger.warning("Clamped %s from %.3f to 2.000", loss_key, loss_value)

    for min_gap_key in ("PHASE_ATTN_MIN_GAP", "PHASE_PAIR_MIN_GAP"):
        if hasattr(config, min_gap_key):
            min_gap_value = int(getattr(config, min_gap_key, 1))
            if min_gap_value < 1:
                setattr(config, min_gap_key, 1)
                logger.warning("Clamped %s from %d to 1", min_gap_key, min_gap_value)

    return overrides


def is_cuda_runtime():
    return torch.cuda.is_available() and str(config.DEVICE).startswith("cuda")


def is_phase_only_mode():
    return bool(getattr(config, "PHASE_ONLY", False))


def is_phase_only_epoch(epoch_idx):
    if is_phase_only_mode():
        return True
    warmup_epochs = max(0, int(getattr(config, "PHASE_ONLY_WARMUP_EPOCHS", 0)))
    return int(epoch_idx) < warmup_epochs


def make_grad_scaler(amp_enabled):
    if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
        return torch.amp.GradScaler("cuda", enabled=amp_enabled)
    return torch.cuda.amp.GradScaler(enabled=amp_enabled)


def autocast_context(amp_enabled):
    if hasattr(torch, "amp") and hasattr(torch.amp, "autocast"):
        return torch.amp.autocast(device_type="cuda", enabled=amp_enabled)
    return torch.cuda.amp.autocast(enabled=amp_enabled)


def setup_performance_backends(logger):
    if not is_cuda_runtime():
        logger.info("CUDA optimizations disabled (running on %s)", config.DEVICE)
        return

    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high")

    torch.backends.cuda.matmul.allow_tf32 = bool(getattr(config, "ENABLE_TF32", True))
    torch.backends.cudnn.allow_tf32 = bool(getattr(config, "ENABLE_TF32", True))
    torch.backends.cudnn.benchmark = bool(getattr(config, "CUDNN_BENCHMARK", True))

    logger.info(
        "CUDA backend config | TF32=%s | cuDNN benchmark=%s",
        torch.backends.cuda.matmul.allow_tf32,
        torch.backends.cudnn.benchmark,
    )


def setup_logger():
    """Configure file + console logging."""
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"training_{timestamp}.log")

    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()

    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.DEBUG)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)

    if sys.platform == "win32":
        import io
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    return logger, log_file


def dataloader_kwargs(shuffle):
    num_workers = int(getattr(config, "NUM_WORKERS", 0))
    use_cuda = is_cuda_runtime()

    kwargs = {
        "batch_size": config.BATCH_SIZE,
        "shuffle": shuffle,
        "num_workers": num_workers,
        "pin_memory": bool(getattr(config, "PIN_MEMORY", True)) and use_cuda,
    }

    if num_workers > 0:
        kwargs["persistent_workers"] = bool(getattr(config, "PERSISTENT_WORKERS", True))
        prefetch_factor = getattr(config, "PREFETCH_FACTOR", None)
        if prefetch_factor is not None:
            kwargs["prefetch_factor"] = int(prefetch_factor)

    return kwargs


def build_dataloaders():
    """Create train/val/test dataloaders."""
    clip_period = int(getattr(config, "CLIP_PERIOD", 1))
    clip_eval_mode = str(getattr(config, "CLIP_EVAL_MODE", "center"))

    train_dataset = EchoDataset(
        config.DATA_DIR,
        split="TRAIN",
        num_frames=config.NUM_FRAMES,
        max_videos=config.MAX_VIDEOS,
        normalize_input=bool(getattr(config, "NORMALIZE_INPUT", True)),
        clip_period=clip_period,
        clip_eval_mode=clip_eval_mode,
    )
    val_dataset = EchoDataset(
        config.DATA_DIR,
        split="VAL",
        num_frames=config.NUM_FRAMES,
        max_videos=config.MAX_VIDEOS,
        normalize_input=bool(getattr(config, "NORMALIZE_INPUT", True)),
        clip_period=clip_period,
        clip_eval_mode=clip_eval_mode,
    )
    test_dataset = EchoDataset(
        config.DATA_DIR,
        split="TEST",
        num_frames=config.NUM_FRAMES,
        max_videos=config.MAX_VIDEOS,
        normalize_input=bool(getattr(config, "NORMALIZE_INPUT", True)),
        clip_period=clip_period,
        clip_eval_mode=clip_eval_mode,
    )

    train_loader = DataLoader(train_dataset, **dataloader_kwargs(shuffle=True))
    val_loader = DataLoader(val_dataset, **dataloader_kwargs(shuffle=False))
    test_loader = DataLoader(test_dataset, **dataloader_kwargs(shuffle=False))
    return train_loader, val_loader, test_loader


def get_pipeline(model):
    return model.pipeline if hasattr(model, "pipeline") else model


def set_backbone_trainable(model, trainable):
    pipeline = get_pipeline(model)
    if not hasattr(pipeline, "stage1"):
        return False

    for p in pipeline.stage1.parameters():
        p.requires_grad = trainable
    return True


def maybe_freeze_ef_head(model, logger):
    if not is_phase_only_mode():
        return

    pipeline = get_pipeline(model)
    ef_head = pipeline.ef_regressor if hasattr(pipeline, "ef_regressor") else None

    if ef_head is None:
        logger.info("Phase-only mode enabled, but EF head was not found for freezing")
        return

    for p in ef_head.parameters():
        p.requires_grad = False
    logger.info("Phase-only mode: EF head frozen")


def build_optimizer(model, logger):
    base_lr = float(config.LEARNING_RATE)
    backbone_mult = float(getattr(config, "BACKBONE_LR_MULT", 1.0))
    pipeline = get_pipeline(model)

    param_groups = []
    used_ids = set()

    if hasattr(pipeline, "stage1"):
        backbone_params = [p for p in pipeline.stage1.parameters() if p.requires_grad]
        if backbone_params:
            param_groups.append({"params": backbone_params, "lr": base_lr * backbone_mult})
            used_ids.update(id(p) for p in backbone_params)

    head_params = [p for p in model.parameters() if p.requires_grad and id(p) not in used_ids]
    if head_params:
        param_groups.append({"params": head_params, "lr": base_lr})

    if not param_groups:
        raise RuntimeError("No trainable parameters found for optimizer")

    weight_decay = float(getattr(config, "WEIGHT_DECAY", 0.0))
    optimizer = torch.optim.AdamW(param_groups, weight_decay=weight_decay)

    group_sizes = [sum(p.numel() for p in g["params"]) for g in param_groups]
    group_lrs = [g["lr"] for g in param_groups]
    logger.info("Optimizer param groups | sizes=%s | lrs=%s | weight_decay=%s", group_sizes, group_lrs, weight_decay)
    return optimizer


def build_model_stack(logger):
    """Create model, optimizer and losses."""
    model = EFModel(num_frames=config.NUM_FRAMES).to(config.DEVICE)

    maybe_freeze_ef_head(model, logger)

    if is_phase_only_mode() and int(getattr(config, "PHASE_BACKBONE_FREEZE_EPOCHS", 0)) > 0:
        if set_backbone_trainable(model, False):
            logger.info("Phase-only mode: Stage1 backbone frozen for warmup epochs")

    optimizer = build_optimizer(model, logger)

    mse_loss = nn.MSELoss()
    phase_regression_loss = nn.MSELoss()  # MSE for continuous phase prediction

    amp_enabled = bool(getattr(config, "USE_MIXED_PRECISION", False)) and is_cuda_runtime()
    scaler = make_grad_scaler(amp_enabled)
    return model, optimizer, mse_loss, phase_regression_loss, amp_enabled, scaler


def log_stage_trainability(model, logger):
    """Log total/trainable params for Stage1/2/3 and EF head."""
    pipeline = get_pipeline(model)

    for name in ("stage1", "stage2", "stage3", "ef_regressor"):
        module = getattr(pipeline, name, None)
        if module is None:
            logger.info("%s: not present", name)
            continue

        total = sum(p.numel() for p in module.parameters())
        trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
        logger.info("%s params | trainable=%d / total=%d", name, trainable, total)


def build_soft_temporal_targets(indices, num_frames, device, sigma, radius):
    """Build Gaussian-like soft targets centered at ground-truth indices."""
    positions = torch.arange(num_frames, device=device, dtype=torch.float32).unsqueeze(0)
    centers = indices.to(device=device, dtype=torch.float32).unsqueeze(1)
    dist = torch.abs(positions - centers)

    weights = torch.exp(-(dist ** 2) / (2.0 * sigma * sigma))

    if radius is not None and radius > 0:
        weights = weights * (dist <= float(radius)).to(weights.dtype)

    sums = weights.sum(dim=1, keepdim=True)
    fallback = (sums.squeeze(1) <= 0)
    if fallback.any():
        weights[fallback] = 0.0
        weights[fallback, indices[fallback]] = 1.0
        sums = weights.sum(dim=1, keepdim=True)

    return weights / sums.clamp_min(1e-8)


def build_frame_phase_targets(ed_idx, es_idx, num_frames, radius):
    """Build per-frame 3-class targets: 0=background, 1=ED neighborhood, 2=ES neighborhood."""
    positions = torch.arange(num_frames, device=ed_idx.device).unsqueeze(0)
    ed_dist = torch.abs(positions - ed_idx.unsqueeze(1))
    es_dist = torch.abs(positions - es_idx.unsqueeze(1))

    targets = torch.zeros((ed_idx.shape[0], num_frames), dtype=torch.long, device=ed_idx.device)
    ed_mask = ed_dist <= radius
    es_mask = es_dist <= radius

    targets[ed_mask & ~es_mask] = 1
    targets[es_mask & ~ed_mask] = 2

    overlap = ed_mask & es_mask
    if overlap.any():
        choose_ed = ed_dist[overlap] <= es_dist[overlap]
        targets[overlap] = torch.where(
            choose_ed,
            torch.ones_like(targets[overlap]),
            torch.full_like(targets[overlap], 2),
        )

    return targets


def build_cyclic_phase_targets(ed_idx, es_idx, num_frames):
    """Build per-frame sin/cos phase targets with ED=0 and ES=pi."""
    positions = torch.arange(num_frames, device=ed_idx.device, dtype=torch.float32).unsqueeze(0)
    ed = ed_idx.to(dtype=torch.float32).unsqueeze(1)
    es = es_idx.to(dtype=torch.float32).unsqueeze(1)
    half_cycle = torch.abs(es - ed).clamp_min(1.0)
    raw_phase = (positions - ed) / (2.0 * half_cycle)
    phase = torch.remainder(raw_phase, 1.0)
    theta = phase * (2.0 * math.pi)
    return torch.stack([torch.sin(theta), torch.cos(theta)], dim=-1)


def phase_logits_from_output(phase_pred):
    if phase_pred.ndim == 3 and phase_pred.shape[-1] >= 3:
        return phase_pred[:, :, :3]
    return None


def phase_vector_from_output(phase_pred):
    if phase_pred.ndim == 3 and phase_pred.shape[-1] >= 5:
        return phase_pred[:, :, 3:5]
    return None


def _attention_heads_and_summary(attention):
    """Normalize attention to (B, T, H) heads plus a single summary curve (B, T)."""
    if attention is None:
        return None, None

    attn = attention.float()
    if attn.ndim == 2:
        attn_heads = attn.unsqueeze(-1)
    elif attn.ndim == 3 and attn.shape[-1] > 0:
        attn_heads = attn
    else:
        return None, None

    attn_heads = attn_heads / attn_heads.sum(dim=1, keepdim=True).clamp_min(1e-8)
    attn_summary = attn_heads.mean(dim=-1)
    attn_summary = attn_summary / attn_summary.sum(dim=1, keepdim=True).clamp_min(1e-8)
    return attn_heads, attn_summary


def compute_attention_alignment_loss(attention, ed_idx, es_idx):
    """KL alignment between Stage2 attention heads and ED/ES-centered soft targets."""
    attn_heads, _ = _attention_heads_and_summary(attention)
    if attn_heads is None:
        return torch.zeros((), device=ed_idx.device)

    sigma = float(getattr(config, "PHASE_ATTN_ALIGN_SIGMA", 0.0))
    if sigma <= 0.0:
        sigma = max(0.5, float(getattr(config, "PHASE_SOFT_SIGMA", 1.5)))

    radius = int(getattr(config, "PHASE_ATTN_ALIGN_RADIUS", int(getattr(config, "PHASE_SOFT_RADIUS", 0))))
    num_frames = attn_heads.shape[1]

    ed_target = build_soft_temporal_targets(ed_idx, num_frames, attn_heads.device, sigma, radius)
    es_target = build_soft_temporal_targets(es_idx, num_frames, attn_heads.device, sigma, radius)

    if attn_heads.shape[-1] >= 2:
        ed_attn = attn_heads[:, :, 0]
        es_attn = attn_heads[:, :, 1]
        ed_kl = F.kl_div(torch.log(ed_attn.clamp_min(1e-8)), ed_target, reduction="batchmean")
        es_kl = F.kl_div(torch.log(es_attn.clamp_min(1e-8)), es_target, reduction="batchmean")
        return 0.5 * (ed_kl + es_kl)

    merged_target = 0.5 * (ed_target + es_target)
    merged_target = merged_target / merged_target.sum(dim=1, keepdim=True).clamp_min(1e-8)
    return F.kl_div(torch.log(attn_heads[:, :, 0].clamp_min(1e-8)), merged_target, reduction="batchmean")


def compute_attention_index_loss(attention, ed_idx, es_idx):
    """Directly supervise Stage2 attention heads toward ED/ES indices and ordering."""
    attn_heads, _ = _attention_heads_and_summary(attention)
    if attn_heads is None:
        zero = torch.zeros((), device=ed_idx.device)
        return zero, zero

    num_frames = attn_heads.shape[1]
    if num_frames <= 0:
        zero = torch.zeros((), device=ed_idx.device)
        return zero, zero

    if attn_heads.shape[-1] >= 2:
        ed_attn = attn_heads[:, :, 0]
        es_attn = attn_heads[:, :, 1]
    else:
        ed_attn = attn_heads[:, :, 0]
        es_attn = attn_heads[:, :, 0]

    positions = torch.arange(num_frames, device=attn_heads.device, dtype=attn_heads.dtype).unsqueeze(0)
    denom = float(max(1, num_frames - 1))

    ed_expect = (ed_attn * positions).sum(dim=1) / denom
    es_expect = (es_attn * positions).sum(dim=1) / denom
    ed_target = ed_idx.to(attn_heads.device, dtype=attn_heads.dtype) / denom
    es_target = es_idx.to(attn_heads.device, dtype=attn_heads.dtype) / denom

    ed_loss = F.smooth_l1_loss(ed_expect, ed_target)
    es_loss = F.smooth_l1_loss(es_expect, es_target)
    index_loss = 0.5 * (ed_loss + es_loss)

    min_gap = float(max(1, int(getattr(config, "PHASE_ATTN_MIN_GAP", 1)))) / denom
    order_loss = torch.relu(min_gap - (es_expect - ed_expect)).mean()
    return index_loss, order_loss


def compute_attention_entropy_loss(attention):
    """Minimize normalized Stage2 attention entropy to discourage diffuse temporal focus."""
    attn_heads, _ = _attention_heads_and_summary(attention)
    if attn_heads is None or attn_heads.shape[1] <= 1:
        device = attention.device if attention is not None else config.DEVICE
        return torch.zeros((), device=device)

    attn_safe = attn_heads.clamp_min(1e-8)
    entropy = -(attn_safe * torch.log(attn_safe)).sum(dim=1) / math.log(attn_heads.shape[1])
    return entropy.mean()


def compute_phase_pair_regularizers(phase_logits, ed_idx, es_idx):
    """Regularize Stage3 ED/ES expectations directly in the same score space used at inference."""
    if phase_logits.ndim == 3 and phase_logits.shape[-1] > 3:
        phase_logits = phase_logits[:, :, :3]
    ed_logits = phase_logits[:, :, 1] - phase_logits[:, :, 0]
    es_logits = phase_logits[:, :, 2] - phase_logits[:, :, 0]

    ed_probs = F.softmax(ed_logits, dim=1)
    es_probs = F.softmax(es_logits, dim=1)

    num_frames = ed_probs.shape[1]
    if num_frames <= 0:
        zero = torch.zeros((), device=phase_logits.device)
        return zero, zero

    positions = torch.arange(num_frames, device=phase_logits.device, dtype=phase_logits.dtype).unsqueeze(0)
    denom = float(max(1, num_frames - 1))

    ed_expect = (ed_probs * positions).sum(dim=1) / denom
    es_expect = (es_probs * positions).sum(dim=1) / denom
    ed_target = ed_idx.to(device=phase_logits.device, dtype=phase_logits.dtype) / denom
    es_target = es_idx.to(device=phase_logits.device, dtype=phase_logits.dtype) / denom

    index_loss = 0.5 * (
        F.smooth_l1_loss(ed_expect, ed_target) +
        F.smooth_l1_loss(es_expect, es_target)
    )

    min_gap = float(max(1, int(getattr(config, "PHASE_PAIR_MIN_GAP", 1)))) / denom
    order_loss = torch.relu(min_gap - (es_expect - ed_expect)).mean()
    return index_loss, order_loss


def compute_continuous_phase_loss(phase_pred, phase_labels, phase_regression_loss_fn):
    """
    Compute phase loss.

    Current models emit ED/ES logits plus sin/cos vectors. Legacy scalar phase
    checkpoints still use MSE against scalar labels.
    """
    logits = phase_logits_from_output(phase_pred)
    if logits is not None:
        return F.cross_entropy(
            logits.reshape(-1, logits.shape[-1]),
            phase_labels.long().reshape(-1),
            label_smoothing=float(getattr(config, "PHASE_LABEL_SMOOTHING", 0.0)),
        )
    return phase_regression_loss_fn(phase_pred, phase_labels)



def move_batch_to_device(videos, efs, ed_idx, es_idx):
    non_blocking = bool(getattr(config, "NON_BLOCKING_TRANSFER", True)) and is_cuda_runtime()
    videos = videos.to(config.DEVICE, non_blocking=non_blocking)
    efs = efs.to(config.DEVICE, non_blocking=non_blocking)
    ed_idx = ed_idx.to(config.DEVICE, non_blocking=non_blocking)
    es_idx = es_idx.to(config.DEVICE, non_blocking=non_blocking)
    return videos, efs, ed_idx, es_idx


def _phase_confidence_from_output(phase_pred, pred_ed_idx, pred_es_idx):
    if phase_pred.ndim != 3:
        return torch.ones_like(pred_ed_idx, dtype=torch.float32), torch.ones_like(pred_es_idx, dtype=torch.float32)

    if phase_pred.shape[-1] >= 7:
        ed_scores = phase_pred[:, :, 5] + phase_pred[:, :, 1]
        es_scores = phase_pred[:, :, 6] + phase_pred[:, :, 2]
    elif phase_pred.shape[-1] >= 3:
        ed_scores = phase_pred[:, :, 1]
        es_scores = phase_pred[:, :, 2]
    else:
        return torch.ones_like(pred_ed_idx, dtype=torch.float32), torch.ones_like(pred_es_idx, dtype=torch.float32)

    ed_probs = F.softmax(ed_scores.float(), dim=1)
    es_probs = F.softmax(es_scores.float(), dim=1)
    ed_conf = ed_probs.gather(1, pred_ed_idx.view(-1, 1).clamp(0, ed_probs.shape[1] - 1)).squeeze(1)
    es_conf = es_probs.gather(1, pred_es_idx.view(-1, 1).clamp(0, es_probs.shape[1] - 1)).squeeze(1)
    return ed_conf, es_conf


def _weighted_consensus_frame(candidates, weights, total_frames):
    candidates = np.asarray(candidates, dtype=np.int32)
    weights = np.asarray(weights, dtype=np.float64)
    total_frames = int(max(1, total_frames))
    if candidates.size == 0:
        return 0
    candidates = np.clip(candidates, 0, total_frames - 1)
    if weights.size != candidates.size or not np.isfinite(weights).all() or float(weights.sum()) <= 0.0:
        weights = np.ones(candidates.shape[0], dtype=np.float64)

    vote_curve = np.zeros(total_frames, dtype=np.float64)
    positions = np.arange(total_frames, dtype=np.float64)
    sigma = max(1.0, float(getattr(config, "PHASE_CONSENSUS_SIGMA", 2.0)))
    for frame_idx, weight in zip(candidates, weights):
        vote_curve += float(weight) * np.exp(-((positions - float(frame_idx)) ** 2) / (2.0 * sigma * sigma))
    return int(np.argmax(vote_curve))


def evaluate_all_clip_consensus(model, dataset, amp_enabled):
    """GT-free deployment-style evaluation: scan all clips and vote in original-frame space."""
    model.eval()
    phase_only = is_phase_only_mode()
    phase_acc_threshold = int(getattr(config, "PHASE_ACC_THRESHOLD_FRAMES", 4))
    clip_batch_size = max(1, int(getattr(config, "CLIP_EVAL_BATCH_SIZE", min(8, config.BATCH_SIZE))))

    total_samples = 0
    total_mae = 0.0
    total_mse = 0.0
    total_ed_acc = 0.0
    total_es_acc = 0.0
    total_joint_acc = 0.0
    total_ed_mae_frames = 0.0
    total_es_mae_frames = 0.0
    total_clips = 0.0

    with torch.no_grad():
        for sample_index in range(len(dataset)):
            row = dataset.filelist.iloc[sample_index]
            file_stem = str(row["FileName"])
            file_name_ext = file_stem + ".avi"
            video_path = os.path.join(dataset.data_dir, "Videos", file_name_ext)
            ed_orig = int(dataset.phase_dict[file_name_ext]["ed"])
            es_orig = int(dataset.phase_dict[file_name_ext]["es"])
            ef_gt = float(row["EF"]) / 100.0

            clips, sampled_indices = dataset.load_video_clips(video_path, mode="all")
            clips = clips.to(config.DEVICE)
            ef_preds = []
            ed_candidates = []
            es_candidates = []
            ed_weights = []
            es_weights = []

            for start in range(0, clips.shape[0], clip_batch_size):
                clip_batch = clips[start : start + clip_batch_size]
                with autocast_context(amp_enabled):
                    ef_pred, _attention, phase_pred = model(clip_batch)
                pred_ed_idx, pred_es_idx = Stage3PhaseDetector.predict_indices(phase_pred)
                ed_conf, es_conf = _phase_confidence_from_output(phase_pred, pred_ed_idx, pred_es_idx)

                ef_preds.extend(ef_pred.detach().float().cpu().tolist())
                batch_indices = sampled_indices[start : start + clip_batch.shape[0]]
                for local_i in range(clip_batch.shape[0]):
                    ed_clip = int(pred_ed_idx[local_i].detach().cpu().item())
                    es_clip = int(pred_es_idx[local_i].detach().cpu().item())
                    ed_candidates.append(int(batch_indices[local_i, np.clip(ed_clip, 0, batch_indices.shape[1] - 1)]))
                    es_candidates.append(int(batch_indices[local_i, np.clip(es_clip, 0, batch_indices.shape[1] - 1)]))
                    ed_weights.append(float(ed_conf[local_i].detach().cpu().item()))
                    es_weights.append(float(es_conf[local_i].detach().cpu().item()))

            total_video_frames = int(np.max(sampled_indices)) + 1 if sampled_indices.size else 1
            pred_ed_orig = _weighted_consensus_frame(ed_candidates, ed_weights, total_video_frames)
            pred_es_orig = _weighted_consensus_frame(es_candidates, es_weights, total_video_frames)
            ef_pred = float(np.mean(ef_preds)) if ef_preds else float("nan")

            if not phase_only and np.isfinite(ef_pred):
                ef_err = abs(ef_pred - ef_gt)
                total_mae += ef_err
                total_mse += ef_err * ef_err

            ed_mae = abs(pred_ed_orig - ed_orig)
            es_mae = abs(pred_es_orig - es_orig)
            total_ed_mae_frames += float(ed_mae)
            total_es_mae_frames += float(es_mae)
            total_ed_acc += float(ed_mae <= phase_acc_threshold)
            total_es_acc += float(es_mae <= phase_acc_threshold)
            total_joint_acc += float(ed_mae <= phase_acc_threshold and es_mae <= phase_acc_threshold)
            total_clips += float(clips.shape[0])
            total_samples += 1

    if total_samples == 0:
        raise RuntimeError("No samples found during all-clip consensus evaluation")

    return {
        "ef_mae": (total_mae / total_samples) if not phase_only else float("nan"),
        "ef_rmse": ((total_mse / total_samples) ** 0.5) if not phase_only else float("nan"),
        "phase_mse": float("nan"),
        "ed_acc": total_ed_acc / total_samples,
        "es_acc": total_es_acc / total_samples,
        "joint_acc": total_joint_acc / total_samples,
        "ed_mae_frames": total_ed_mae_frames / total_samples,
        "es_mae_frames": total_es_mae_frames / total_samples,
        "stage1_temporal_tokens": float(getattr(config, "NUM_FRAMES", 0)),
        "stage1_feature_norm": float("nan"),
        "stage1_temporal_std": float("nan"),
        "stage2_temporal_tokens": float(getattr(config, "NUM_FRAMES", 0)),
        "stage2_attention_entropy": float("nan"),
        "stage2_attention_peak": float("nan"),
        "stage2_peak_to_event_mae_frames": float("nan"),
        "stage2_ed_es_feature_distance": float("nan"),
        "stage3_ed_index_ce": float("nan"),
        "stage3_es_index_ce": float("nan"),
        "eval_clips_per_video": total_clips / total_samples,
    }


def evaluate(model, loader, amp_enabled):
    """Evaluate EF regression and phase prediction."""
    if str(getattr(config, "CLIP_EVAL_MODE", "center")).lower() == "all":
        return evaluate_all_clip_consensus(model, loader.dataset, amp_enabled)

    model.eval()

    total_samples = 0
    total_mae = 0.0
    total_mse = 0.0
    total_phase_loss = 0.0
    
    # Phase prediction metrics
    total_ed_acc = 0.0
    total_es_acc = 0.0
    total_joint_acc = 0.0
    total_ed_mae_frames = 0.0
    total_es_mae_frames = 0.0
    total_s1_tokens = 0.0
    total_s1_norm = 0.0
    total_s1_std = 0.0
    total_s2_tokens = 0.0
    total_s2_entropy = 0.0
    total_s2_peak = 0.0
    total_s2_peak_mae = 0.0
    total_s2_feat_dist = 0.0
    total_s3_ed_ce = 0.0
    total_s3_es_ce = 0.0

    phase_only = is_phase_only_mode()
    phase_acc_threshold = int(getattr(config, "PHASE_ACC_THRESHOLD_FRAMES", 4))

    with torch.no_grad():
        for videos, efs, ed_idx, es_idx in loader:
            videos, efs, ed_idx, es_idx = move_batch_to_device(videos, efs, ed_idx, es_idx)

            with autocast_context(amp_enabled):
                model_out = model(videos, return_stage_outputs=True)

            if isinstance(model_out, tuple) and len(model_out) == 4:
                ef_pred, attention, phase_pred, stage_outputs = model_out
            else:
                ef_pred, attention, phase_pred = model_out
                stage_outputs = {}

            batch_size = videos.size(0)

            if not phase_only:
                mae = torch.abs(ef_pred - efs).mean()
                mse = torch.mean((ef_pred - efs) ** 2)
                total_mae += mae.item() * batch_size
                total_mse += mse.item() * batch_size

            num_frames = videos.size(2)  # videos shape: (batch, channels, frames, height, width)
            phase_labels = build_frame_phase_targets(
                ed_idx,
                es_idx,
                num_frames,
                radius=max(0, int(getattr(config, "PHASE_FRAME_RADIUS", 1))),
            )
            
            phase_loss = compute_continuous_phase_loss(phase_pred, phase_labels, nn.MSELoss())
            phase_vec = phase_vector_from_output(phase_pred)
            if phase_vec is not None:
                cyclic_targets = build_cyclic_phase_targets(ed_idx, es_idx, num_frames)
                phase_loss = phase_loss + F.smooth_l1_loss(phase_vec, cyclic_targets)
            total_phase_loss += phase_loss.item() * batch_size

            pred_ed_idx, pred_es_idx = Stage3PhaseDetector.predict_indices(phase_pred)
            
            # Compute ED/ES accuracy metrics
            ed_mae = torch.abs(pred_ed_idx.float() - ed_idx.float())
            es_mae = torch.abs(pred_es_idx.float() - es_idx.float())
            
            total_ed_mae_frames += ed_mae.mean().item() * batch_size
            total_es_mae_frames += es_mae.mean().item() * batch_size
            
            # Accuracy: within threshold frames
            ed_acc_batch = (ed_mae <= phase_acc_threshold).float().mean()
            es_acc_batch = (es_mae <= phase_acc_threshold).float().mean()
            joint_acc_batch = ((ed_mae <= phase_acc_threshold) & (es_mae <= phase_acc_threshold)).float().mean()
            
            total_ed_acc += ed_acc_batch.item() * batch_size
            total_es_acc += es_acc_batch.item() * batch_size
            total_joint_acc += joint_acc_batch.item() * batch_size

            stage1_features = stage_outputs.get("stage1_features")
            temporal_features = stage_outputs.get("stage2_temporal_features")
            if stage1_features is not None:
                total_s1_tokens += float(stage1_features.shape[-1]) * batch_size
                total_s1_norm += stage1_features.float().norm(dim=1).mean().item() * batch_size
                total_s1_std += stage1_features.float().std(dim=-1).mean().item() * batch_size

            attn_heads, attn_summary = _attention_heads_and_summary(attention)
            if attn_heads is not None:
                attn_safe = attn_heads.clamp_min(1e-8)
                entropy = (-(attn_safe * torch.log(attn_safe)).sum(dim=1) / math.log(attn_heads.shape[1])).mean()
                total_s2_entropy += entropy.item() * batch_size
                total_s2_peak += attn_heads.amax(dim=1).mean().item() * batch_size
                if attn_heads.shape[-1] >= 2:
                    peak_ed = torch.argmax(attn_heads[:, :, 0], dim=1)
                    peak_es = torch.argmax(attn_heads[:, :, 1], dim=1)
                    peak_mae = 0.5 * (
                        torch.abs(peak_ed.float() - ed_idx.float()).mean() +
                        torch.abs(peak_es.float() - es_idx.float()).mean()
                    )
                    total_s2_peak_mae += peak_mae.item() * batch_size

            if temporal_features is not None:
                total_s2_tokens += float(temporal_features.shape[1]) * batch_size
                gather_ed = ed_idx.clamp(0, temporal_features.shape[1] - 1).view(-1, 1, 1).expand(-1, 1, temporal_features.shape[2])
                gather_es = es_idx.clamp(0, temporal_features.shape[1] - 1).view(-1, 1, 1).expand(-1, 1, temporal_features.shape[2])
                ed_feat = temporal_features.gather(1, gather_ed).squeeze(1)
                es_feat = temporal_features.gather(1, gather_es).squeeze(1)
                total_s2_feat_dist += torch.norm(ed_feat - es_feat, dim=1).mean().item() * batch_size

            logits = phase_logits_from_output(phase_pred)
            if logits is not None:
                total_s3_ed_ce += F.cross_entropy(logits[:, :, 1], ed_idx.clamp(0, logits.shape[1] - 1)).item() * batch_size
                total_s3_es_ce += F.cross_entropy(logits[:, :, 2], es_idx.clamp(0, logits.shape[1] - 1)).item() * batch_size

            total_samples += batch_size

    if total_samples == 0:
        raise RuntimeError("No samples found during evaluation")

    metrics = {
        "ef_mae": (total_mae / total_samples) if not phase_only else float("nan"),
        "ef_rmse": ((total_mse / total_samples) ** 0.5) if not phase_only else float("nan"),
        "phase_mse": total_phase_loss / total_samples,
        "ed_acc": total_ed_acc / total_samples,
        "es_acc": total_es_acc / total_samples,
        "joint_acc": total_joint_acc / total_samples,
        "ed_mae_frames": total_ed_mae_frames / total_samples,
        "es_mae_frames": total_es_mae_frames / total_samples,
        # Stage-specific diagnostic metrics (set to defaults if not available)
        "stage1_temporal_tokens": total_s1_tokens / total_samples,
        "stage1_feature_norm": total_s1_norm / total_samples,
        "stage1_temporal_std": total_s1_std / total_samples,
        "stage2_temporal_tokens": total_s2_tokens / total_samples,
        "stage2_attention_entropy": total_s2_entropy / total_samples,
        "stage2_attention_peak": total_s2_peak / total_samples,
        "stage2_peak_to_event_mae_frames": total_s2_peak_mae / total_samples,
        "stage2_ed_es_feature_distance": total_s2_feat_dist / total_samples,
        "stage3_ed_index_ce": total_s3_ed_ce / total_samples,
        "stage3_es_index_ce": total_s3_es_ce / total_samples,
    }
    return metrics

def train_one_epoch(
    model,
    loader,
    optimizer,
    mse_loss,
    phase_regression_loss_fn,
    logger,
    epoch_idx,
    amp_enabled,
    scaler,
):
    """Run one training epoch and return loss + timing breakdown."""
    model.train()
    total_train_loss = 0.0
    total_samples = 0

    data_time = 0.0
    compute_time = 0.0
    loop_end = time.perf_counter()

    accumulation_steps = max(1, int(getattr(config, "GRADIENT_ACCUMULATION_STEPS", 1)))
    optimizer.zero_grad(set_to_none=True)

    num_batches = len(loader)
    phase_only = is_phase_only_epoch(epoch_idx)
    attn_align_weight = max(0.0, float(getattr(config, "PHASE_ATTN_ALIGN_WEIGHT", 0.0)))
    attn_index_weight = max(0.0, float(getattr(config, "PHASE_ATTN_INDEX_WEIGHT", 0.0)))
    attn_order_weight = max(0.0, float(getattr(config, "PHASE_ATTN_ORDER_WEIGHT", 0.0)))
    attn_entropy_weight = max(0.0, float(getattr(config, "PHASE_ATTN_ENTROPY_WEIGHT", 0.0)))
    phase_pair_index_weight = max(0.0, float(getattr(config, "PHASE_PAIR_INDEX_WEIGHT", 0.0)))
    phase_pair_order_weight = max(0.0, float(getattr(config, "PHASE_PAIR_ORDER_WEIGHT", 0.0)))

    for batch_idx, (videos, efs, ed_idx, es_idx) in enumerate(loader):
        data_time += time.perf_counter() - loop_end

        videos, efs, ed_idx, es_idx = move_batch_to_device(videos, efs, ed_idx, es_idx)
        batch_size = videos.size(0)
        total_samples += batch_size

        num_frames = videos.size(2)  # videos shape: (batch, channels, frames, height, width)
        phase_labels = build_frame_phase_targets(
            ed_idx,
            es_idx,
            num_frames,
            radius=max(0, int(getattr(config, "PHASE_FRAME_RADIUS", 1))),
        )

        compute_start = time.perf_counter()

        with autocast_context(amp_enabled):
            ef_pred, attention, phase_pred = model(videos)

            if phase_only:
                ef_loss = torch.zeros((), device=videos.device)
            else:
                ef_loss = mse_loss(ef_pred, efs)

            phase_loss = compute_continuous_phase_loss(
                phase_pred=phase_pred,
                phase_labels=phase_labels,
                phase_regression_loss_fn=phase_regression_loss_fn,
            )

            phase_vec = phase_vector_from_output(phase_pred)
            if phase_vec is not None:
                cyclic_targets = build_cyclic_phase_targets(ed_idx, es_idx, num_frames)
                phase_loss = phase_loss + F.smooth_l1_loss(phase_vec, cyclic_targets)

            if attn_align_weight > 0.0:
                phase_loss = phase_loss + attn_align_weight * compute_attention_alignment_loss(attention, ed_idx, es_idx)
            if attn_index_weight > 0.0 or attn_order_weight > 0.0:
                attn_index_loss, attn_order_loss = compute_attention_index_loss(attention, ed_idx, es_idx)
                phase_loss = phase_loss + attn_index_weight * attn_index_loss + attn_order_weight * attn_order_loss
            if attn_entropy_weight > 0.0:
                phase_loss = phase_loss + attn_entropy_weight * compute_attention_entropy_loss(attention)
            if phase_pair_index_weight > 0.0 or phase_pair_order_weight > 0.0:
                phase_logits = phase_logits_from_output(phase_pred)
                if phase_logits is not None:
                    pair_index_loss, pair_order_loss = compute_phase_pair_regularizers(phase_logits, ed_idx, es_idx)
                    phase_loss = phase_loss + phase_pair_index_weight * pair_index_loss + phase_pair_order_weight * pair_order_loss
            
            loss = ef_loss + config.PHASE_LOSS_WEIGHT * phase_loss
            loss_for_backward = loss / accumulation_steps

        if amp_enabled:
            scaler.scale(loss_for_backward).backward()
        else:
            loss_for_backward.backward()

        should_step = ((batch_idx + 1) % accumulation_steps == 0) or (batch_idx + 1 == num_batches)
        if should_step:
            max_grad_norm = float(getattr(config, "MAX_GRAD_NORM", 0.0))
            if max_grad_norm > 0.0:
                if amp_enabled:
                    scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

            if amp_enabled:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        compute_time += time.perf_counter() - compute_start
        total_train_loss += loss.item()

        if batch_idx == 0:
            # Extract ED/ES from continuous phase predictions
            pred_ed_idx, pred_es_idx = Stage3PhaseDetector.predict_indices(phase_pred)
            logger.info(
                "Epoch %d batch %d | Phase-only %s | EF loss %.4f | Phase loss %.6f | Pred ED/ES (%d/%d) | Attention shape %s",
                epoch_idx + 1,
                batch_idx,
                phase_only,
                ef_loss.item(),
                phase_loss.item(),
                pred_ed_idx[0].item(),
                pred_es_idx[0].item(),
                tuple(attention.shape) if attention is not None else "None",
            )

        loop_end = time.perf_counter()

    metrics = {
        "train_loss": total_train_loss / num_batches,
        "data_time": data_time,
        "compute_time": compute_time,
        "samples": total_samples,
    }
    return metrics


def save_checkpoint(model, optimizer, monitor_name, monitor_value, epoch, val_mae=None):
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "monitor_name": monitor_name,
            "monitor_value": monitor_value,
            "val_mae": val_mae,
            "epoch": epoch,
            "runtime_config": {
                "NUM_FRAMES": int(getattr(config, "NUM_FRAMES", 32)),
                "CLIP_PERIOD": int(getattr(config, "CLIP_PERIOD", 1)),
                "CLIP_EVAL_MODE": str(getattr(config, "CLIP_EVAL_MODE", "center")),
                "PHASE_ONLY_WARMUP_EPOCHS": int(getattr(config, "PHASE_ONLY_WARMUP_EPOCHS", 0)),
                "PHASE_ATTN_ENTROPY_WEIGHT": float(getattr(config, "PHASE_ATTN_ENTROPY_WEIGHT", 0.0)),
                "PHASE_PAIR_MIN_GAP": int(getattr(config, "PHASE_PAIR_MIN_GAP", 1)),
            },
            "args": {
                "num_frames": int(getattr(config, "NUM_FRAMES", 32)),
                "clip_period": int(getattr(config, "CLIP_PERIOD", 1)),
                "clip_eval_mode": str(getattr(config, "CLIP_EVAL_MODE", "center")),
                "phase_only_warmup_epochs": int(getattr(config, "PHASE_ONLY_WARMUP_EPOCHS", 0)),
                "phase_attn_entropy_weight": float(getattr(config, "PHASE_ATTN_ENTROPY_WEIGHT", 0.0)),
                "train_stage123": True,
                "phase_only": bool(getattr(config, "PHASE_ONLY", False)),
            },
        },
        config.CHECKPOINT_PATH,
    )


def maybe_warm_start_from_checkpoint(model, logger, enabled=True):
    """Optionally initialize model weights from existing checkpoint path."""
    if not enabled:
        return False

    ckpt_path = str(getattr(config, "CHECKPOINT_PATH", "")).strip()
    if not ckpt_path or not os.path.exists(ckpt_path):
        return False

    try:
        checkpoint = torch.load(ckpt_path, map_location=config.DEVICE)
        state_dict = checkpoint.get("model_state_dict", checkpoint)
        model_state = model.state_dict()
        filtered_state_dict = {
            key: value
            for key, value in state_dict.items()
            if key in model_state and tuple(value.shape) == tuple(model_state[key].shape)
        }
        incompatible = model.load_state_dict(filtered_state_dict, strict=False)
        missing = list(getattr(incompatible, "missing_keys", []))
        unexpected = list(getattr(incompatible, "unexpected_keys", []))

        if missing or unexpected:
            logger.warning(
                "Warm-start loaded with key mismatch | missing=%d unexpected=%d",
                len(missing),
                len(unexpected),
            )

        logger.info("Warm-started model from checkpoint: %s", ckpt_path)
        return True
    except Exception as exc:
        logger.warning("Warm-start skipped (failed to load checkpoint %s): %s", ckpt_path, exc)
        return False


def load_existing_monitor_baseline(logger, expected_monitor_name):
    """Read monitor value from existing checkpoint for no-regression protection."""
    ckpt_path = str(getattr(config, "CHECKPOINT_PATH", "")).strip()
    if not ckpt_path or not os.path.exists(ckpt_path):
        return None

    try:
        checkpoint = torch.load(ckpt_path, map_location=config.DEVICE)
    except Exception as exc:
        logger.warning("Checkpoint protection skipped (failed to read %s): %s", ckpt_path, exc)
        return None

    monitor_name = checkpoint.get("monitor_name")
    monitor_value = checkpoint.get("monitor_value", checkpoint.get("val_mae", None))

    if monitor_name != expected_monitor_name:
        logger.info(
            "Checkpoint protection: monitor mismatch (existing=%s expected=%s), ignoring baseline",
            monitor_name,
            expected_monitor_name,
        )
        return None

    try:
        monitor_value = float(monitor_value)
    except Exception:
        logger.info("Checkpoint protection: invalid existing monitor value, ignoring baseline")
        return None

    if not math.isfinite(monitor_value):
        logger.info("Checkpoint protection: non-finite existing monitor value, ignoring baseline")
        return None

    logger.info(
        "Checkpoint protection baseline loaded from %s | %s=%s",
        ckpt_path,
        monitor_name,
        monitor_value,
    )
    return monitor_value


def log_header(logger, amp_enabled):
    logger.info("=" * 80)
    logger.info("TRAINING SCRIPT STARTED")
    logger.info("=" * 80)
    logger.info("Using device: %s", config.DEVICE)
    logger.info("Data directory: %s", config.DATA_DIR)
    logger.info("Batch size: %d", config.BATCH_SIZE)
    logger.info("Number of frames: %d", config.NUM_FRAMES)
    logger.info("Max videos: %s", config.MAX_VIDEOS if config.MAX_VIDEOS else "All")
    logger.info("Learning rate: %s", config.LEARNING_RATE)
    logger.info("Epochs: %d", config.EPOCHS)
    logger.info("Workers: %d", config.NUM_WORKERS)
    logger.info("Pin memory: %s", getattr(config, "PIN_MEMORY", True))
    logger.info("Persistent workers: %s", getattr(config, "PERSISTENT_WORKERS", True))
    logger.info("Prefetch factor: %s", getattr(config, "PREFETCH_FACTOR", None))
    logger.info("AMP enabled: %s", amp_enabled)
    logger.info("Normalize input: %s", bool(getattr(config, "NORMALIZE_INPUT", True)))
    logger.info("Validate every: %d epoch(s)", int(getattr(config, "VALIDATE_EVERY", 1)))
    logger.info("Phase loss weight: %.3f", float(getattr(config, "PHASE_LOSS_WEIGHT", 0.5)))
    logger.info("Phase label smoothing: %.3f", float(getattr(config, "PHASE_LABEL_SMOOTHING", 0.0)))
    logger.info("Phase-only mode: %s", is_phase_only_mode())
    logger.info("Phase-only warmup epochs: %d", int(getattr(config, "PHASE_ONLY_WARMUP_EPOCHS", 0)))
    logger.info("Phase backbone freeze epochs: %d", int(getattr(config, "PHASE_BACKBONE_FREEZE_EPOCHS", 0)))
    logger.info("Backbone LR multiplier: %.3f", float(getattr(config, "BACKBONE_LR_MULT", 1.0)))
    logger.info("Phase soft sigma: %.3f", float(getattr(config, "PHASE_SOFT_SIGMA", 0.0)))
    logger.info("Phase soft radius: %d", int(getattr(config, "PHASE_SOFT_RADIUS", 0)))
    logger.info("Phase frame CE weight: %.3f", float(getattr(config, "PHASE_FRAME_CE_WEIGHT", 0.0)))
    logger.info("Phase frame radius: %d", int(getattr(config, "PHASE_FRAME_RADIUS", 1)))
    logger.info("Phase attn align weight: %.3f", float(getattr(config, "PHASE_ATTN_ALIGN_WEIGHT", 0.0)))
    logger.info("Phase attn align sigma: %.3f", float(getattr(config, "PHASE_ATTN_ALIGN_SIGMA", 0.0)))
    logger.info("Phase attn align radius: %d", int(getattr(config, "PHASE_ATTN_ALIGN_RADIUS", 0)))
    logger.info("Phase attn index weight: %.3f", float(getattr(config, "PHASE_ATTN_INDEX_WEIGHT", 0.0)))
    logger.info("Phase attn order weight: %.3f", float(getattr(config, "PHASE_ATTN_ORDER_WEIGHT", 0.0)))
    logger.info("Phase attn entropy weight: %.3f", float(getattr(config, "PHASE_ATTN_ENTROPY_WEIGHT", 0.0)))
    logger.info("Phase attn min gap: %d", int(getattr(config, "PHASE_ATTN_MIN_GAP", 1)))
    logger.info("Phase pair index weight: %.3f", float(getattr(config, "PHASE_PAIR_INDEX_WEIGHT", 0.0)))
    logger.info("Phase pair order weight: %.3f", float(getattr(config, "PHASE_PAIR_ORDER_WEIGHT", 0.0)))
    logger.info("Phase pair min gap: %d", int(getattr(config, "PHASE_PAIR_MIN_GAP", 1)))
    logger.info("Phase hard index weight: %.3f", float(getattr(config, "PHASE_HARD_INDEX_WEIGHT", 0.0)))
    logger.info("Phase unfreeze LR mult: %.3f", float(getattr(config, "PHASE_UNFREEZE_LR_MULT", 1.0)))
    logger.info("Weight decay: %s", getattr(config, "WEIGHT_DECAY", 0.0))
    logger.info("Max grad norm: %s", getattr(config, "MAX_GRAD_NORM", 0.0))
    logger.info("Clip period: %d", int(getattr(config, "CLIP_PERIOD", 1)))
    logger.info("Clip eval mode: %s", str(getattr(config, "CLIP_EVAL_MODE", "center")))


def main(argv=None):
    args = parse_args(argv)
    logger, _ = setup_logger()
    apply_runtime_overrides(args, logger)
    setup_performance_backends(logger)

    train_loader, val_loader, test_loader = build_dataloaders()
    model, optimizer, mse_loss, phase_regression_loss_fn, amp_enabled, scaler = build_model_stack(logger)

    phase_only = is_phase_only_mode()
    train_stage123_mode = bool(getattr(args, "train_stage123", False))
    freeze_epochs = max(0, int(getattr(config, "PHASE_BACKBONE_FREEZE_EPOCHS", 0)))

    if phase_only:
        best_monitor = -float("inf")
        monitor_name = "phase_score_joint_minus_mae"
    elif train_stage123_mode:
        best_monitor = -float("inf")
        monitor_name = "stage123_joint_score"
    else:
        best_monitor = float("inf")
        monitor_name = "ef_mae"

    warm_start_enabled = bool(getattr(args, "warm_start_checkpoint", True))
    protect_checkpoint_enabled = bool(getattr(args, "protect_best_checkpoint", True))

    maybe_warm_start_from_checkpoint(model, logger, enabled=warm_start_enabled)

    if bool(getattr(args, "train_stage123", False)):
        set_backbone_trainable(model, True)
        freeze_epochs = 0
        optimizer = build_optimizer(model, logger)
        logger.info("Stage1-3 mode active: Stage1/Stage2/Stage3 will be trained jointly")

    log_stage_trainability(model, logger)

    if protect_checkpoint_enabled:
        baseline_monitor = load_existing_monitor_baseline(logger, expected_monitor_name=monitor_name)
        if baseline_monitor is not None:
            if phase_only or train_stage123_mode:
                best_monitor = max(best_monitor, baseline_monitor)
            else:
                best_monitor = min(best_monitor, baseline_monitor)

    log_header(logger, amp_enabled)
    logger.info("Train Stage1+Stage2+Stage3 mode: %s", bool(getattr(args, "train_stage123", False)))

    epochs_without_improvement = 0
    validate_every = max(1, int(getattr(config, "VALIDATE_EVERY", 1)))

    for epoch in range(config.EPOCHS):
        if phase_only and freeze_epochs > 0 and epoch == freeze_epochs:
            if set_backbone_trainable(model, True):
                logger.info("Unfreezing Stage1 backbone at epoch %d", epoch + 1)
                unfreeze_lr_mult = float(getattr(config, "PHASE_UNFREEZE_LR_MULT", 1.0))
                if unfreeze_lr_mult > 0 and unfreeze_lr_mult != 1.0:
                    config.LEARNING_RATE = float(config.LEARNING_RATE) * unfreeze_lr_mult
                    logger.info("Applying LR multiplier at unfreeze: x%.3f -> LR=%s", unfreeze_lr_mult, config.LEARNING_RATE)
                optimizer = build_optimizer(model, logger)

        epoch_start = time.perf_counter()

        train_metrics = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            mse_loss=mse_loss,
            phase_regression_loss_fn=phase_regression_loss_fn,
            logger=logger,
            epoch_idx=epoch,
            amp_enabled=amp_enabled,
            scaler=scaler,
        )

        val_metrics = None
        val_duration = 0.0
        should_validate = ((epoch + 1) % validate_every == 0) or (epoch + 1 == config.EPOCHS)
        if should_validate:
            val_start = time.perf_counter()
            val_metrics = evaluate(model, val_loader, amp_enabled=amp_enabled)
            val_duration = time.perf_counter() - val_start

            if phase_only:
                logger.info(
                    "Epoch [%d/%d] | Train Loss: %.4f | Val ED Acc: %.2f%% | Val ES Acc: %.2f%% | Val Joint Acc: %.2f%% | Val ED MAE(fr): %.3f | Val ES MAE(fr): %.3f",
                    epoch + 1,
                    config.EPOCHS,
                    train_metrics["train_loss"],
                    val_metrics["ed_acc"] * 100,
                    val_metrics["es_acc"] * 100,
                    val_metrics["joint_acc"] * 100,
                    val_metrics["ed_mae_frames"],
                    val_metrics["es_mae_frames"],
                )

                phase_score = val_metrics["joint_acc"] - 0.01 * (
                    val_metrics["ed_mae_frames"] + val_metrics["es_mae_frames"]
                )
                current_monitor = phase_score
                improved = current_monitor > best_monitor
            elif train_stage123_mode:
                logger.info(
                    "Epoch [%d/%d] | Train Loss: %.4f | Val EF MAE: %.2f%% | Val EF RMSE: %.2f%% | Val ED Acc: %.2f%% | Val ES Acc: %.2f%% | Val Joint Acc: %.2f%% | Val ED/ES MAE(fr): %.3f / %.3f",
                    epoch + 1,
                    config.EPOCHS,
                    train_metrics["train_loss"],
                    val_metrics["ef_mae"] * 100,
                    val_metrics["ef_rmse"] * 100,
                    val_metrics["ed_acc"] * 100,
                    val_metrics["es_acc"] * 100,
                    val_metrics["joint_acc"] * 100,
                    val_metrics["ed_mae_frames"],
                    val_metrics["es_mae_frames"],
                )

                phase_score = val_metrics["joint_acc"] - 0.01 * (
                    val_metrics["ed_mae_frames"] + val_metrics["es_mae_frames"]
                )
                ef_score = 1.0 - val_metrics["ef_mae"]
                current_monitor = 0.65 * phase_score + 0.35 * ef_score
                improved = current_monitor > best_monitor
            else:
                logger.info(
                    "Epoch [%d/%d] | Train Loss: %.4f | Val EF MAE: %.2f%% | Val EF RMSE: %.2f%% | Val ED Acc: %.2f%% | Val ES Acc: %.2f%% | Val Joint Acc: %.2f%%",
                    epoch + 1,
                    config.EPOCHS,
                    train_metrics["train_loss"],
                    val_metrics["ef_mae"] * 100,
                    val_metrics["ef_rmse"] * 100,
                    val_metrics["ed_acc"] * 100,
                    val_metrics["es_acc"] * 100,
                    val_metrics["joint_acc"] * 100,
                )

                current_monitor = val_metrics["ef_mae"]
                improved = current_monitor < best_monitor

            logger.info(
                "Stage diagnostics | S1(T'~%.1f) feat-norm: %.3f temp-std: %.3f | S2(T~%.1f) attn-entropy: %.3f peak-w: %.3f peak->ED/ES MAE(fr): %.3f ED-ES feat-dist: %.3f | S3 index CE (ED/ES): %.3f / %.3f | eval clips/video: %.1f",
                val_metrics["stage1_temporal_tokens"],
                val_metrics["stage1_feature_norm"],
                val_metrics["stage1_temporal_std"],
                val_metrics["stage2_temporal_tokens"],
                val_metrics["stage2_attention_entropy"],
                val_metrics["stage2_attention_peak"],
                val_metrics["stage2_peak_to_event_mae_frames"],
                val_metrics["stage2_ed_es_feature_distance"],
                val_metrics["stage3_ed_index_ce"],
                val_metrics["stage3_es_index_ce"],
                val_metrics.get("eval_clips_per_video", 1.0),
            )

            if improved:
                best_monitor = current_monitor
                epochs_without_improvement = 0
                save_checkpoint(
                    model,
                    optimizer,
                    monitor_name=monitor_name,
                    monitor_value=current_monitor,
                    epoch=epoch,
                    val_mae=val_metrics["ef_mae"] if not phase_only else None,
                )
                logger.info("Saving best model to %s", config.CHECKPOINT_PATH)
            else:
                epochs_without_improvement += 1
                logger.info("No improvement (%d/%d)", epochs_without_improvement, config.PATIENCE)

            if epochs_without_improvement >= config.PATIENCE:
                logger.info("Early stopping triggered")
                break
        else:
            logger.info(
                "Epoch [%d/%d] | Train Loss: %.4f | Validation skipped",
                epoch + 1,
                config.EPOCHS,
                train_metrics["train_loss"],
            )

        epoch_duration = time.perf_counter() - epoch_start
        train_duration = max(1e-9, train_metrics["data_time"] + train_metrics["compute_time"])
        samples_per_sec = train_metrics["samples"] / train_duration

        logger.info(
            "Timing | epoch: %.2fs | train(data): %.2fs | train(compute): %.2fs | val: %.2fs | throughput: %.2f samples/s",
            epoch_duration,
            train_metrics["data_time"],
            train_metrics["compute_time"],
            val_duration,
            samples_per_sec,
        )

    logger.info("Loading best model for final testing")
    checkpoint = torch.load(config.CHECKPOINT_PATH, map_location=config.DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"], strict=False)

    monitor_name_ckpt = checkpoint.get("monitor_name", "val_mae")
    monitor_value_ckpt = checkpoint.get("monitor_value", checkpoint.get("val_mae", float("nan")))
    logger.info("Best checkpoint monitor: %s = %s", monitor_name_ckpt, monitor_value_ckpt)

    test_start = time.perf_counter()
    test_metrics = evaluate(model, test_loader, amp_enabled=amp_enabled)
    test_duration = time.perf_counter() - test_start

    logger.info("=" * 80)
    logger.info("FINAL TEST RESULTS")
    if not phase_only:
        logger.info("Test EF MAE: %.2f%%", test_metrics["ef_mae"] * 100)
        logger.info("Test EF RMSE: %.2f%%", test_metrics["ef_rmse"] * 100)
    else:
        logger.info("Test EF metrics: N/A (phase-only mode)")
    logger.info("Test Phase loss: %.6f", test_metrics["phase_mse"])
    logger.info(
        "Test Phase Acc: ED %.2f%% | ES %.2f%% | Joint %.2f%%",
        test_metrics["ed_acc"] * 100,
        test_metrics["es_acc"] * 100,
        test_metrics["joint_acc"] * 100,
    )
    logger.info(
        "Test ED/ES MAE(fr): %.3f / %.3f",
        test_metrics["ed_mae_frames"],
        test_metrics["es_mae_frames"],
    )
    logger.info("Test duration: %.2fs", test_duration)
    logger.info("=" * 80)


if __name__ == "__main__":
    main()


