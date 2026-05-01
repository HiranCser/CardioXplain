import argparse
from datetime import datetime
import logging
import math
import os
import sys
import time

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
    parser.add_argument("--max-videos", type=int, default=None, help="Override config.MAX_VIDEOS")
    parser.add_argument("--epochs", type=int, default=None, help="Override config.EPOCHS")
    parser.add_argument("--learning-rate", "--lr", dest="learning_rate", type=float, default=None, help="Override config.LEARNING_RATE")
    parser.add_argument("--batch-size", type=int, default=None, help="Override config.BATCH_SIZE")
    parser.add_argument("--num-frames", type=int, default=None, help="Override config.NUM_FRAMES")
    parser.add_argument("--dataset-period", type=int, default=None, help="Override config.DATASET_PERIOD")
    parser.add_argument("--dataset-max-length", type=int, default=None, help="Override config.DATASET_MAX_LENGTH")
    parser.add_argument("--eval-clips", type=int, default=None, help="Override config.EVAL_CLIPS")
    parser.add_argument("--train-pad", type=int, default=None, help="Override config.TRAIN_PAD")
    parser.add_argument("--train-noise", type=float, default=None, help="Override config.TRAIN_NOISE")
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
    parser.add_argument("--phase-attn-min-gap", type=int, default=None, help="Override config.PHASE_ATTN_MIN_GAP")
    parser.add_argument("--phase-pair-index-weight", type=float, default=None, help="Override config.PHASE_PAIR_INDEX_WEIGHT")
    parser.add_argument("--phase-pair-order-weight", type=float, default=None, help="Override config.PHASE_PAIR_ORDER_WEIGHT")
    parser.add_argument("--phase-pair-min-gap", type=int, default=None, help="Override config.PHASE_PAIR_MIN_GAP")
    parser.add_argument("--phase-unfreeze-lr-mult", type=float, default=None, help="Override config.PHASE_UNFREEZE_LR_MULT")
    parser.add_argument("--weight-decay", type=float, default=None, help="Override config.WEIGHT_DECAY")
    parser.add_argument("--max-grad-norm", type=float, default=None, help="Override config.MAX_GRAD_NORM")
    parser.add_argument("--phase-temporal-window-mode", type=str, choices=["full", "tracing"], default=None, help="Override config.PHASE_TEMPORAL_WINDOW_MODE")
    parser.add_argument("--phase-temporal-window-margin-mult", type=float, default=None, help="Override config.PHASE_TEMPORAL_WINDOW_MARGIN_MULT")
    parser.add_argument("--phase-temporal-window-jitter-mult", type=float, default=None, help="Override config.PHASE_TEMPORAL_WINDOW_JITTER_MULT")
    parser.add_argument("--warm-start-checkpoint", action=argparse.BooleanOptionalAction, default=True, help="Warm-start model weights from existing checkpoint path if available")
    parser.add_argument("--protect-best-checkpoint", action=argparse.BooleanOptionalAction, default=True, help="Do not overwrite checkpoint unless monitor improves over existing checkpoint")
    parser.add_argument("--train-stage123", action=argparse.BooleanOptionalAction, default=False, help="Train Stage1+Stage2+Stage3 end-to-end with joint EF+phase supervision")
    return parser.parse_args(argv)


def apply_runtime_overrides(args, logger):
    overrides = {}

    if args.smoke:
        overrides.update(SMOKE_DEFAULTS)

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
    if args.dataset_period is not None:
        overrides["DATASET_PERIOD"] = args.dataset_period
    if args.dataset_max_length is not None:
        overrides["DATASET_MAX_LENGTH"] = args.dataset_max_length
    if args.eval_clips is not None:
        overrides["EVAL_CLIPS"] = args.eval_clips
    if args.train_pad is not None:
        overrides["TRAIN_PAD"] = args.train_pad
    if args.train_noise is not None:
        overrides["TRAIN_NOISE"] = args.train_noise
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
    if args.phase_temporal_window_mode is not None:
        overrides["PHASE_TEMPORAL_WINDOW_MODE"] = args.phase_temporal_window_mode
    if args.phase_temporal_window_margin_mult is not None:
        overrides["PHASE_TEMPORAL_WINDOW_MARGIN_MULT"] = args.phase_temporal_window_margin_mult
    if args.phase_temporal_window_jitter_mult is not None:
        overrides["PHASE_TEMPORAL_WINDOW_JITTER_MULT"] = args.phase_temporal_window_jitter_mult

    if bool(getattr(args, "train_stage123", False)):
        logger.info("Enabled Stage1-3 end-to-end training profile (joint EF + phase)")
        # Stage1/2/3 should train jointly without disabling EF regression.
        # Respect explicit CLI values when present.
        if args.phase_only is None:
            overrides["PHASE_ONLY"] = False
        if args.phase_loss_weight is None:
            overrides["PHASE_LOSS_WEIGHT"] = 1.0
        if args.phase_temporal_window_mode is None:
            overrides["PHASE_TEMPORAL_WINDOW_MODE"] = "tracing"
        if args.phase_temporal_window_margin_mult is None:
            overrides["PHASE_TEMPORAL_WINDOW_MARGIN_MULT"] = 1.0
        if args.phase_temporal_window_jitter_mult is None:
            overrides["PHASE_TEMPORAL_WINDOW_JITTER_MULT"] = 0.03
        if args.phase_attn_align_weight is None:
            overrides["PHASE_ATTN_ALIGN_WEIGHT"] = 0.45
        if args.phase_attn_index_weight is None:
            overrides["PHASE_ATTN_INDEX_WEIGHT"] = 0.30
        if args.phase_attn_order_weight is None:
            overrides["PHASE_ATTN_ORDER_WEIGHT"] = 0.10
        if args.phase_attn_min_gap is None:
            overrides["PHASE_ATTN_MIN_GAP"] = 2
        if args.phase_pair_index_weight is None:
            overrides["PHASE_PAIR_INDEX_WEIGHT"] = 0.24
        if args.phase_pair_order_weight is None:
            overrides["PHASE_PAIR_ORDER_WEIGHT"] = 0.10
        if args.phase_pair_min_gap is None:
            overrides["PHASE_PAIR_MIN_GAP"] = 2
        overrides["PHASE_BACKBONE_FREEZE_EPOCHS"] = 0

    for key, value in overrides.items():
        setattr(config, key, value)
        logger.info("Runtime override: %s=%s", key, value)

    jitter_key = "PHASE_TEMPORAL_WINDOW_JITTER_MULT"
    if hasattr(config, jitter_key):
        jitter_value = float(getattr(config, jitter_key, 0.0))
        if jitter_value < 0.0:
            setattr(config, jitter_key, 0.0)
            logger.warning("Clamped %s from %.3f to 0.000", jitter_key, jitter_value)
        elif jitter_value > 0.10:
                setattr(config, jitter_key, 0.10)
                logger.warning("Clamped %s from %.3f to 0.100 to avoid over-augmentation", jitter_key, jitter_value)

    if hasattr(config, "DATASET_PERIOD"):
        period_value = int(getattr(config, "DATASET_PERIOD", 1))
        if period_value < 1:
            setattr(config, "DATASET_PERIOD", 1)
            logger.warning("Clamped DATASET_PERIOD from %d to 1", period_value)

    if hasattr(config, "EVAL_CLIPS"):
        eval_clips_value = int(getattr(config, "EVAL_CLIPS", 1))
        if eval_clips_value < 1:
            setattr(config, "EVAL_CLIPS", 1)
            logger.warning("Clamped EVAL_CLIPS from %d to 1", eval_clips_value)

    if hasattr(config, "TRAIN_PAD"):
        train_pad_value = getattr(config, "TRAIN_PAD", None)
        if train_pad_value is not None and int(train_pad_value) < 0:
            setattr(config, "TRAIN_PAD", 0)
            logger.warning("Clamped TRAIN_PAD from %s to 0", train_pad_value)

    if hasattr(config, "TRAIN_NOISE"):
        train_noise_value = getattr(config, "TRAIN_NOISE", None)
        if train_noise_value is not None:
            train_noise_value = float(train_noise_value)
            clamped_noise = min(1.0, max(0.0, train_noise_value))
            if clamped_noise != train_noise_value:
                setattr(config, "TRAIN_NOISE", clamped_noise)
                logger.warning("Clamped TRAIN_NOISE from %.3f to %.3f", train_noise_value, clamped_noise)

    attn_align_key = "PHASE_ATTN_ALIGN_WEIGHT"
    if hasattr(config, attn_align_key):
        attn_align_value = float(getattr(config, attn_align_key, 0.0))
        if attn_align_value < 0.0:
            setattr(config, attn_align_key, 0.0)
            logger.warning("Clamped %s from %.3f to 0.000", attn_align_key, attn_align_value)
        elif attn_align_value > 2.0:
            setattr(config, attn_align_key, 2.0)
            logger.warning("Clamped %s from %.3f to 2.000", attn_align_key, attn_align_value)

    for loss_key in ("PHASE_ATTN_INDEX_WEIGHT", "PHASE_ATTN_ORDER_WEIGHT", "PHASE_PAIR_INDEX_WEIGHT", "PHASE_PAIR_ORDER_WEIGHT"):
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
    temporal_window_mode = str(getattr(config, "PHASE_TEMPORAL_WINDOW_MODE", "full"))
    temporal_window_margin_mult = float(getattr(config, "PHASE_TEMPORAL_WINDOW_MARGIN_MULT", 1.5))
    temporal_window_jitter_mult = float(getattr(config, "PHASE_TEMPORAL_WINDOW_JITTER_MULT", 0.0))
    dataset_period = int(getattr(config, "DATASET_PERIOD", 1))
    dataset_max_length = getattr(config, "DATASET_MAX_LENGTH", None)
    eval_clips = int(getattr(config, "EVAL_CLIPS", 1))
    train_pad = getattr(config, "TRAIN_PAD", None)
    train_noise = getattr(config, "TRAIN_NOISE", None)

    common_kwargs = {
        "data_dir": config.DATA_DIR,
        "num_frames": config.NUM_FRAMES,
        "max_videos": config.MAX_VIDEOS,
        "normalize_input": bool(getattr(config, "NORMALIZE_INPUT", True)),
        "period": dataset_period,
        "max_length": dataset_max_length,
        "temporal_window_mode": temporal_window_mode,
        "temporal_window_margin_mult": temporal_window_margin_mult,
        "temporal_window_jitter_mult": temporal_window_jitter_mult,
    }

    train_dataset = EchoDataset(
        split="TRAIN",
        clips=1,
        pad=train_pad,
        noise=train_noise,
        **common_kwargs,
    )
    val_dataset = EchoDataset(
        split="VAL",
        clips=eval_clips,
        pad=None,
        noise=None,
        **common_kwargs,
    )
    test_dataset = EchoDataset(
        split="TEST",
        clips=eval_clips,
        pad=None,
        noise=None,
        **common_kwargs,
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
    phase_index_loss = nn.CrossEntropyLoss(
        label_smoothing=float(getattr(config, "PHASE_LABEL_SMOOTHING", 0.0))
    )

    amp_enabled = bool(getattr(config, "USE_MIXED_PRECISION", False)) and is_cuda_runtime()
    scaler = make_grad_scaler(amp_enabled)
    return model, optimizer, mse_loss, phase_index_loss, amp_enabled, scaler


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


def compute_phase_pair_regularizers(phase_logits, ed_idx, es_idx):
    """Regularize Stage3 ED/ES expectations directly in the same score space used at inference."""
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


def compute_phase_index_loss(phase_logits, ed_idx, es_idx, phase_index_loss_fn):
    """
    Train phase detection as two temporal index tasks:
    - ED index from ED-vs-background score curve
    - ES index from ES-vs-background score curve

    Optional stabilizer: frame-wise CE over {background, ED, ES} neighborhoods.
    """
    # Match training targets to the exact score definition used at inference time
    # in Stage3PhaseDetector.predict_indices(...).
    ed_logits = phase_logits[:, :, 1] - phase_logits[:, :, 0]  # (B, T)
    es_logits = phase_logits[:, :, 2] - phase_logits[:, :, 0]  # (B, T)

    sigma = float(getattr(config, "PHASE_SOFT_SIGMA", 0.0))
    radius = int(getattr(config, "PHASE_SOFT_RADIUS", 0))
    hard_weight = float(getattr(config, "PHASE_HARD_INDEX_WEIGHT", 0.0))
    hard_weight = min(1.0, max(0.0, hard_weight))

    if sigma > 0.0:
        num_frames = ed_logits.shape[1]
        ed_target = build_soft_temporal_targets(ed_idx, num_frames, ed_logits.device, sigma, radius)
        es_target = build_soft_temporal_targets(es_idx, num_frames, es_logits.device, sigma, radius)

        ed_soft_loss = F.kl_div(F.log_softmax(ed_logits, dim=1), ed_target, reduction="batchmean")
        es_soft_loss = F.kl_div(F.log_softmax(es_logits, dim=1), es_target, reduction="batchmean")
        ed_hard_loss = phase_index_loss_fn(ed_logits, ed_idx)
        es_hard_loss = phase_index_loss_fn(es_logits, es_idx)

        ed_loss = (1.0 - hard_weight) * ed_soft_loss + hard_weight * ed_hard_loss
        es_loss = (1.0 - hard_weight) * es_soft_loss + hard_weight * es_hard_loss
    else:
        ed_loss = phase_index_loss_fn(ed_logits, ed_idx)
        es_loss = phase_index_loss_fn(es_logits, es_idx)

    index_loss = 0.5 * (ed_loss + es_loss)

    frame_weight = float(getattr(config, "PHASE_FRAME_CE_WEIGHT", 0.0))
    frame_weight = min(1.0, max(0.0, frame_weight))

    if frame_weight > 0.0:
        frame_radius = max(0, int(getattr(config, "PHASE_FRAME_RADIUS", 1)))
        frame_targets = build_frame_phase_targets(
            ed_idx=ed_idx,
            es_idx=es_idx,
            num_frames=phase_logits.shape[1],
            radius=frame_radius,
        )
        frame_loss = F.cross_entropy(
            phase_logits.reshape(-1, phase_logits.shape[-1]),
            frame_targets.reshape(-1),
        )
        phase_loss = (1.0 - frame_weight) * index_loss + frame_weight * frame_loss
    else:
        frame_loss = torch.zeros_like(index_loss)
        phase_loss = index_loss

    return phase_loss, ed_loss, es_loss, frame_loss


def move_batch_to_device(videos, efs, ed_idx, es_idx):
    non_blocking = bool(getattr(config, "NON_BLOCKING_TRANSFER", True)) and is_cuda_runtime()
    videos = videos.to(config.DEVICE, non_blocking=non_blocking)
    efs = efs.to(config.DEVICE, non_blocking=non_blocking)
    ed_idx = ed_idx.to(config.DEVICE, non_blocking=non_blocking)
    es_idx = es_idx.to(config.DEVICE, non_blocking=non_blocking)
    return videos, efs, ed_idx, es_idx


def flatten_eval_clips(videos, ed_idx, es_idx):
    """Flatten multi-clip batches into clip-wise batches for model evaluation."""
    if videos.ndim != 6:
        return videos, ed_idx, es_idx, 1

    batch_size, num_clips = videos.shape[:2]
    videos = videos.reshape(batch_size * num_clips, *videos.shape[2:])
    ed_idx = ed_idx.reshape(batch_size * num_clips)
    es_idx = es_idx.reshape(batch_size * num_clips)
    return videos, ed_idx, es_idx, num_clips


def evaluate(model, loader, amp_enabled):
    """Evaluate EF regression, phase localization, and stage-wise diagnostics."""
    model.eval()

    total_ef_samples = 0
    total_phase_samples = 0
    total_mae = 0.0
    total_mse = 0.0
    total_ed_correct = 0
    total_es_correct = 0
    total_joint_correct = 0
    total_ed_abs_err = 0.0
    total_es_abs_err = 0.0

    # Stage 1 diagnostics
    stage1_feat_norm_sum = 0.0
    stage1_temp_std_sum = 0.0
    stage1_tokens_sum = 0.0
    stage1_count = 0

    # Stage 2 diagnostics
    stage2_attn_entropy_sum = 0.0
    stage2_attn_peak_sum = 0.0
    stage2_peak_to_event_sum = 0.0
    stage2_ed_es_feat_dist_sum = 0.0
    stage2_tokens_sum = 0.0
    stage2_attn_count = 0
    stage2_feat_count = 0

    # Stage 3 diagnostics
    stage3_ed_ce_sum = 0.0
    stage3_es_ce_sum = 0.0

    phase_only = is_phase_only_mode()

    with torch.no_grad():
        for videos, efs, ed_idx, es_idx in loader:
            videos, efs, ed_idx, es_idx = move_batch_to_device(videos, efs, ed_idx, es_idx)
            sample_batch_size = videos.shape[0]
            videos, ed_idx, es_idx, num_clips = flatten_eval_clips(videos, ed_idx, es_idx)

            with autocast_context(amp_enabled):
                model_out = model(videos, return_stage_outputs=True)

            if isinstance(model_out, tuple) and len(model_out) == 4:
                ef_pred, attention, phase_logits, stage_outputs = model_out
            else:
                ef_pred, attention, phase_logits = model_out
                stage_outputs = {}

            batch_size = videos.size(0)

            if not phase_only:
                ef_pred_eval = ef_pred.reshape(sample_batch_size, num_clips).mean(dim=1) if num_clips > 1 else ef_pred.reshape(sample_batch_size)
                mae = torch.abs(ef_pred_eval - efs).mean()
                mse = torch.mean((ef_pred_eval - efs) ** 2)
                total_mae += mae.item() * sample_batch_size
                total_mse += mse.item() * sample_batch_size
                total_ef_samples += sample_batch_size

            pred_ed_idx, pred_es_idx = Stage3PhaseDetector.predict_indices(phase_logits)

            ed_abs = torch.abs(pred_ed_idx - ed_idx)
            es_abs = torch.abs(pred_es_idx - es_idx)

            total_ed_abs_err += ed_abs.sum().item()
            total_es_abs_err += es_abs.sum().item()

            ed_ok = ed_abs <= config.TOLERANCE
            es_ok = es_abs <= config.TOLERANCE

            total_ed_correct += ed_ok.sum().item()
            total_es_correct += es_ok.sum().item()
            total_joint_correct += (ed_ok & es_ok).sum().item()

            # Stage 1: feature magnitude and temporal variation
            stage1_features = stage_outputs.get("stage1_features")
            if stage1_features is not None:
                s1 = stage1_features.float()
                feat_norm = torch.linalg.vector_norm(s1.flatten(start_dim=1), dim=1)
                temp_std = s1.std(dim=2, unbiased=False).mean(dim=1)
                stage1_feat_norm_sum += feat_norm.sum().item()
                stage1_temp_std_sum += temp_std.sum().item()
                stage1_tokens_sum += float(s1.shape[2]) * batch_size
                stage1_count += batch_size

            # Stage 2: temporal attention diagnostics
            stage2_attention = stage_outputs.get("stage2_attention", attention)
            _attn_heads, attn = _attention_heads_and_summary(stage2_attention)
            if attn is not None and attn.ndim == 2 and attn.shape[1] > 0:
                peak_vals, peak_idx = attn.max(dim=1)
                stage2_attn_peak_sum += peak_vals.sum().item()

                peak_to_event = torch.minimum(
                    torch.abs(peak_idx - ed_idx),
                    torch.abs(peak_idx - es_idx),
                )
                stage2_peak_to_event_sum += peak_to_event.sum().item()

                if attn.shape[1] > 1:
                    attn_safe = torch.clamp(attn, min=1e-8)
                    attn_entropy = -(attn_safe * torch.log(attn_safe)).sum(dim=1) / math.log(attn.shape[1])
                else:
                    attn_entropy = torch.zeros(batch_size, device=attn.device)
                stage2_attn_entropy_sum += attn_entropy.sum().item()
                stage2_tokens_sum += float(attn.shape[1]) * batch_size
                stage2_attn_count += batch_size

            stage2_temporal_features = stage_outputs.get("stage2_temporal_features")
            if stage2_temporal_features is not None:
                tf = stage2_temporal_features.float()  # (B, T, F)
                if tf.ndim == 3 and tf.shape[1] > 0:
                    t = tf.shape[1]
                    fdim = tf.shape[2]
                    ed_safe = ed_idx.clamp(0, t - 1)
                    es_safe = es_idx.clamp(0, t - 1)

                    ed_gather = ed_safe.view(batch_size, 1, 1).expand(-1, 1, fdim)
                    es_gather = es_safe.view(batch_size, 1, 1).expand(-1, 1, fdim)

                    ed_feat = tf.gather(1, ed_gather).squeeze(1)
                    es_feat = tf.gather(1, es_gather).squeeze(1)
                    ed_es_dist = torch.linalg.vector_norm(ed_feat - es_feat, dim=1)

                    stage2_ed_es_feat_dist_sum += ed_es_dist.sum().item()
                    stage2_feat_count += batch_size

            # Stage 3: ED/ES index CE (diagnostic only), using the same
            # relative score curves that inference uses for pair decoding.
            ed_ce = F.cross_entropy(phase_logits[:, :, 1] - phase_logits[:, :, 0], ed_idx, reduction="sum")
            es_ce = F.cross_entropy(phase_logits[:, :, 2] - phase_logits[:, :, 0], es_idx, reduction="sum")
            stage3_ed_ce_sum += ed_ce.item()
            stage3_es_ce_sum += es_ce.item()

            total_phase_samples += batch_size

    if total_phase_samples == 0:
        raise RuntimeError("No samples found during evaluation")

    metrics = {
        "ef_mae": (total_mae / total_ef_samples) if (not phase_only and total_ef_samples > 0) else float("nan"),
        "ef_rmse": ((total_mse / total_ef_samples) ** 0.5) if (not phase_only and total_ef_samples > 0) else float("nan"),
        "ed_acc": total_ed_correct / total_phase_samples,
        "es_acc": total_es_correct / total_phase_samples,
        "joint_acc": total_joint_correct / total_phase_samples,
        "ed_mae_frames": total_ed_abs_err / total_phase_samples,
        "es_mae_frames": total_es_abs_err / total_phase_samples,
        "stage1_feature_norm": (stage1_feat_norm_sum / stage1_count) if stage1_count > 0 else float("nan"),
        "stage1_temporal_std": (stage1_temp_std_sum / stage1_count) if stage1_count > 0 else float("nan"),
        "stage1_temporal_tokens": (stage1_tokens_sum / stage1_count) if stage1_count > 0 else float("nan"),
        "stage2_attention_entropy": (stage2_attn_entropy_sum / stage2_attn_count) if stage2_attn_count > 0 else float("nan"),
        "stage2_attention_peak": (stage2_attn_peak_sum / stage2_attn_count) if stage2_attn_count > 0 else float("nan"),
        "stage2_peak_to_event_mae_frames": (stage2_peak_to_event_sum / stage2_attn_count) if stage2_attn_count > 0 else float("nan"),
        "stage2_temporal_tokens": (stage2_tokens_sum / stage2_attn_count) if stage2_attn_count > 0 else float("nan"),
        "stage2_ed_es_feature_distance": (stage2_ed_es_feat_dist_sum / stage2_feat_count) if stage2_feat_count > 0 else float("nan"),
        "stage3_ed_index_ce": stage3_ed_ce_sum / total_phase_samples,
        "stage3_es_index_ce": stage3_es_ce_sum / total_phase_samples,
    }
    return metrics

def train_one_epoch(
    model,
    loader,
    optimizer,
    mse_loss,
    phase_index_loss_fn,
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
    phase_only = is_phase_only_mode()
    attn_align_weight = max(0.0, float(getattr(config, "PHASE_ATTN_ALIGN_WEIGHT", 0.0)))
    attn_index_weight = max(0.0, float(getattr(config, "PHASE_ATTN_INDEX_WEIGHT", 0.0)))
    attn_order_weight = max(0.0, float(getattr(config, "PHASE_ATTN_ORDER_WEIGHT", 0.0)))
    phase_pair_index_weight = max(0.0, float(getattr(config, "PHASE_PAIR_INDEX_WEIGHT", 0.0)))
    phase_pair_order_weight = max(0.0, float(getattr(config, "PHASE_PAIR_ORDER_WEIGHT", 0.0)))

    for batch_idx, (videos, efs, ed_idx, es_idx) in enumerate(loader):
        data_time += time.perf_counter() - loop_end

        videos, efs, ed_idx, es_idx = move_batch_to_device(videos, efs, ed_idx, es_idx)
        batch_size = videos.size(0)
        total_samples += batch_size

        compute_start = time.perf_counter()

        with autocast_context(amp_enabled):
            ef_pred, attention, phase_logits = model(videos)

            if phase_only:
                ef_loss = torch.zeros((), device=videos.device)
            else:
                ef_loss = mse_loss(ef_pred, efs)

            phase_loss, ed_phase_loss, es_phase_loss, frame_phase_loss = compute_phase_index_loss(
                phase_logits=phase_logits,
                ed_idx=ed_idx,
                es_idx=es_idx,
                phase_index_loss_fn=phase_index_loss_fn,
            )
            phase_pair_index_loss, phase_pair_order_loss = compute_phase_pair_regularizers(phase_logits, ed_idx, es_idx)
            attn_align_loss = compute_attention_alignment_loss(attention, ed_idx, es_idx)
            attn_index_loss, attn_order_loss = compute_attention_index_loss(attention, ed_idx, es_idx)

            loss = (
                ef_loss
                + config.PHASE_LOSS_WEIGHT * phase_loss
                + phase_pair_index_weight * phase_pair_index_loss
                + phase_pair_order_weight * phase_pair_order_loss
                + attn_align_weight * attn_align_loss
                + attn_index_weight * attn_index_loss
                + attn_order_weight * attn_order_loss
            )
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
            pred_ed_idx, pred_es_idx = Stage3PhaseDetector.predict_indices(phase_logits)
            logger.info(
                "Epoch %d batch %d | EF loss %.4f | Phase loss %.4f (ED %.4f / ES %.4f / FrameCE %.4f) | PhasePair %.4f (w=%.3f) | PhaseOrder %.4f (w=%.3f) | AttnAlign %.4f (w=%.3f) | AttnIndex %.4f (w=%.3f) | AttnOrder %.4f (w=%.3f) | GT ED/ES (%d/%d) | Pred ED/ES (%d/%d) | Attention shape %s",
                epoch_idx + 1,
                batch_idx,
                ef_loss.item(),
                phase_loss.item(),
                ed_phase_loss.item(),
                es_phase_loss.item(),
                frame_phase_loss.item(),
                phase_pair_index_loss.item(),
                phase_pair_index_weight,
                phase_pair_order_loss.item(),
                phase_pair_order_weight,
                attn_align_loss.item(),
                attn_align_weight,
                attn_index_loss.item(),
                attn_index_weight,
                attn_order_loss.item(),
                attn_order_weight,
                ed_idx[0].item(),
                es_idx[0].item(),
                pred_ed_idx[0].item(),
                pred_es_idx[0].item(),
                tuple(attention.shape),
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
                "DATASET_PERIOD": int(getattr(config, "DATASET_PERIOD", 1)),
                "DATASET_MAX_LENGTH": getattr(config, "DATASET_MAX_LENGTH", None),
                "EVAL_CLIPS": int(getattr(config, "EVAL_CLIPS", 1)),
                "TRAIN_PAD": getattr(config, "TRAIN_PAD", None),
                "TRAIN_NOISE": getattr(config, "TRAIN_NOISE", None),
                "PHASE_TEMPORAL_WINDOW_MODE": str(getattr(config, "PHASE_TEMPORAL_WINDOW_MODE", "full")),
                "PHASE_TEMPORAL_WINDOW_MARGIN_MULT": float(getattr(config, "PHASE_TEMPORAL_WINDOW_MARGIN_MULT", 1.5)),
                "PHASE_TEMPORAL_WINDOW_JITTER_MULT": float(getattr(config, "PHASE_TEMPORAL_WINDOW_JITTER_MULT", 0.0)),
            },
            "args": {
                "num_frames": int(getattr(config, "NUM_FRAMES", 32)),
                "dataset_period": int(getattr(config, "DATASET_PERIOD", 1)),
                "dataset_max_length": getattr(config, "DATASET_MAX_LENGTH", None),
                "eval_clips": int(getattr(config, "EVAL_CLIPS", 1)),
                "train_pad": getattr(config, "TRAIN_PAD", None),
                "train_noise": getattr(config, "TRAIN_NOISE", None),
                "phase_temporal_window_mode": str(getattr(config, "PHASE_TEMPORAL_WINDOW_MODE", "full")),
                "phase_temporal_window_margin_mult": float(getattr(config, "PHASE_TEMPORAL_WINDOW_MARGIN_MULT", 1.5)),
                "phase_temporal_window_jitter_mult": float(getattr(config, "PHASE_TEMPORAL_WINDOW_JITTER_MULT", 0.0)),
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
    logger.info("Dataset period: %d", int(getattr(config, "DATASET_PERIOD", 1)))
    logger.info("Dataset max length: %s", getattr(config, "DATASET_MAX_LENGTH", None))
    logger.info("Eval clips: %d", int(getattr(config, "EVAL_CLIPS", 1)))
    logger.info("Train pad: %s", getattr(config, "TRAIN_PAD", None))
    logger.info("Train noise: %s", getattr(config, "TRAIN_NOISE", None))
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
    logger.info("Phase attn min gap: %d", int(getattr(config, "PHASE_ATTN_MIN_GAP", 1)))
    logger.info("Phase pair index weight: %.3f", float(getattr(config, "PHASE_PAIR_INDEX_WEIGHT", 0.0)))
    logger.info("Phase pair order weight: %.3f", float(getattr(config, "PHASE_PAIR_ORDER_WEIGHT", 0.0)))
    logger.info("Phase pair min gap: %d", int(getattr(config, "PHASE_PAIR_MIN_GAP", 1)))
    logger.info("Phase hard index weight: %.3f", float(getattr(config, "PHASE_HARD_INDEX_WEIGHT", 0.0)))
    logger.info("Phase unfreeze LR mult: %.3f", float(getattr(config, "PHASE_UNFREEZE_LR_MULT", 1.0)))
    logger.info("Weight decay: %s", getattr(config, "WEIGHT_DECAY", 0.0))
    logger.info("Max grad norm: %s", getattr(config, "MAX_GRAD_NORM", 0.0))
    logger.info("Phase temporal window mode: %s", str(getattr(config, "PHASE_TEMPORAL_WINDOW_MODE", "full")))
    logger.info("Phase temporal window margin mult: %.3f", float(getattr(config, "PHASE_TEMPORAL_WINDOW_MARGIN_MULT", 1.5)))
    logger.info("Phase temporal window jitter mult: %.3f", float(getattr(config, "PHASE_TEMPORAL_WINDOW_JITTER_MULT", 0.0)))


def main(argv=None):
    args = parse_args(argv)
    logger, _ = setup_logger()
    apply_runtime_overrides(args, logger)
    setup_performance_backends(logger)

    train_loader, val_loader, test_loader = build_dataloaders()
    model, optimizer, mse_loss, phase_index_loss_fn, amp_enabled, scaler = build_model_stack(logger)

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
            phase_index_loss_fn=phase_index_loss_fn,
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
                "Stage diagnostics | S1(T'~%.1f) feat-norm: %.3f temp-std: %.3f | S2(T~%.1f) attn-entropy: %.3f peak-w: %.3f peak->ED/ES MAE(fr): %.3f ED-ES feat-dist: %.3f | S3 index CE (ED/ES): %.3f / %.3f",
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
    logger.info("Test ED Accuracy: %.2f%%", test_metrics["ed_acc"] * 100)
    logger.info("Test ES Accuracy: %.2f%%", test_metrics["es_acc"] * 100)
    logger.info("Test Joint Accuracy: %.2f%%", test_metrics["joint_acc"] * 100)
    logger.info("Test ED MAE (frames): %.3f", test_metrics["ed_mae_frames"])
    logger.info("Test ES MAE (frames): %.3f", test_metrics["es_mae_frames"])
    logger.info(
        "Stage diagnostics (test) | S1(T'~%.1f) feat-norm: %.3f temp-std: %.3f | S2(T~%.1f) attn-entropy: %.3f peak-w: %.3f peak->ED/ES MAE(fr): %.3f ED-ES feat-dist: %.3f | S3 index CE (ED/ES): %.3f / %.3f",
        test_metrics["stage1_temporal_tokens"],
        test_metrics["stage1_feature_norm"],
        test_metrics["stage1_temporal_std"],
        test_metrics["stage2_temporal_tokens"],
        test_metrics["stage2_attention_entropy"],
        test_metrics["stage2_attention_peak"],
        test_metrics["stage2_peak_to_event_mae_frames"],
        test_metrics["stage2_ed_es_feature_distance"],
        test_metrics["stage3_ed_index_ce"],
        test_metrics["stage3_es_index_ce"],
    )
    logger.info("Test duration: %.2fs", test_duration)
    logger.info("=" * 80)


if __name__ == "__main__":
    main()


