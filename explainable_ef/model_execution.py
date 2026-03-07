import argparse
from datetime import datetime
import logging
import os
import sys
import time

import torch
import torch.nn as nn
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
    return parser.parse_args(argv)


def apply_runtime_overrides(args, logger):
    overrides = {}

    if args.smoke:
        overrides.update(SMOKE_DEFAULTS)

    if args.max_videos is not None:
        overrides["MAX_VIDEOS"] = args.max_videos
    if args.epochs is not None:
        overrides["EPOCHS"] = args.epochs
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

    for key, value in overrides.items():
        setattr(config, key, value)
        logger.info("Runtime override: %s=%s", key, value)

    return overrides


def is_cuda_runtime():
    return torch.cuda.is_available() and str(config.DEVICE).startswith("cuda")


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
    train_dataset = EchoDataset(
        config.DATA_DIR,
        split="TRAIN",
        num_frames=config.NUM_FRAMES,
        max_videos=config.MAX_VIDEOS,
    )
    val_dataset = EchoDataset(
        config.DATA_DIR,
        split="VAL",
        num_frames=config.NUM_FRAMES,
        max_videos=config.MAX_VIDEOS,
    )
    test_dataset = EchoDataset(
        config.DATA_DIR,
        split="TEST",
        num_frames=config.NUM_FRAMES,
        max_videos=config.MAX_VIDEOS,
    )

    train_loader = DataLoader(train_dataset, **dataloader_kwargs(shuffle=True))
    val_loader = DataLoader(val_dataset, **dataloader_kwargs(shuffle=False))
    test_loader = DataLoader(test_dataset, **dataloader_kwargs(shuffle=False))
    return train_loader, val_loader, test_loader


def build_model_stack():
    """Create model, optimizer and losses."""
    model = EFModel(num_frames=config.NUM_FRAMES).to(config.DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
    mse_loss = nn.MSELoss()
    ce_loss = nn.CrossEntropyLoss()

    amp_enabled = bool(getattr(config, "USE_MIXED_PRECISION", False)) and is_cuda_runtime()
    scaler = make_grad_scaler(amp_enabled)
    return model, optimizer, mse_loss, ce_loss, amp_enabled, scaler


def build_phase_targets(ed_idx, es_idx, num_frames, device):
    """Create per-frame phase target tensor with classes: 0=none, 1=ED, 2=ES."""
    batch_size = ed_idx.shape[0]
    phase_targets = torch.zeros(batch_size, num_frames, dtype=torch.long, device=device)
    for i in range(batch_size):
        phase_targets[i, ed_idx[i]] = 1
        phase_targets[i, es_idx[i]] = 2
    return phase_targets


def move_batch_to_device(videos, efs, ed_idx, es_idx):
    non_blocking = bool(getattr(config, "NON_BLOCKING_TRANSFER", True)) and is_cuda_runtime()
    videos = videos.to(config.DEVICE, non_blocking=non_blocking)
    efs = efs.to(config.DEVICE, non_blocking=non_blocking)
    ed_idx = ed_idx.to(config.DEVICE, non_blocking=non_blocking)
    es_idx = es_idx.to(config.DEVICE, non_blocking=non_blocking)
    return videos, efs, ed_idx, es_idx


def evaluate(model, loader, amp_enabled):
    """Evaluate EF regression and phase localization metrics."""
    model.eval()

    total_samples = 0
    total_mae = 0.0
    total_mse = 0.0
    total_ed_correct = 0
    total_es_correct = 0

    with torch.no_grad():
        for videos, efs, ed_idx, es_idx in loader:
            videos, efs, ed_idx, es_idx = move_batch_to_device(videos, efs, ed_idx, es_idx)

            with autocast_context(amp_enabled):
                ef_pred, _, phase_logits = model(videos)

            batch_size = videos.size(0)

            mae = torch.abs(ef_pred - efs).mean()
            mse = torch.mean((ef_pred - efs) ** 2)
            total_mae += mae.item() * batch_size
            total_mse += mse.item() * batch_size

            pred_ed_idx, pred_es_idx = Stage3PhaseDetector.predict_indices(phase_logits)
            total_ed_correct += (torch.abs(pred_ed_idx - ed_idx) <= config.TOLERANCE).sum().item()
            total_es_correct += (torch.abs(pred_es_idx - es_idx) <= config.TOLERANCE).sum().item()
            total_samples += batch_size

    metrics = {
        "ef_mae": total_mae / total_samples,
        "ef_rmse": (total_mse / total_samples) ** 0.5,
        "ed_acc": total_ed_correct / total_samples,
        "es_acc": total_es_correct / total_samples,
    }
    return metrics


def train_one_epoch(model, loader, optimizer, mse_loss, ce_loss, logger, epoch_idx, amp_enabled, scaler):
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

    for batch_idx, (videos, efs, ed_idx, es_idx) in enumerate(loader):
        data_time += time.perf_counter() - loop_end

        videos, efs, ed_idx, es_idx = move_batch_to_device(videos, efs, ed_idx, es_idx)
        batch_size = videos.size(0)
        total_samples += batch_size

        compute_start = time.perf_counter()

        with autocast_context(amp_enabled):
            ef_pred, attention, phase_logits = model(videos)

            ef_loss = mse_loss(ef_pred, efs)
            _, num_frames, _ = phase_logits.shape
            phase_targets = build_phase_targets(ed_idx, es_idx, num_frames, config.DEVICE)
            phase_loss = ce_loss(phase_logits.view(-1, 3), phase_targets.view(-1))

            loss = ef_loss + config.PHASE_LOSS_WEIGHT * phase_loss
            loss_for_backward = loss / accumulation_steps

        if amp_enabled:
            scaler.scale(loss_for_backward).backward()
        else:
            loss_for_backward.backward()

        should_step = ((batch_idx + 1) % accumulation_steps == 0) or (batch_idx + 1 == num_batches)
        if should_step:
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
                "Epoch %d batch %d | EF loss %.4f | Phase loss %.4f | GT ED/ES (%d/%d) | Pred ED/ES (%d/%d) | Attention shape %s",
                epoch_idx + 1,
                batch_idx,
                ef_loss.item(),
                phase_loss.item(),
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


def save_checkpoint(model, optimizer, val_mae, epoch):
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "val_mae": val_mae,
            "epoch": epoch,
        },
        config.CHECKPOINT_PATH,
    )


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
    logger.info("Validate every: %d epoch(s)", int(getattr(config, "VALIDATE_EVERY", 1)))


def main(argv=None):
    args = parse_args(argv)
    logger, _ = setup_logger()
    apply_runtime_overrides(args, logger)
    setup_performance_backends(logger)

    train_loader, val_loader, test_loader = build_dataloaders()
    model, optimizer, mse_loss, ce_loss, amp_enabled, scaler = build_model_stack()

    log_header(logger, amp_enabled)

    best_val_mae = float("inf")
    epochs_without_improvement = 0
    validate_every = max(1, int(getattr(config, "VALIDATE_EVERY", 1)))

    for epoch in range(config.EPOCHS):
        epoch_start = time.perf_counter()

        train_metrics = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            mse_loss=mse_loss,
            ce_loss=ce_loss,
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

            logger.info(
                "Epoch [%d/%d] | Train Loss: %.4f | Val EF MAE: %.2f%% | Val EF RMSE: %.2f%% | Val ED Acc: %.2f%% | Val ES Acc: %.2f%%",
                epoch + 1,
                config.EPOCHS,
                train_metrics["train_loss"],
                val_metrics["ef_mae"] * 100,
                val_metrics["ef_rmse"] * 100,
                val_metrics["ed_acc"] * 100,
                val_metrics["es_acc"] * 100,
            )

            if val_metrics["ef_mae"] < best_val_mae:
                best_val_mae = val_metrics["ef_mae"]
                epochs_without_improvement = 0
                save_checkpoint(model, optimizer, val_metrics["ef_mae"], epoch)
                logger.info("Saving best model to %s", config.CHECKPOINT_PATH)
            else:
                epochs_without_improvement += 1
                logger.info("No improvement (%d/%d)", epochs_without_improvement, config.PATIENCE)

            if epochs_without_improvement >= config.PATIENCE:
                logger.info("Early stopping triggered")
                break
        else:
            logger.info("Epoch [%d/%d] | Train Loss: %.4f | Validation skipped", epoch + 1, config.EPOCHS, train_metrics["train_loss"])

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
    model.load_state_dict(checkpoint["model_state_dict"])
    logger.info("Best Validation MAE: %.2f%%", checkpoint["val_mae"] * 100)

    test_start = time.perf_counter()
    test_metrics = evaluate(model, test_loader, amp_enabled=amp_enabled)
    test_duration = time.perf_counter() - test_start

    logger.info("=" * 80)
    logger.info("FINAL TEST RESULTS")
    logger.info("Test EF MAE: %.2f%%", test_metrics["ef_mae"] * 100)
    logger.info("Test EF RMSE: %.2f%%", test_metrics["ef_rmse"] * 100)
    logger.info("Test ED Accuracy: %.2f%%", test_metrics["ed_acc"] * 100)
    logger.info("Test ES Accuracy: %.2f%%", test_metrics["es_acc"] * 100)
    logger.info("Test duration: %.2fs", test_duration)
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
