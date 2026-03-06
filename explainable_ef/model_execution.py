import argparse
from datetime import datetime
import logging
import os
import sys

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

    for key, value in overrides.items():
        setattr(config, key, value)
        logger.info("Runtime override: %s=%s", key, value)

    return overrides


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

    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=config.BATCH_SIZE, shuffle=False)
    return train_loader, val_loader, test_loader


def build_model_stack():
    """Create model, optimizer and losses."""
    model = EFModel(num_frames=config.NUM_FRAMES).to(config.DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
    mse_loss = nn.MSELoss()
    ce_loss = nn.CrossEntropyLoss()
    return model, optimizer, mse_loss, ce_loss


def build_phase_targets(ed_idx, es_idx, num_frames, device):
    """Create per-frame phase target tensor with classes: 0=none, 1=ED, 2=ES."""
    batch_size = ed_idx.shape[0]
    phase_targets = torch.zeros(batch_size, num_frames, dtype=torch.long, device=device)
    for i in range(batch_size):
        phase_targets[i, ed_idx[i]] = 1
        phase_targets[i, es_idx[i]] = 2
    return phase_targets


def evaluate(model, loader):
    """Evaluate EF regression and phase localization metrics."""
    model.eval()

    total_samples = 0
    total_mae = 0.0
    total_mse = 0.0
    total_ed_correct = 0
    total_es_correct = 0

    with torch.no_grad():
        for videos, efs, ed_idx, es_idx in loader:
            videos = videos.to(config.DEVICE)
            efs = efs.to(config.DEVICE)
            ed_idx = ed_idx.to(config.DEVICE)
            es_idx = es_idx.to(config.DEVICE)

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


def train_one_epoch(model, loader, optimizer, mse_loss, ce_loss, logger, epoch_idx):
    """Run one training epoch and return average loss."""
    model.train()
    total_train_loss = 0.0

    for batch_idx, (videos, efs, ed_idx, es_idx) in enumerate(loader):
        videos = videos.to(config.DEVICE)
        efs = efs.to(config.DEVICE)
        ed_idx = ed_idx.to(config.DEVICE)
        es_idx = es_idx.to(config.DEVICE)

        optimizer.zero_grad()

        ef_pred, attention, phase_logits = model(videos)

        ef_loss = mse_loss(ef_pred, efs)
        _, num_frames, _ = phase_logits.shape
        phase_targets = build_phase_targets(ed_idx, es_idx, num_frames, config.DEVICE)
        phase_loss = ce_loss(phase_logits.view(-1, 3), phase_targets.view(-1))

        loss = ef_loss + config.PHASE_LOSS_WEIGHT * phase_loss
        loss.backward()
        optimizer.step()

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

    return total_train_loss / len(loader)


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


def log_header(logger):
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


def main(argv=None):
    args = parse_args(argv)
    logger, _ = setup_logger()
    apply_runtime_overrides(args, logger)
    log_header(logger)

    train_loader, val_loader, test_loader = build_dataloaders()
    model, optimizer, mse_loss, ce_loss = build_model_stack()

    best_val_mae = float("inf")
    epochs_without_improvement = 0

    for epoch in range(config.EPOCHS):
        train_loss = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            mse_loss=mse_loss,
            ce_loss=ce_loss,
            logger=logger,
            epoch_idx=epoch,
        )

        val_metrics = evaluate(model, val_loader)

        logger.info(
            "Epoch [%d/%d] | Train Loss: %.4f | Val EF MAE: %.2f%% | Val EF RMSE: %.2f%% | Val ED Acc: %.2f%% | Val ES Acc: %.2f%%",
            epoch + 1,
            config.EPOCHS,
            train_loss,
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

    logger.info("Loading best model for final testing")
    checkpoint = torch.load(config.CHECKPOINT_PATH, map_location=config.DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])
    logger.info("Best Validation MAE: %.2f%%", checkpoint["val_mae"] * 100)

    test_metrics = evaluate(model, test_loader)
    logger.info("=" * 80)
    logger.info("FINAL TEST RESULTS")
    logger.info("Test EF MAE: %.2f%%", test_metrics["ef_mae"] * 100)
    logger.info("Test EF RMSE: %.2f%%", test_metrics["ef_rmse"] * 100)
    logger.info("Test ED Accuracy: %.2f%%", test_metrics["ed_acc"] * 100)
    logger.info("Test ES Accuracy: %.2f%%", test_metrics["es_acc"] * 100)
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
