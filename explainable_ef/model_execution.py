from datetime import datetime
import logging
import sys

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import os

from data.dataset import EchoDataset
from models.ef_model import EFModel
import config

def setup_logger():
    """Setup logging configuration."""
    log_dir = "logs"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"training_{timestamp}.log")

    # Create logger
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)

    # File handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.DEBUG)

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)

    # Configure encoding for console output (UTF-8)
    if sys.platform == 'win32':
        import io
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

    # Formatter
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger, log_file


# ==============================
# CONFIG
# ==============================


# ==============================
# DATASETS
# ==============================
train_dataset = EchoDataset(config.DATA_DIR, split="TRAIN", num_frames=config.NUM_FRAMES, max_videos=config.MAX_VIDEOS)
val_dataset   = EchoDataset(config.DATA_DIR, split="VAL",   num_frames=config.NUM_FRAMES, max_videos=config.MAX_VIDEOS)

train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True)
val_loader   = DataLoader(val_dataset,   batch_size=config.BATCH_SIZE, shuffle=False)

# ==============================
# MODEL
# ==============================
model = EFModel(num_frames=config.NUM_FRAMES).to(config.DEVICE)

optimizer = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
mse_loss = nn.MSELoss()
ce_loss = nn.CrossEntropyLoss()

# ==============================
# EVALUATION FUNCTION
# ==============================
def evaluate(model, loader):
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

            batch_size = videos.size(0)

            ef_pred, attn, phase_logits = model(videos)

            # ---- EF Metrics ----
            mae = torch.abs(ef_pred - efs).mean()
            mse = torch.mean((ef_pred - efs) ** 2)

            total_mae += mae.item() * batch_size
            total_mse += mse.item() * batch_size

            # ---- Phase Accuracy ----
            pred_ed_idx = torch.argmax(phase_logits[:, :, 1], dim=1)
            pred_es_idx = torch.argmax(phase_logits[:, :, 2], dim=1)

            ed_correct = (
                torch.abs(pred_ed_idx - ed_idx) <= config.TOLERANCE
            ).sum()

            es_correct = (
                torch.abs(pred_es_idx - es_idx) <= config.TOLERANCE
            ).sum()

            total_ed_correct += ed_correct.item()
            total_es_correct += es_correct.item()

            total_samples += batch_size

    avg_mae = total_mae / total_samples
    avg_rmse = (total_mse / total_samples) ** 0.5
    ed_acc = total_ed_correct / total_samples
    es_acc = total_es_correct / total_samples

    return avg_mae, avg_rmse, ed_acc, es_acc

# ==============================
# TRAINING LOOP
# ==============================
best_val_mae = float("inf")
epochs_without_improvement = 0

logger, log_file = setup_logger()
    
logger.info("="*80)
logger.info("TRAINING SCRIPT STARTED")
logger.info("="*80)

# Log configuration
logger.info(f"Using device: {config.DEVICE}")
logger.info(f"Data directory: {config.DATA_DIR}")
logger.info(f"Batch size: {config.BATCH_SIZE}")
logger.info(f"Number of frames: {config.NUM_FRAMES}")
logger.info(f"Max videos: {config.MAX_VIDEOS if config.MAX_VIDEOS else 'All'}")
logger.info(f"Learning rate: {config.LEARNING_RATE}")
logger.info(f"Epochs: {config.EPOCHS}")
logger.info(f"Workers: {config.NUM_WORKERS}")

for epoch in range(config.EPOCHS):

    model.train()
    total_train_loss = 0.0

    print(f"\nEpoch #{epoch+1}")
    logger.info(f"\nEpoch [{epoch+1}/{config.EPOCHS}]")

    for videos, efs, ed_idx, es_idx in train_loader:

        videos = videos.to(config.DEVICE)
        efs = efs.to(config.DEVICE)
        ed_idx = ed_idx.to(config.DEVICE)
        es_idx = es_idx.to(config.DEVICE)

        optimizer.zero_grad()

        ef_pred, attn, phase_logits = model(videos)

        # ---- EF Loss ----
        ef_loss = mse_loss(ef_pred, efs)

        # ---- Phase Targets ----
        B, T, _ = phase_logits.shape
        phase_targets = torch.zeros(B, T, dtype=torch.long).to(config.DEVICE)

        for i in range(B):
            phase_targets[i, ed_idx[i]] = 1
            phase_targets[i, es_idx[i]] = 2

        phase_loss = ce_loss(
            phase_logits.view(-1, 3),
            phase_targets.view(-1)
        )

        # ---- Combined Loss ----
        loss = ef_loss + config.PHASE_LOSS_WEIGHT * phase_loss

        loss.backward()
        optimizer.step()

        total_train_loss += loss.item()

        # Predicted indices
        pred_ed_idx = torch.argmax(phase_logits[:, :, 1], dim=1)
        pred_es_idx = torch.argmax(phase_logits[:, :, 2], dim=1)
        
        # Print for first sample in batch
        print("----- Phase Debug -----")
        print("GT ED frame:", ed_idx[0].item())
        print("Pred ED frame:", pred_ed_idx[0].item())
        print("ED frame error:", abs(pred_ed_idx[0].item() - ed_idx[0].item()))

        print("GT ES frame:", es_idx[0].item())
        print("Pred ES frame:", pred_es_idx[0].item())
        print("ES frame error:", abs(pred_es_idx[0].item() - es_idx[0].item()))
        print("------------------------")

        print(f"Predicted EF: {ef_pred.mean().item():.4f}, True EF: {efs.mean().item():.4f}")
        print("Attention shape:", attn.shape)
        print("Attention sample:", attn[0][:])

    avg_train_loss = total_train_loss / len(train_loader)

    # =========================
    # VALIDATION
    # =========================
    val_mae, val_rmse, val_ed_acc, val_es_acc = evaluate(model, val_loader)

    print(f"\nEpoch [{epoch+1}/{config.EPOCHS}]")
    print(f"Train Loss: {avg_train_loss:.4f}")
    print(f"Val EF MAE (%): {val_mae * 100:.2f}")
    print(f"Val EF RMSE (%): {val_rmse * 100:.2f}")
    print(f"Val ED Acc: {val_ed_acc * 100:.2f}%")
    print(f"Val ES Acc: {val_es_acc * 100:.2f}%")

    # =========================
    # EARLY STOPPING & CHECKPOINT
    # =========================
    if val_mae < best_val_mae:
        print("✔ Saving best model...")
        best_val_mae = val_mae
        epochs_without_improvement = 0

        torch.save({
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "val_mae": val_mae,
            "epoch": epoch
        }, config.CHECKPOINT_PATH)

    else:
        epochs_without_improvement += 1
        print(f"No improvement ({epochs_without_improvement}/{config.PATIENCE})")

    if epochs_without_improvement >= config.PATIENCE:
        print("⛔ Early stopping triggered.")
        break

# ==============================
# LOAD BEST MODEL FOR TEST
# ==============================
print("\nLoading best model for testing...")

checkpoint = torch.load(config.CHECKPOINT_PATH, map_location=config.DEVICE)
model.load_state_dict(checkpoint["model_state_dict"])

test_dataset  = EchoDataset(config.DATA_DIR, split="TEST",  num_frames=config.NUM_FRAMES, max_videos=config.MAX_VIDEOS)
test_loader  = DataLoader(test_dataset,  batch_size=config.BATCH_SIZE, shuffle=False)

print(f"Best Validation MAE (%): {checkpoint['val_mae'] * 100:.2f}")

# ==============================
# TEST EVALUATION
# ==============================
test_mae, test_rmse, test_ed_acc, test_es_acc = evaluate(model, test_loader)

print("\n===== FINAL TEST RESULTS =====")
print(f"Test EF MAE (%): {test_mae * 100:.2f}")
print(f"Test EF RMSE (%): {test_rmse * 100:.2f}")
print(f"Test ED Accuracy: {test_ed_acc * 100:.2f}%")
print(f"Test ES Accuracy: {test_es_acc * 100:.2f}%")
print("================================")