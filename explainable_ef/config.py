import torch
import os

# Data configuration
# Points to ../../dynamic/a4c-video-dir (from cx/explainable_ef to CardioXplain/dynamic)
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "dynamic", "a4c-video-dir")

# Model configuration
BATCH_SIZE = 20
NUM_FRAMES = 32  # Reduce to 16 for 2x speedup with minimal quality loss
IMAGE_SIZE = 112  # Reduce to 64 for 2x speedup, 56 for 3x speedup
MAX_VIDEOS = None  # None = use all videos, or specify number (e.g., 100)
PHASE_LOSS_WEIGHT = 0.5  # Balance between EF regression loss and phase classification loss

# Training configuration
LEARNING_RATE = 1e-4
EPOCHS = 50
TOLERANCE = 1
PATIENCE = 5
VALIDATE_EVERY = 1  # Run validation every N epochs

# Device configuration - automatically detect CUDA availability
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CHECKPOINT_PATH = "best_model.pth"

# ============================================================================
# PERFORMANCE OPTIMIZATION SETTINGS
# ============================================================================
# Adjust these for faster training

# DataLoader workers - increase for fast storage + many CPU cores
NUM_WORKERS = 8
PIN_MEMORY = True
PERSISTENT_WORKERS = True
PREFETCH_FACTOR = 4
NON_BLOCKING_TRANSFER = True

# GPU compute acceleration
USE_MIXED_PRECISION = True  # Only active when DEVICE is CUDA
ENABLE_TF32 = True
CUDNN_BENCHMARK = True

# Use gradient accumulation (slower but uses less memory)
GRADIENT_ACCUMULATION_STEPS = 1

# Cache decoded frames in memory (requires more RAM but much faster training)
# Only enable if you have >16GB RAM
CACHE_FRAMES = False
