import torch
import os

# Data configuration
# Points to ../../dynamic/a4c-video-dir (from cx/explainable_ef to CardioXplain/dynamic)
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "dynamic", "a4c-video-dir")

# Model configuration
BATCH_SIZE = 20
NUM_FRAMES = 32  # Reduce to 16 for 2x speedup with minimal quality loss
IMAGE_SIZE = 112  # Reduce to 64 for 2x speedup, 56 for 3x speedup
MAX_VIDEOS = 100  # None = use all videos, or specify number (e.g., 100)

# Training configuration
LR = 1e-4
EPOCHS = 10

# Device configuration - automatically detect CUDA availability
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ============================================================================
# PERFORMANCE OPTIMIZATION SETTINGS
# ============================================================================
# Adjust these for faster training

# DataLoader workers - parallel data loading
# 0 = single process (simpler, no multiprocessing overhead)
# 2-4 = use multiple cores for data loading (may add overhead on older systems)
NUM_WORKERS = 0  # Set to 0 for single-process data loading

# Enable mixed precision training (requires GPU)
USE_MIXED_PRECISION = False  # Only works with CUDA

# Use gradient accumulation (slower but uses less memory)
GRADIENT_ACCUMULATION_STEPS = 1

# Cache decoded frames in memory (requires more RAM but much faster training)
# Only enable if you have >16GB RAM
CACHE_FRAMES = False