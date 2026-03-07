import torch
import os

# Data configuration
# Points to ../dynamic/a4c-video-dir (from cx/explainable_ef to cx/dynamic)
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "dynamic", "a4c-video-dir")

# Model configuration
BATCH_SIZE = 20
NUM_FRAMES = 32  # Increase to 48/64 for finer phase localization
IMAGE_SIZE = 112
MAX_VIDEOS = None  # None = use all videos
PHASE_LOSS_WEIGHT = 0.5  # Weight of phase index loss relative to EF regression loss
PHASE_LABEL_SMOOTHING = 0.0  # Label smoothing for hard CE on ED/ES temporal indices
PHASE_ONLY = False  # If True, disable EF loss and optimize only phase detection

# Phase training stabilizers
PHASE_BACKBONE_FREEZE_EPOCHS = 5  # Freeze stage1 backbone for early phase-only warmup
BACKBONE_LR_MULT = 0.2  # Backbone LR = LEARNING_RATE * BACKBONE_LR_MULT
PHASE_SOFT_SIGMA = 1.5  # >0 enables soft temporal targets (Gaussian around GT index)
PHASE_SOFT_RADIUS = 4  # Optional radius for soft target support; <=0 means no clipping
PHASE_FRAME_CE_WEIGHT = 0.35  # Mix ratio for frame-wise CE in phase loss
PHASE_FRAME_RADIUS = 2  # Radius around ED/ES used for frame-wise supervision

# Training configuration
LEARNING_RATE = 1e-4
EPOCHS = 50
TOLERANCE = 1
PATIENCE = 10
VALIDATE_EVERY = 1  # Run validation every N epochs

# Device configuration
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CHECKPOINT_PATH = "best_model.pth"

# ============================================================================
# PERFORMANCE OPTIMIZATION SETTINGS
# ============================================================================
NUM_WORKERS = 8
PIN_MEMORY = True
PERSISTENT_WORKERS = True
PREFETCH_FACTOR = 4
NON_BLOCKING_TRANSFER = True

# GPU compute acceleration
USE_MIXED_PRECISION = True
ENABLE_TF32 = True
CUDNN_BENCHMARK = True

# Use gradient accumulation (slower but uses less memory)
GRADIENT_ACCUMULATION_STEPS = 1

# Cache decoded frames in memory (requires more RAM)
CACHE_FRAMES = False
