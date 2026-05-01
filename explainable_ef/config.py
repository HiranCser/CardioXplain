import torch
import os

# Data configuration
# Points to ../dynamic/a4c-video-dir (from cx/explainable_ef to cx/dynamic)
DATA_DIR = '/kaggle/input/datasets/hirancser/echoefnet/a4c-video-dir'

# Model configuration
BATCH_SIZE = 20
NUM_FRAMES = 32  # Increase to 48/64 for finer phase localization
IMAGE_SIZE = 112
MAX_VIDEOS = None  # None = use all videos
DATASET_PERIOD = 1  # EchoNet-style temporal stride within a clip
DATASET_MAX_LENGTH = None  # Optional cap on sampled clip length
EVAL_CLIPS = 1  # EchoNet-style multi-clip evaluation; training remains single-clip
TRAIN_PAD = None  # EchoNet-style random spatial pad/crop augmentation
TRAIN_NOISE = None  # EchoNet-style random blackout noise fraction
PHASE_LOSS_WEIGHT = 0.5  # Weight of phase index loss relative to EF regression loss
PHASE_LABEL_SMOOTHING = 0.0  # Label smoothing for hard CE on ED/ES temporal indices
PHASE_ONLY = False  # If True, disable EF loss and optimize only phase detection

# Phase training stabilizers
PHASE_BACKBONE_FREEZE_EPOCHS = 5  # Freeze stage1 backbone for early phase-only warmup
BACKBONE_LR_MULT = 0.2  # Backbone LR = LEARNING_RATE * BACKBONE_LR_MULT
PHASE_SOFT_SIGMA = 1.5  # >0 enables soft temporal targets (Gaussian around GT index)
PHASE_SOFT_RADIUS = 4  # Optional radius for soft target support; <=0 means no clipping
PHASE_HARD_INDEX_WEIGHT = 0.5  # Mix hard CE with soft index loss when soft targets are enabled
PHASE_FRAME_CE_WEIGHT = 0.35  # Mix ratio for frame-wise CE in phase loss
PHASE_FRAME_RADIUS = 2  # Radius around ED/ES used for frame-wise supervision
PHASE_ATTN_ALIGN_WEIGHT = 0.35  # KL alignment weight for Stage2 temporal attention around ED/ES
PHASE_ATTN_ALIGN_SIGMA = 2.0  # Gaussian sigma for attention alignment targets
PHASE_ATTN_ALIGN_RADIUS = 5  # Optional support radius for attention alignment targets
PHASE_ATTN_INDEX_WEIGHT = 0.25  # Direct supervision on Stage2 attention head expected ED/ES indices
PHASE_ATTN_ORDER_WEIGHT = 0.08  # Encourage ED attention to remain before ES attention
PHASE_ATTN_MIN_GAP = 2  # Minimum frame gap enforced between ED and ES attention expectations
PHASE_PAIR_INDEX_WEIGHT = 0.18  # Direct Stage3 ED/ES expectation supervision using the same score curves as inference
PHASE_PAIR_ORDER_WEIGHT = 0.08  # Encourage Stage3 ES expectation to remain after ED expectation
PHASE_PAIR_MIN_GAP = 2  # Minimum frame gap enforced between Stage3 ED and ES expectations
PHASE_UNFREEZE_LR_MULT = 0.5  # Multiply LR when unfreezing stage1 in phase-only mode
PHASE_TEMPORAL_WINDOW_MODE = "full"  # Legacy compatibility knob; dataset sampling is now label-free and EchoNet-style
PHASE_TEMPORAL_WINDOW_MARGIN_MULT = 1.5  # Legacy compatibility knob
PHASE_TEMPORAL_WINDOW_JITTER_MULT = 0.0  # Legacy compatibility knob

# Training configuration
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4
MAX_GRAD_NORM = 1.0  # 0 disables gradient clipping
EPOCHS = 50
TOLERANCE = 1
PATIENCE = 5
VALIDATE_EVERY = 1  # Run validation every N epochs

# Device configuration
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CHECKPOINT_PATH = "best_model.pth"
STAGE4_CHECKPOINT_PATH = "best_stage4_segmentation_area.pth"

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
NORMALIZE_INPUT = True

# Use gradient accumulation (slower but uses less memory)
GRADIENT_ACCUMULATION_STEPS = 1

# Cache decoded frames in memory (requires more RAM)
CACHE_FRAMES = False
