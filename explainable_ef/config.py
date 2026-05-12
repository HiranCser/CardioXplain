import torch
import os

# Data configuration
# Points to ../dynamic/a4c-video-dir (from cx/explainable_ef to cx/dynamic)
#DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "dynamic", "a4c-video-dir")
DATA_DIR = '/datasets/efnet1/a4c-video-dir'  # Override for absolute path if needed


# Model configuration
BATCH_SIZE = 20
NUM_FRAMES = 64  # Smaller temporal search space improves ED/ES localization stability
IMAGE_SIZE = 112
MAX_VIDEOS = None  # LIMITED TO 10 VIDEOS FOR TESTING

# Continuous cardiac phase prediction (new approach)
# Instead of classifying ED/ES directly, we predict continuous phase ∈ [0, 1]:
# - ED (End-Diastole, max LV cavity) → phase ≈ 0.0
# - ES (End-Systole, min LV cavity) → phase ≈ 0.5
# - Full cycle: 0.0 → 0.5 → 1.0 → (back to 0.0)
PHASE_LOSS_WEIGHT = 1.5  # Weight of phase regression loss relative to EF regression loss
PHASE_ONLY = False  # If True, disable EF loss and optimize only phase detection
PHASE_ONLY_WARMUP_EPOCHS = 5  # Joint mode can temporarily disable EF loss for phase learning warmup

# These settings are kept for backward compatibility but are no longer used:
PHASE_LABEL_SMOOTHING = 0.0
PHASE_BACKBONE_FREEZE_EPOCHS = 0
BACKBONE_LR_MULT = 0.2
PHASE_SOFT_SIGMA = 1.5  # Not used for regression
PHASE_SOFT_RADIUS = 0   # Not used for regression
PHASE_HARD_INDEX_WEIGHT = 0.15  # Not used for regression
PHASE_FRAME_CE_WEIGHT = 0.0    # Not used for regression
PHASE_FRAME_RADIUS = 4         # Not used for regression
PHASE_ATTN_ALIGN_WEIGHT = 0.3  # Not used for regression
PHASE_ATTN_ALIGN_SIGMA = 0.0   # Not used for regression
PHASE_ATTN_ALIGN_RADIUS = 0    # Not used for regression
PHASE_ATTN_INDEX_WEIGHT = 0.0  # Not used for regression
PHASE_ATTN_ORDER_WEIGHT = 0.0  # Not used for regression
PHASE_ATTN_ENTROPY_WEIGHT = 0.0  # Not used for regression
PHASE_ATTN_MIN_GAP = 15        # Used for ED/ES extraction from phase
PHASE_PAIR_INDEX_WEIGHT = 0.0  # Not used for regression
PHASE_PAIR_ORDER_WEIGHT = 0.0  # Not used for regression
PHASE_PAIR_MIN_GAP = 15        # Used for ED/ES extraction from phase
PHASE_UNFREEZE_LR_MULT = 0.5   # Not used for regression
CLIP_PERIOD = 1  # EchoNet-style fixed clip sampling stride
CLIP_EVAL_MODE = "all"  # "center" for single-clip validation/test, "all" for callers that aggregate clips

# Training configuration
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4
MAX_GRAD_NORM = 1.0  # 0 disables gradient clipping
EPOCHS = 50  # LIMITED TO 1 EPOCH FOR TESTING
TOLERANCE = 1
PATIENCE = 5
VALIDATE_EVERY = 1  # Run validation every N epochs

# Device configuration
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CHECKPOINT_PATH = "best_model_stage123_96f.pth"
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
