# Explainable EF (CardioXplain)

This repository trains and evaluates an echocardiography EF pipeline with stage-based modules:

- Stage 1: Feature extraction
- Stage 2: Temporal modeling (learns frame attention/temporal weights)
- Stage 3: Phase detection (ED/ES frame localization)
- Stage 4: LV segmentation (trainable model)
- Stage 5: EF computation (from ED/ES areas)

## Stage 2 vs Stage 3 (important)

Stage 2 owns temporal weighting. It outputs attention weights over frames.

Stage 3 owns phase supervision. Its ED/ES losses teach the network where cardiac events occur. That supervision backpropagates through Stage 3 into Stage 2, so Stage 2 temporal weights improve even though there is no separate "ground-truth temporal-weight" file.

## Recommended Model Split

Phase detection is treated as a first-class task. For experiments where ED/ES localization is as important as EF regression, use two checkpoints:

- `best_phase_detector.pth`: dedicated Stage 1/2/3 phase detector trained with EF loss disabled.
- `best_model.pth` or `best_model_stage123_96f.pth`: EF-oriented Stage 1/2/3 model trained with EF loss enabled.

This avoids forcing one shared checkpoint to optimize two objectives that can compete: EF regression needs global functional evidence, while phase detection needs sharper temporal localization. The phase detector reports ED accuracy, ES accuracy, joint ED+ES accuracy, and frame MAE. A 95% phase target is configured as a validation/test goal, not as a guaranteed result.

## 1. Prerequisites

- Python 3.10+
- `pip`
- NVIDIA GPU recommended for full runs

## 2. Setup

```powershell
cd D:\datascience\MTech\Sem4\Project\CardioXplain\cx\explainable_ef
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install torch torchvision pandas numpy opencv-python matplotlib tqdm
```

If you need a specific CUDA build, install `torch`/`torchvision` from the official PyTorch selector first.

## 3. Dataset Layout

Default path from `config.py`:

`D:\datascience\MTech\Sem4\Project\CardioXplain\dynamic\a4c-video-dir`

Expected structure:

```text
a4c-video-dir/
  FileList.csv
  VolumeTracings.csv
  Videos/
    *.avi
```

## 4. Training and Validation Workflows

### A) Dedicated phase detector (recommended for phase accuracy)

```powershell
python pipeline\train_phase_detector.py --init-checkpoint best_model.pth --num-frames 32 --epochs 50 --workers 8 --phase-target-accuracy 0.95
```

This trains a separate Stage 1/2/3 phase checkpoint with EF loss disabled and saves to `best_phase_detector.pth` by default. It monitors validation joint ED/ES accuracy and can stop once the target is reached. Warm-starting from `best_model.pth` is recommended because scratch phase training has to learn echo features and ED/ES localization at the same time.

The phase-detector preset uses `phase_window` sampling for TRAIN and `global` sampling for VAL/TEST. `phase_window` creates training clips that contain the labeled ED/ES frames, giving the model denser phase supervision without making validation/test label-aware by default.

For stricter reporting, keep the default `--tolerance 1` frame tolerance or set it explicitly:

```powershell
python pipeline\train_phase_detector.py --init-checkpoint best_model.pth --num-frames 32 --epochs 50 --workers 8 --tolerance 1 --phase-target-accuracy 0.95
```

To compare sampling strategies:

```powershell
python pipeline\train_phase_detector.py --init-checkpoint best_model.pth --num-frames 32 --train-sampling-mode phase_window --eval-sampling-mode global --epochs 50 --workers 8 --batch-size 5 --tolerance 1
```

### B) Combined Stage 1+2+3 (joint EF + phase)

```powershell
python model_execution.py --no-phase-only --checkpoint best_model.pth
```

This trains Stage 1/2/3 jointly and keeps EF loss enabled. Use this as the EF-oriented model when you want a separate phase detector.

### C) Phase-only Stage 1+2+3 through the base trainer

```powershell
python model_execution.py --phase-detector
```

This is equivalent to the dedicated wrapper and uses the phase-only preset.

### D) Default Stage 1+2+3 training

```powershell
python model_execution.py
```

### E) Smoke test (tiny run)

```powershell
python model_execution.py --smoke
```

### F) Stage 3 validation + visualization

```powershell
python visualization\validate_phase_detection.py --split TEST --checkpoint best_phase_detector.pth --num-samples 12
```

### G) ED/ES validation against reference annotations

Use your expert frame table (CSV/XLSX), or use `VolumeTracings.csv` directly via `--reference-mode volume_tracings`.

```powershell
python validation\validate_ed_es_against_reference.py --split VAL --reference "D:\datascience\MTech\Sem4\Project\CardioXplain\dynamic\a4c-video-dir\VolumeTracings.csv" --reference-mode volume_tracings --tolerance 1 --output validation\outputs\ed_es_validation.csv --mismatch-output validation\outputs\ed_es_mismatches.csv
```

### H) Generate annotation template for manual ED/ES review

```powershell
python validation\generate_reference_frame_template.py --split VAL --include-meta --prefill-from-tracings --output validation\outputs\ed_es_template_val.csv
```

### I) Stage 4 full training + area validation (per frame + per video)

Recommended:

```powershell
python pipeline\train_stage4_segmentation.py --data-dir "D:\datascience\MTech\Sem4\Project\CardioXplain\dynamic\a4c-video-dir" --model-name deeplabv3_resnet50 --pretrained --batch-size 20 --epochs 50 --learning-rate 1e-4 --optimizer adamw --workers 8 --image-size 112 --eval-threshold 0.5
```

CPU/smoke:

```powershell
python pipeline\train_stage4_segmentation.py --model-name unet --batch-size 4 --epochs 2 --max-videos 24 --no-amp
```

Outputs:

- `best_stage4_segmentation.pth`
- `validation/outputs/stage4/val_best_frame_areas.csv`
- `validation/outputs/stage4/val_best_frame_areas_video_summary.csv`
- `validation/outputs/stage4/test_frame_areas.csv`
- `validation/outputs/stage4/test_frame_areas_video_summary.csv`

### J) Stage 4/5 predicted-mask utility

```powershell
python pipeline\run_stage45_from_tracings.py --split VAL --max-videos 25 --save-overlays
```

### K) Stage 6 + Stage 7 training (similarity + uncertainty)

```powershell
python pipeline\train_stage67_similarity.py --stage123-checkpoint best_model.pth --num-frames 32
```

Outputs:

- `validation/outputs/stage67/stage6_similarity_engine.npz`
- `validation/outputs/stage67/stage7_calibration.json`
- `validation/outputs/stage67/stage67_summary.json`
- per-split prediction CSVs (train/val/test)

### L) Full Stage 1-7 orchestration (single command)

```powershell
python pipeline\train_all_stages.py --train-phase-detector --stage123-num-frames 32 --stage123-epochs 50 --phase-target-accuracy 0.95 --stage4-epochs 50 --stage4-optimizer adamw --stage4-learning-rate 1e-4
```

This orchestrates:

- Stage1-3 EF-oriented training (`model_execution.py --no-phase-only`)
- Optional dedicated phase detector training (`pipeline/train_phase_detector.py`)
- Stage4 segmentation training
- Stage5 EF evaluation from predicted Stage4 masks (TRAIN/VAL/TEST)
- Stage6 similarity training + Stage7 uncertainty calibration using Stage5 predicted-mask metrics

Stage4 defaults are now class-imbalance aware (`--eval-threshold 0.5`, auto BCE `pos_weight`) to prevent empty-mask collapse.

Use `--skip-stage123`, `--skip-stage4`, `--skip-stage5`, `--skip-stage67` to resume partial runs.

### M) One-page UI for Stage 1-5 outputs (recommended for review)

Install Streamlit once:

```powershell
pip install streamlit
```

Launch UI:

```powershell
streamlit run ui\stage_results_app.py
```

UI behavior:

- Loads videos from selected split (default `TEST`) in `FileList.csv`
- User selects any video case from dropdown
- Source `.avi` videos are auto-converted to browser-friendly MP4 preview (with frame-view fallback)
- Runs Stage1-3 checkpoint inference and shows:
  - Stage1 diagnostics
  - Stage2 temporal weights curve
  - Stage3 ED/ES predictions + errors
  - EF prediction vs GT
- If Stage4 checkpoint is provided, also runs Stage4/5 and shows:
  - Predicted ED/ES masks on selected frames
  - Area metrics and Dice (if traced frame available)
  - EF from mask areas (Stage5)
- Adds auto explanation text for quick interpretation

## 5. Stage-wise metrics now printed in logs (Stage 1-3 runs)

Validation and test now report:

- Stage 1 diagnostics: feature norm, temporal std, effective temporal tokens `T'`
- Stage 2 diagnostics: attention entropy, attention peak weight, peak-to-(ED/ES) frame MAE, ED-ES feature distance, temporal tokens `T`
- Stage 3 diagnostics: ED/ES index CE, ED/ES/joint tolerance accuracy, ED/ES frame MAE
- EF diagnostics (if not phase-only): EF MAE and EF RMSE

## 6. CLI Help Options (all current scripts)

Run any script with `--help` for the latest values/defaults.

### `model_execution.py` options

- `-h, --help`
- `--smoke`
- `--phase-detector`
- `--max-videos MAX_VIDEOS`
- `--epochs EPOCHS`
- `--learning-rate, --lr LEARNING_RATE`
- `--batch-size BATCH_SIZE`
- `--num-frames NUM_FRAMES`
- `--image-size IMAGE_SIZE`
- `--dataset-period DATASET_PERIOD`
- `--dataset-max-length DATASET_MAX_LENGTH`
- `--sampling-mode {global,echonet,phase_window}`
- `--train-sampling-mode {global,echonet,phase_window}`
- `--eval-sampling-mode {global,echonet,phase_window}`
- `--checkpoint CHECKPOINT`
- `--workers WORKERS`
- `--validate-every VALIDATE_EVERY`
- `--prefetch-factor PREFETCH_FACTOR`
- `--amp, --no-amp`
- `--pin-memory, --no-pin-memory`
- `--persistent-workers, --no-persistent-workers`
- `--tf32, --no-tf32`
- `--benchmark, --no-benchmark`
- `--normalize-input, --no-normalize-input`
- `--phase-loss-weight PHASE_LOSS_WEIGHT`
- `--phase-label-smoothing PHASE_LABEL_SMOOTHING`
- `--phase-only, --no-phase-only`
- `--phase-target-accuracy PHASE_TARGET_ACCURACY`
- `--stop-on-phase-target, --no-stop-on-phase-target`
- `--phase-backbone-freeze-epochs PHASE_BACKBONE_FREEZE_EPOCHS`
- `--backbone-lr-mult BACKBONE_LR_MULT`
- `--phase-soft-sigma PHASE_SOFT_SIGMA`
- `--phase-soft-radius PHASE_SOFT_RADIUS`
- `--phase-hard-index-weight PHASE_HARD_INDEX_WEIGHT`
- `--phase-frame-ce-weight PHASE_FRAME_CE_WEIGHT`
- `--phase-frame-radius PHASE_FRAME_RADIUS`
- `--phase-attn-align-weight PHASE_ATTN_ALIGN_WEIGHT`
- `--phase-attn-align-sigma PHASE_ATTN_ALIGN_SIGMA`
- `--phase-attn-align-radius PHASE_ATTN_ALIGN_RADIUS`
- `--phase-unfreeze-lr-mult PHASE_UNFREEZE_LR_MULT`
- `--weight-decay WEIGHT_DECAY`
- `--max-grad-norm MAX_GRAD_NORM`
- `--tolerance TOLERANCE`

### `pipeline/train_phase_detector.py` options

- `-h, --help`
- `--checkpoint CHECKPOINT`
- `--init-checkpoint INIT_CHECKPOINT`
- `--epochs EPOCHS`
- `--learning-rate, --lr LEARNING_RATE`
- `--batch-size BATCH_SIZE`
- `--num-frames NUM_FRAMES`
- `--image-size IMAGE_SIZE`
- `--dataset-period DATASET_PERIOD`
- `--dataset-max-length DATASET_MAX_LENGTH`
- `--train-sampling-mode {global,echonet,phase_window}`
- `--eval-sampling-mode {global,echonet,phase_window}`
- `--max-videos MAX_VIDEOS`
- `--workers WORKERS`
- `--prefetch-factor PREFETCH_FACTOR`
- `--validate-every VALIDATE_EVERY`
- `--homogenization-stats HOMOGENIZATION_STATS`
- `--phase-target-accuracy PHASE_TARGET_ACCURACY`
- `--stop-on-phase-target, --no-stop-on-phase-target`
- `--tolerance TOLERANCE`
- `--amp, --no-amp`
- `--phase-backbone-freeze-epochs PHASE_BACKBONE_FREEZE_EPOCHS`
- `--backbone-lr-mult BACKBONE_LR_MULT`
- `--phase-soft-sigma PHASE_SOFT_SIGMA`
- `--phase-soft-radius PHASE_SOFT_RADIUS`
- `--phase-hard-index-weight PHASE_HARD_INDEX_WEIGHT`
- `--phase-frame-ce-weight PHASE_FRAME_CE_WEIGHT`
- `--phase-frame-radius PHASE_FRAME_RADIUS`
- `--phase-unfreeze-lr-mult PHASE_UNFREEZE_LR_MULT`
- `--weight-decay WEIGHT_DECAY`
- `--max-grad-norm MAX_GRAD_NORM`

### `pipeline/train_stage4_segmentation.py` options

- `-h, --help`
- `--data-dir DATA_DIR`
- `--image-size IMAGE_SIZE`
- `--batch-size BATCH_SIZE`
- `--epochs EPOCHS`
- `--learning-rate LEARNING_RATE`
- `--weight-decay WEIGHT_DECAY`
- `--workers WORKERS`
- `--max-videos MAX_VIDEOS`
- `--dice-weight DICE_WEIGHT`
- `--eval-threshold EVAL_THRESHOLD`
- `--pos-weight POS_WEIGHT`
- `--pos-weight-max POS_WEIGHT_MAX`
- `--amp, --no-amp`
- `--checkpoint CHECKPOINT`
- `--init-checkpoint INIT_CHECKPOINT`
- `--output-dir OUTPUT_DIR`
- `--device DEVICE`
- `--patience PATIENCE`
- `--seed SEED`
- `--model-name MODEL_NAME`
- `--pretrained, --no-pretrained`
- `--base-channels BASE_CHANNELS`
- `--optimizer {sgd,adamw}`
- `--lr-step-period LR_STEP_PERIOD`
- `--lr-gamma LR_GAMMA`
- `--normalize {auto,none,imagenet}`

### `pipeline/run_stage45_from_tracings.py` options

- `-h, --help`
- `--split SPLIT`
- `--max-videos MAX_VIDEOS`
- `--save-overlays`
- `--output-dir OUTPUT_DIR`
- `--stage4-checkpoint STAGE4_CHECKPOINT`
- `--stage4-model-name STAGE4_MODEL_NAME`
- `--stage4-base-channels STAGE4_BASE_CHANNELS`
- `--eval-threshold EVAL_THRESHOLD`
- `--curve-smooth-window CURVE_SMOOTH_WINDOW`
- `--curve-batch-size CURVE_BATCH_SIZE`
- `--device DEVICE`

### `pipeline/train_stage67_similarity.py` options

- `-h, --help`
- `--data-dir DATA_DIR`
- `--stage123-checkpoint STAGE123_CHECKPOINT`
- `--num-frames NUM_FRAMES`
- `--max-videos MAX_VIDEOS`
- `--device DEVICE`
- `--normal-threshold NORMAL_THRESHOLD`
- `--severe-threshold SEVERE_THRESHOLD`
- `--output-dir OUTPUT_DIR`
- `--stage5-metrics-dir STAGE5_METRICS_DIR`
- `--stage5-video-metrics-name STAGE5_VIDEO_METRICS_NAME`
- `--save-per-split-csv, --no-save-per-split-csv`

### `pipeline/train_all_stages.py` options

- `-h, --help`
- `--data-dir DATA_DIR`
- `--skip-stage123`
- `--skip-stage4`
- `--skip-stage5`
- `--skip-stage67`
- `--stage123-checkpoint STAGE123_CHECKPOINT`
- `--train-phase-detector`
- `--phase-checkpoint PHASE_CHECKPOINT`
- `--phase-init-checkpoint PHASE_INIT_CHECKPOINT`
- `--phase-epochs PHASE_EPOCHS`
- `--phase-learning-rate PHASE_LEARNING_RATE`
- `--phase-batch-size PHASE_BATCH_SIZE`
- `--phase-num-frames PHASE_NUM_FRAMES`
- `--phase-train-sampling-mode {global,echonet,phase_window}`
- `--phase-eval-sampling-mode {global,echonet,phase_window}`
- `--phase-workers PHASE_WORKERS`
- `--phase-max-videos PHASE_MAX_VIDEOS`
- `--phase-target-accuracy PHASE_TARGET_ACCURACY`
- `--phase-stop-on-target, --no-phase-stop-on-target`
- `--phase-tolerance PHASE_TOLERANCE`
- `--stage123-epochs STAGE123_EPOCHS`
- `--stage123-learning-rate STAGE123_LEARNING_RATE`
- `--stage123-batch-size STAGE123_BATCH_SIZE`
- `--stage123-num-frames STAGE123_NUM_FRAMES`
- `--stage123-workers STAGE123_WORKERS`
- `--stage123-max-videos STAGE123_MAX_VIDEOS`
- `--stage4-checkpoint STAGE4_CHECKPOINT`
- `--stage4-epochs STAGE4_EPOCHS`
- `--stage4-learning-rate STAGE4_LEARNING_RATE`
- `--stage4-batch-size STAGE4_BATCH_SIZE`
- `--stage4-workers STAGE4_WORKERS`
- `--stage4-image-size STAGE4_IMAGE_SIZE`
- `--stage4-max-videos STAGE4_MAX_VIDEOS`
- `--stage4-model-name STAGE4_MODEL_NAME`
- `--stage4-pretrained, --no-stage4-pretrained`
- `--stage4-optimizer {sgd,adamw}`
- `--stage5-max-videos STAGE5_MAX_VIDEOS`
- `--stage5-save-overlays`
- `--stage67-output-dir STAGE67_OUTPUT_DIR`
- `--stage67-stage5-metrics-dir STAGE67_STAGE5_METRICS_DIR`
- `--stage67-max-videos STAGE67_MAX_VIDEOS`
- `--stage67-normal-threshold STAGE67_NORMAL_THRESHOLD`
- `--stage67-severe-threshold STAGE67_SEVERE_THRESHOLD`
- `--device DEVICE`

### `visualization/validate_phase_detection.py` options

- `-h, --help`
- `--split SPLIT`
- `--checkpoint CHECKPOINT`
- `--batch-size BATCH_SIZE`
- `--num-samples NUM_SAMPLES`
- `--output-dir OUTPUT_DIR`

### `validation/validate_ed_es_against_reference.py` options

- `-h, --help`
- `--data-dir DATA_DIR`
- `--split SPLIT`
- `--max-videos MAX_VIDEOS`
- `--detector {curve,global_extrema}`
- `--smooth-window SMOOTH_WINDOW`
- `--enforce-es-after-ed, --no-enforce-es-after-ed`
- `--reference REFERENCE`
- `--reference-mode REFERENCE_MODE`
- `--sheet SHEET`
- `--file-col FILE_COL`
- `--ed-col ED_COL`
- `--es-col ES_COL`
- `--tolerance TOLERANCE`
- `--output OUTPUT`
- `--mismatch-output MISMATCH_OUTPUT`

### `validation/generate_reference_frame_template.py` options

- `-h, --help`
- `--data-dir DATA_DIR`
- `--split SPLIT`
- `--max-videos MAX_VIDEOS`
- `--include-meta, --no-include-meta`
- `--prefill-from-tracings, --no-prefill-from-tracings`
- `--prefill-method {global_extrema,curve}`
- `--prefill-smooth-window PREFILL_SMOOTH_WINDOW`
- `--enforce-es-after-ed, --no-enforce-es-after-ed`
- `--output OUTPUT`

## 7. Notes

- Stage 4 tracing-to-mask logic is aligned with EchoNet contour construction from `VolumeTracings.csv`.
- Stage 4 area validation is exported at both frame and video level.
- For reproducibility, keep one config profile per experiment and log CLI overrides used for that run.

## 7. Update: Stage5 Predicted Masks + Stage6 MLP Backend

### Stage5 from learned Stage4 masks

Use this after Stage4 training to evaluate EF using **predicted segmentation masks**:

```powershell
python pipeline\run_stage45_from_tracings.py --split VAL --stage4-checkpoint best_stage4_segmentation.pth --eval-threshold 0.5 --max-videos 25 --save-overlays
```

Run it for `TRAIN`, `VAL`, and `TEST` before Stage6/7 so the calibrator uses the same predicted-mask features that deployment will have.

### Stage6 backend options

Prototype backend (default):

```powershell
python pipeline\train_stage67_similarity.py --stage123-checkpoint best_model.pth --num-frames 32 --stage5-metrics-dir validation/outputs/stage45 --stage6-backend similarity
```

Trainable MLP backend (creates `.pth` artifact):

```powershell
python pipeline\train_stage67_similarity.py --stage123-checkpoint best_model.pth --num-frames 32 --stage5-metrics-dir validation/outputs/stage45 --stage6-backend mlp --stage6-mlp-epochs 80 --stage6-mlp-hidden-dim 64
```

Artifacts:
- similarity backend: `validation/outputs/stage67/stage6_similarity_engine.npz`
- MLP backend: `validation/outputs/stage67/stage6_mlp_model.pth`
- calibration: `validation/outputs/stage67/stage7_calibration.json`

### Orchestrator flags

`train_all_stages.py` now supports:
- `--stage5-stage4-checkpoint ...`
- `--stage67-stage5-metrics-dir ...`
- `--stage67-backend similarity|mlp`
- `--stage67-mlp-*` training flags
