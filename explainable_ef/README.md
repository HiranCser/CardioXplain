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

### A) Combined Stage 1+2+3 (joint EF + phase)

```powershell
python model_execution.py --train-stage123 --no-phase-only
```

This now trains Stage 1/2/3 jointly and keeps EF loss enabled.

`--train-stage123` now also applies phase-friendly defaults unless explicitly overridden: tracing window sampling, higher phase loss weight, and Stage2 attention-alignment loss.

### B) Phase-only Stage 1+2+3

```powershell
python model_execution.py --phase-only
```

In this mode, EF loss is intentionally set to zero.

### C) Default Stage 1+2+3 training

```powershell
python model_execution.py
```

### D) Smoke test (tiny run)

```powershell
python model_execution.py --smoke
```

### E) Stage 3 validation + visualization

```powershell
python visualization\validate_phase_detection.py --split TEST --checkpoint best_model.pth --num-samples 12
```

### F) ED/ES validation against reference annotations

Use your expert frame table (CSV/XLSX), or use `VolumeTracings.csv` directly via `--reference-mode volume_tracings`.

```powershell
python validation\validate_ed_es_against_reference.py --split VAL --reference "D:\datascience\MTech\Sem4\Project\CardioXplain\dynamic\a4c-video-dir\VolumeTracings.csv" --reference-mode volume_tracings --tolerance 1 --output validation\outputs\ed_es_validation.csv --mismatch-output validation\outputs\ed_es_mismatches.csv
```

### G) Generate annotation template for manual ED/ES review

```powershell
python validation\generate_reference_frame_template.py --split VAL --include-meta --prefill-from-tracings --output validation\outputs\ed_es_template_val.csv
```

### H) Stage 4 full training + area validation (per frame + per video)

Recommended:

```powershell
python pipeline\train_stage4_segmentation.py --data-dir "D:\datascience\MTech\Sem4\Project\CardioXplain\dynamic\a4c-video-dir" --model-name deeplabv3_resnet50 --pretrained --batch-size 20 --epochs 50 --learning-rate 1e-5 --optimizer sgd --workers 8 --image-size 112
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

### I) Stage 4/5 tracing baseline utility

```powershell
python pipeline\run_stage45_from_tracings.py --split VAL --max-videos 25 --save-overlays
```

### J) One-page UI for Stage 1-5 outputs (recommended for review)

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
- `--max-videos MAX_VIDEOS`
- `--epochs EPOCHS`
- `--learning-rate, --lr LEARNING_RATE`
- `--batch-size BATCH_SIZE`
- `--num-frames NUM_FRAMES`
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
- `--phase-temporal-window-mode {full,tracing}`
- `--phase-temporal-window-margin-mult PHASE_TEMPORAL_WINDOW_MARGIN_MULT`
- `--phase-temporal-window-jitter-mult PHASE_TEMPORAL_WINDOW_JITTER_MULT`
- `--warm-start-checkpoint, --no-warm-start-checkpoint`
- `--protect-best-checkpoint, --no-protect-best-checkpoint`
- `--train-stage123, --no-train-stage123`

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
- `--amp, --no-amp`
- `--checkpoint CHECKPOINT`
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
