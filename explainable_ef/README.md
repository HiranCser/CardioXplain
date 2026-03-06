# Explainable EF (CardioXplain)

This repository trains and evaluates an echocardiography EF model with a stage-based pipeline:

- Stage 1: Feature extraction
- Stage 2: Temporal modeling
- Stage 3: Phase detection (ED/ES frame localization)
- Stage 4: LV segmentation (from tracings)
- Stage 5: EF computation (from ED/ES areas)

## 1. Prerequisites

- Python 3.10+ (3.11/3.12 recommended for broad PyTorch compatibility)
- `pip`
- Enough disk space for model checkpoints and logs

## 2. Setup

From repository root:

```powershell
cd D:\datascience\MTech\Sem4\Project\CardioXplain\cx\explainable_ef
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

Install dependencies:

```powershell
pip install torch torchvision pandas numpy opencv-python matplotlib tqdm
```

If your environment needs a specific PyTorch build (CUDA or CPU), install `torch`/`torchvision` from the official PyTorch selector first, then install the remaining packages.

## 3. Dataset Layout

By default, `config.py` expects data at:

`D:\datascience\MTech\Sem4\Project\CardioXplain\dynamic\a4c-video-dir`

Expected structure:

```text
a4c-video-dir/
  FileList.csv
  VolumeTracings.csv
  Videos/
    *.avi
```

If your dataset is elsewhere, update `DATA_DIR` in `config.py`.

## 4. Core Training / Evaluation (Stage 1-3)

Run end-to-end training + validation + test:

```powershell
python model_execution.py
```

Outputs:

- Checkpoint: `best_model.pth`
- Logs: `logs/training_*.log`

## 5. One-Command Smoke Test

Run a tiny end-to-end smoke run (small data + few epochs):

```powershell
python model_execution.py --smoke
```

Smoke defaults:

- `MAX_VIDEOS=24`
- `EPOCHS=2`
- `BATCH_SIZE=4`
- `NUM_FRAMES=16`
- `CHECKPOINT_PATH=best_model_smoke.pth`

You can override smoke defaults explicitly, for example:

```powershell
python model_execution.py --smoke --epochs 3 --max-videos 40 --checkpoint best_model_smoke_v2.pth
```

## 6. Phase Validation + Visualization (Stage 3)

After training, run:

```powershell
python visualization\validate_phase_detection.py --split TEST --checkpoint best_model.pth --num-samples 12
```

This computes:

- ED/ES MAE in frames
- Within-tolerance ED/ES accuracy
- Joint ED+ES tolerance metrics

And saves plots to:

`visualization/outputs/phase_validation`

## 7. Stage 4/5 Pipeline (LV Segmentation + EF from Tracings)

Run Stage 4/5 baseline on a subset:

```powershell
python pipeline\run_stage45_from_tracings.py --split VAL --max-videos 25 --save-overlays
```

This computes:

- ED/ES masks from `VolumeTracings.csv`
- ED/ES cavity areas
- EF proxy from mask areas
- EF proxy MAE summary

And saves overlays to:

`visualization/outputs/stage45`

## 8. Optional: Preprocess Videos

To pre-decode videos for faster loading:

```powershell
python preprocess_videos.py
```

## 9. Project Structure

```text
explainable_ef/
  config.py
  model_execution.py
  models/
    ef_model.py
  data/
    dataset.py
  pipeline/
    stage1_feature_extractor.py
    stage2_temporal_model.py
    stage3_phase_detector.py
    stage45_pipeline.py
    orchestrator.py
    run_stage45_from_tracings.py
  visualization/
    validate_phase_detection.py
    visualize_attention.py
```
