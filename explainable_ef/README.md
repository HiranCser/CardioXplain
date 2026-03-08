# Explainable EF (CardioXplain)

This repository trains and evaluates an echocardiography EF pipeline with stage-based modules:

- Stage 1: Feature extraction
- Stage 2: Temporal modeling
- Stage 3: Phase detection (ED/ES frame localization)
- Stage 4: LV segmentation (trainable model)
- Stage 5: EF computation (from ED/ES areas)

## 1. Prerequisites

- Python 3.10+
- `pip`
- NVIDIA GPU (recommended for full training)

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

Required structure:

```text
a4c-video-dir/
  FileList.csv
  VolumeTracings.csv
  Videos/
    *.avi
```

## 4. Stage 1-3 Training (Feature + Temporal + Phase)

```powershell
python model_execution.py
```

Smoke run:

```powershell
python model_execution.py --smoke
```

## 5. Stage 3 Validation (ED/ES)

```powershell
python visualization\validate_phase_detection.py --split TEST --checkpoint best_model.pth --num-samples 12
```

## 6. Stage 4 Full Training + Area Validation

Stage 4 now supports EchoNet-style segmentation backbones (`deeplabv3_resnet50`, `fcn_resnet50`, etc.) and validates predicted mask area against `VolumeTracings.csv` for each traced frame and each video.

Recommended full run:

```powershell
python pipeline\train_stage4_segmentation.py --data-dir "D:\datascience\MTech\Sem4\Project\CardioXplain\dynamic\a4c-video-dir" --model-name deeplabv3_resnet50 --pretrained --batch-size 20 --epochs 50 --learning-rate 1e-5 --optimizer sgd --workers 8 --image-size 112
```

CPU/smoke run:

```powershell
python pipeline\train_stage4_segmentation.py --model-name unet --batch-size 4 --epochs 2 --max-videos 24 --no-amp
```

Outputs:

- Best checkpoint: `best_stage4_segmentation.pth`
- Validation frame-level areas: `validation/outputs/stage4/val_best_frame_areas.csv`
- Validation video summary: `validation/outputs/stage4/val_best_frame_areas_video_summary.csv`
- Test frame-level areas: `validation/outputs/stage4/test_frame_areas.csv`
- Test video summary: `validation/outputs/stage4/test_frame_areas_video_summary.csv`

CSV fields include `gt_area`, `pred_area`, `abs_error`, `pct_error`.

## 7. Stage 4/5 Tracing Baseline Utility

```powershell
python pipeline\run_stage45_from_tracings.py --split VAL --max-videos 25 --save-overlays
```

## 8. Notes

- Stage 4 tracing-to-mask logic is aligned with EchoNet contour construction from `VolumeTracings.csv`.
- Video-level area validation is aggregated by `file_name` and exported automatically.

