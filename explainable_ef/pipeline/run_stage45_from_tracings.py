import argparse
import os
import sys

import cv2
import numpy as np
import pandas as pd
import torch

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import config
from models.stage4_segmentation_model import build_stage4_segmentation_model
from pipeline.stage45_pipeline import Stage45Pipeline


def _safe_checkpoint_state_dict(checkpoint_obj):
    if isinstance(checkpoint_obj, dict) and "model_state_dict" in checkpoint_obj:
        return checkpoint_obj["model_state_dict"], checkpoint_obj
    return checkpoint_obj, {}


def _resolve_device(device):
    dev = str(device).lower()
    if dev == "cuda" and not torch.cuda.is_available():
        return "cpu"
    if dev == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return dev


def _normalize_stage4_input(image_tensor, normalize_mode, pretrained_flag=False):
    mode = str(normalize_mode).lower()
    if mode == "auto":
        mode = "imagenet" if bool(pretrained_flag) else "none"
    if mode == "imagenet":
        mean = torch.tensor([0.485, 0.456, 0.406], dtype=image_tensor.dtype).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], dtype=image_tensor.dtype).view(3, 1, 1)
        image_tensor = (image_tensor - mean) / std
    return image_tensor


def _load_stage4_model(checkpoint_path, fallback_model_name, fallback_base_channels, device):
    if not checkpoint_path or not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Stage4 checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict, checkpoint_dict = _safe_checkpoint_state_dict(checkpoint)

    args = checkpoint_dict.get("args", {}) if isinstance(checkpoint_dict, dict) else {}
    model_name = str(args.get("model_name", fallback_model_name))
    base_channels = int(args.get("base_channels", fallback_base_channels))

    model = build_stage4_segmentation_model(
        model_name=model_name,
        pretrained=False,
        in_channels=3,
        base_channels=base_channels,
    ).to(device)
    model.load_state_dict(state_dict, strict=False)
    model.eval()

    metadata = {
        "model_name": model_name,
        "image_size": int(args.get("image_size", 112)),
        "normalize": str(args.get("normalize", "none")),
        "pretrained": bool(args.get("pretrained", False)),
    }
    return model, metadata


def _predict_mask_area_stage4(model, frame_bgr, image_size, normalize_mode, pretrained_flag, device, eval_threshold):
    h, w = frame_bgr.shape[:2]
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(frame_rgb, (image_size, image_size), interpolation=cv2.INTER_LINEAR)

    image_t = torch.from_numpy(resized).permute(2, 0, 1).float() / 255.0
    image_t = _normalize_stage4_input(image_t, normalize_mode, pretrained_flag=pretrained_flag)

    with torch.no_grad():
        logits = model(image_t.unsqueeze(0).to(device))
        if isinstance(logits, dict):
            logits = logits["out"]
        prob = torch.sigmoid(logits)[0, 0].detach().cpu().numpy()

    mask_small = (prob >= float(eval_threshold)).astype(np.uint8)
    mask_orig = cv2.resize(mask_small, (w, h), interpolation=cv2.INTER_NEAREST)
    return mask_orig, float(mask_orig.sum())


def _predict_video_area_curve_stage4(model, video_path, image_size, normalize_mode, pretrained_flag, device, eval_threshold, batch_size=16):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")

    frame_ids = []
    frame_areas = []
    batch_images = []
    batch_sizes = []
    batch_ids = []
    frame_idx = 0

    def flush_batch():
        nonlocal batch_images, batch_sizes, batch_ids, frame_areas
        if not batch_images:
            return
        image_batch = torch.stack(batch_images, dim=0).to(device)
        with torch.no_grad():
            logits = model(image_batch)
            if isinstance(logits, dict):
                logits = logits["out"]
            probs = torch.sigmoid(logits[:, 0]).detach().cpu().numpy()

        for prob, (h, w), fid in zip(probs, batch_sizes, batch_ids):
            mask_small = (prob >= float(eval_threshold)).astype(np.uint8)
            mask_orig = cv2.resize(mask_small, (w, h), interpolation=cv2.INTER_NEAREST)
            frame_areas.append((int(fid), float(mask_orig.sum())))
        batch_images = []
        batch_sizes = []
        batch_ids = []

    while True:
        ok, frame_bgr = cap.read()
        if not ok or frame_bgr is None:
            break
        h, w = frame_bgr.shape[:2]
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(frame_rgb, (image_size, image_size), interpolation=cv2.INTER_LINEAR)
        image_t = torch.from_numpy(resized).permute(2, 0, 1).float() / 255.0
        image_t = _normalize_stage4_input(image_t, normalize_mode, pretrained_flag=pretrained_flag)
        batch_images.append(image_t)
        batch_sizes.append((h, w))
        batch_ids.append(frame_idx)
        frame_ids.append(frame_idx)
        frame_idx += 1
        if len(batch_images) >= int(max(1, batch_size)):
            flush_batch()

    cap.release()
    flush_batch()

    if not frame_areas:
        return np.zeros(0, dtype=np.int32), np.zeros(0, dtype=np.float64)

    curve_frame_ids = np.array([fid for fid, _ in frame_areas], dtype=np.int32)
    curve_areas = np.array([area for _, area in frame_areas], dtype=np.float64)
    return curve_frame_ids, curve_areas


def read_video_frame(video_path, frame_idx):
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_idx))
    ok, frame = cap.read()
    cap.release()
    if not ok or frame is None:
        raise ValueError(f"Could not read frame {frame_idx} from {video_path}")
    return frame


def _canonicalize_ed_es_pair_safe(ed_frame, ed_area, es_frame, es_area):
    helper = getattr(Stage45Pipeline, "canonicalize_ed_es_pair", None)
    if callable(helper):
        return helper(ed_frame, ed_area, es_frame, es_area)

    ed_frame = int(ed_frame)
    es_frame = int(es_frame)
    ed_area = float(ed_area)
    es_area = float(es_area)
    swapped = np.isfinite(ed_area) and np.isfinite(es_area) and es_area > ed_area
    if swapped:
        ed_frame, es_frame = es_frame, ed_frame
        ed_area, es_area = es_area, ed_area
    return {
        "ed_frame": ed_frame,
        "ed_area": ed_area,
        "es_frame": es_frame,
        "es_area": es_area,
        "swapped": bool(swapped),
    }


def _write_overlay(path, frame_bgr, pred_mask=None, gt_mask=None, text=""):
    vis = frame_bgr.copy()
    if gt_mask is not None:
        vis = Stage45Pipeline.overlay_mask(vis, gt_mask, color=(0, 255, 255), alpha=0.25)
    if pred_mask is not None:
        vis = Stage45Pipeline.overlay_mask(vis, pred_mask, color=(0, 255, 0), alpha=0.35)

    if text:
        cv2.putText(
            vis,
            text,
            (5, 16),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
    cv2.imwrite(path, vis)


def parse_args():
    parser = argparse.ArgumentParser(description="Run Stage 4/5 evaluation from tracings or Stage4 predicted masks.")
    parser.add_argument("--split", type=str, default="VAL", help="Dataset split: TRAIN/VAL/TEST")
    parser.add_argument("--data-dir", type=str, default=config.DATA_DIR)
    parser.add_argument("--max-videos", type=int, default=25, help="Number of videos to process")
    parser.add_argument("--save-overlays", action="store_true", help="Save ED/ES overlays")
    parser.add_argument(
        "--output-dir",
        type=str,
        default=os.path.join("visualization", "outputs", "stage45"),
        help="Directory for outputs",
    )

    parser.add_argument(
        "--mode",
        type=str,
        choices=["tracing", "predicted_masks"],
        default="tracing",
        help="tracing: ED/ES/areas from GT tracings; predicted_masks: ED/ES/areas from Stage4 predicted masks",
    )
    parser.add_argument("--stage4-checkpoint", type=str, default=getattr(config, "STAGE4_CHECKPOINT_PATH", "best_stage4_segmentation_area.pth"))
    parser.add_argument("--stage4-model-name", type=str, default="deeplabv3_resnet50")
    parser.add_argument("--stage4-base-channels", type=int, default=32)
    parser.add_argument("--eval-threshold", type=float, default=0.5)
    parser.add_argument("--curve-smooth-window", type=int, default=11, help="Smoothing window for full-video Stage4 size curve")
    parser.add_argument("--curve-batch-size", type=int, default=16, help="Batch size for full-video Stage4 inference")
    parser.add_argument("--device", type=str, default="auto")
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    split_u = str(args.split).upper()
    stage45 = Stage45Pipeline()

    filelist_path = os.path.join(args.data_dir, "FileList.csv")
    tracing_path = os.path.join(args.data_dir, "VolumeTracings.csv")

    filelist = pd.read_csv(filelist_path)
    tracings = pd.read_csv(tracing_path)

    split_df = filelist[filelist["Split"].astype(str).str.upper() == split_u].copy()
    if args.max_videos > 0:
        split_df = split_df.head(int(args.max_videos))

    device = _resolve_device(args.device)
    model4 = None
    model4_meta = None

    if args.mode == "predicted_masks":
        model4, model4_meta = _load_stage4_model(
            checkpoint_path=args.stage4_checkpoint,
            fallback_model_name=args.stage4_model_name,
            fallback_base_channels=args.stage4_base_channels,
            device=device,
        )

    per_video_rows = []
    per_frame_rows = []
    errors = []

    for _, row in split_df.iterrows():
        fname = str(row["FileName"])
        fname_ext = f"{fname}.avi"
        height = int(row["FrameHeight"])
        width = int(row["FrameWidth"])
        ef_gt = float(row["EF"]) / 100.0

        video_rows = tracings[tracings["FileName"] == fname_ext]
        frame_ids = sorted(int(v) for v in video_rows["Frame"].unique().tolist())
        if len(frame_ids) == 0:
            continue

        video_path = os.path.join(args.data_dir, "Videos", fname_ext)
        if not os.path.exists(video_path):
            print(f"Warning: missing video {video_path}")
            continue

        gt_frame_areas = []
        pred_frame_areas = []
        frame_masks_gt = {}
        frame_masks_pred = {}

        for frame_id in frame_ids:
            fr = video_rows[video_rows["Frame"] == frame_id]
            gt_mask = Stage45Pipeline.tracing_to_mask(fr, height=height, width=width)
            gt_area = Stage45Pipeline.mask_area(gt_mask)
            frame_masks_gt[int(frame_id)] = gt_mask
            gt_frame_areas.append((int(frame_id), float(gt_area)))

            if args.mode == "predicted_masks":
                frame_bgr = read_video_frame(video_path, frame_id)
                pred_mask, pred_area = _predict_mask_area_stage4(
                    model=model4,
                    frame_bgr=frame_bgr,
                    image_size=int(model4_meta["image_size"]),
                    normalize_mode=model4_meta["normalize"],
                    pretrained_flag=bool(model4_meta.get("pretrained", False)),
                    device=device,
                    eval_threshold=float(args.eval_threshold),
                )
                frame_masks_pred[int(frame_id)] = pred_mask
                pred_frame_areas.append((int(frame_id), float(pred_area)))

                abs_err_area = abs(float(pred_area) - float(gt_area))
                pct_err_area = abs_err_area / max(1e-6, float(gt_area))
                per_frame_rows.append(
                    {
                        "file_name": fname,
                        "file_name_ext": fname_ext,
                        "frame_id": int(frame_id),
                        "gt_area": float(gt_area),
                        "pred_area": float(pred_area),
                        "abs_error": float(abs_err_area),
                        "pct_error": float(pct_err_area),
                        "mode": args.mode,
                    }
                )

        gt_ed_frame, gt_ed_area = max(gt_frame_areas, key=lambda x: x[1])
        gt_es_frame, gt_es_area = min(gt_frame_areas, key=lambda x: x[1])
        ef_gt_proxy = Stage45Pipeline.compute_ef_from_areas(gt_ed_area, gt_es_area)

        if args.mode == "predicted_masks":
            curve_frame_ids, curve_areas = _predict_video_area_curve_stage4(
                model=model4,
                video_path=video_path,
                image_size=int(model4_meta["image_size"]),
                normalize_mode=model4_meta["normalize"],
                pretrained_flag=bool(model4_meta.get("pretrained", False)),
                device=device,
                eval_threshold=float(args.eval_threshold),
                batch_size=int(args.curve_batch_size),
            )
            if curve_frame_ids.size == 0:
                print(f"Warning: empty Stage4 area curve for {fname_ext}")
                continue

            detected = stage45.detect_ed_es_from_size_curve(
                frame_ids=curve_frame_ids,
                areas=curve_areas,
                smooth_window=int(args.curve_smooth_window),
                enforce_es_after_ed=True,
            )
            curve_area_lookup = {int(fid): float(area) for fid, area in zip(curve_frame_ids.tolist(), curve_areas.tolist())}
            pred_ed_frame = int(detected["ed_frame"])
            pred_es_frame = int(detected["es_frame"])
            pred_ed_area = float(curve_area_lookup.get(pred_ed_frame, float(np.max(curve_areas))))
            pred_es_area = float(curve_area_lookup.get(pred_es_frame, float(np.min(curve_areas))))
            canonical_pair = _canonicalize_ed_es_pair_safe(
                pred_ed_frame,
                pred_ed_area,
                pred_es_frame,
                pred_es_area,
            )
            pred_ed_frame = int(canonical_pair["ed_frame"])
            pred_es_frame = int(canonical_pair["es_frame"])
            pred_ed_area = float(canonical_pair["ed_area"])
            pred_es_area = float(canonical_pair["es_area"])
            pred_pair_swapped = bool(canonical_pair["swapped"])
            ef_pred = Stage45Pipeline.compute_ef_from_areas(pred_ed_area, pred_es_area)
        else:
            pred_ed_frame, pred_ed_area = gt_ed_frame, gt_ed_area
            pred_es_frame, pred_es_area = gt_es_frame, gt_es_area
            pred_pair_swapped = False
            ef_pred = ef_gt_proxy

        abs_err = abs(float(ef_pred) - float(ef_gt))
        errors.append(abs_err)

        per_video_rows.append(
            {
                "file_name": fname,
                "file_name_ext": fname_ext,
                "split": split_u,
                "mode": args.mode,
                "ef_gt": float(ef_gt),
                "ef_gt_proxy": float(ef_gt_proxy),
                "ef_pred": float(ef_pred),
                "ef_abs_error": float(abs_err),
                "gt_ed_frame": int(gt_ed_frame),
                "gt_es_frame": int(gt_es_frame),
                "pred_ed_frame": int(pred_ed_frame),
                "pred_es_frame": int(pred_es_frame),
                "gt_ed_area": float(gt_ed_area),
                "gt_es_area": float(gt_es_area),
                "pred_ed_area": float(pred_ed_area),
                "pred_es_area": float(pred_es_area),
                "pred_ed_frame_error": float(abs(pred_ed_frame - gt_ed_frame)),
                "pred_es_frame_error": float(abs(pred_es_frame - gt_es_frame)),
                "pred_curve_method": ("full_video_stage4_curve" if args.mode == "predicted_masks" else "tracing"),
                "pred_pair_swapped": bool(pred_pair_swapped),
            }
        )

        if args.save_overlays:
            try:
                ed_img = cv2.resize(read_video_frame(video_path, pred_ed_frame), (width, height))
                es_img = cv2.resize(read_video_frame(video_path, pred_es_frame), (width, height))

                ed_gt_mask = frame_masks_gt.get(int(pred_ed_frame))
                es_gt_mask = frame_masks_gt.get(int(pred_es_frame))
                if args.mode == "predicted_masks" and int(pred_ed_frame) not in frame_masks_pred:
                    pred_mask_ed, _ = _predict_mask_area_stage4(
                        model=model4,
                        frame_bgr=read_video_frame(video_path, pred_ed_frame),
                        image_size=int(model4_meta["image_size"]),
                        normalize_mode=model4_meta["normalize"],
                        pretrained_flag=bool(model4_meta.get("pretrained", False)),
                        device=device,
                        eval_threshold=float(args.eval_threshold),
                    )
                    frame_masks_pred[int(pred_ed_frame)] = pred_mask_ed
                if args.mode == "predicted_masks" and int(pred_es_frame) not in frame_masks_pred:
                    pred_mask_es, _ = _predict_mask_area_stage4(
                        model=model4,
                        frame_bgr=read_video_frame(video_path, pred_es_frame),
                        image_size=int(model4_meta["image_size"]),
                        normalize_mode=model4_meta["normalize"],
                        pretrained_flag=bool(model4_meta.get("pretrained", False)),
                        device=device,
                        eval_threshold=float(args.eval_threshold),
                    )
                    frame_masks_pred[int(pred_es_frame)] = pred_mask_es
                ed_pred_mask = frame_masks_pred.get(int(pred_ed_frame)) if args.mode == "predicted_masks" else ed_gt_mask
                es_pred_mask = frame_masks_pred.get(int(pred_es_frame)) if args.mode == "predicted_masks" else es_gt_mask

                _write_overlay(
                    os.path.join(args.output_dir, f"{fname}_ed_overlay.png"),
                    frame_bgr=ed_img,
                    pred_mask=ed_pred_mask,
                    gt_mask=ed_gt_mask,
                    text=f"ED frame={pred_ed_frame} predA={pred_ed_area:.1f} gtA={gt_ed_area:.1f}",
                )
                _write_overlay(
                    os.path.join(args.output_dir, f"{fname}_es_overlay.png"),
                    frame_bgr=es_img,
                    pred_mask=es_pred_mask,
                    gt_mask=es_gt_mask,
                    text=f"ES frame={pred_es_frame} predA={pred_es_area:.1f} gtA={gt_es_area:.1f}",
                )
            except Exception as exc:
                print(f"Warning: overlay generation failed for {fname_ext}: {exc}")

    if not per_video_rows:
        print("No videos were processed.")
        return

    mae = float(np.mean(errors))

    video_csv = os.path.join(args.output_dir, "stage5_video_metrics.csv")
    frame_csv = os.path.join(args.output_dir, "stage5_frame_metrics.csv")

    pd.DataFrame(per_video_rows).to_csv(video_csv, index=False)
    if per_frame_rows:
        pd.DataFrame(per_frame_rows).to_csv(frame_csv, index=False)

    print("=" * 88)
    print(f"STAGE 5 SUMMARY ({args.mode.upper()})")
    print("=" * 88)
    print(f"Split:                 {split_u}")
    print(f"Mode:                  {args.mode}")
    print(f"Videos processed:      {len(per_video_rows)}")
    print(f"EF MAE (0-1):          {mae:.4f}")
    print(f"EF MAE (%):            {mae * 100:.2f}")
    print(f"Video metrics CSV:     {os.path.abspath(video_csv)}")
    if per_frame_rows:
        print(f"Frame metrics CSV:     {os.path.abspath(frame_csv)}")
    print(f"Output dir:            {os.path.abspath(args.output_dir)}")
    print("=" * 88)


if __name__ == "__main__":
    main()
