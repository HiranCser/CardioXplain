import os
import io
import sys
import json
import math
import base64
import shutil
import hashlib
import subprocess

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
import torch

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import config
from data.dataset import EchoDataset
from models.ef_model import EFModel
from models.stage4_segmentation_model import build_stage4_segmentation_model
from pipeline.stage3_phase_detector import Stage3PhaseDetector
from pipeline.stage45_pipeline import Stage45Pipeline


st.set_page_config(page_title="CardioXplain Stage Dashboard", layout="wide")
PREVIEW_MAX_WIDTH = 440
PREVIEW_DISPLAY_WIDTH = 440
FRAME_DISPLAY_WIDTH = 440


def _render_centered_image(image_data, caption, width=PREVIEW_DISPLAY_WIDTH):
    _left_col, preview_col, _right_col = st.columns([1.6, 2.0, 1.6])
    with preview_col:
        st.image(image_data, caption=caption, width=width)


def _inject_page_styles():
    st.markdown(
        """
        <style>
            [data-testid="stAppViewContainer"] {
                background:
                    radial-gradient(circle at top left, rgba(30, 136, 229, 0.07), transparent 26%),
                    radial-gradient(circle at top right, rgba(14, 165, 140, 0.05), transparent 22%),
                    linear-gradient(180deg, #f6faff 0%, #f1f6fb 46%, #f8fbfe 100%);
            }
            .block-container {
                padding-top: 1.2rem;
                padding-bottom: 2.8rem;
                max-width: 1320px;
            }
            [data-testid="stSidebar"] {
                background: linear-gradient(180deg, #f8fbfe 0%, #eef4fa 100%);
                border-right: 1px solid rgba(148, 163, 184, 0.22);
            }
            h1, h2, h3 {
                color: #12324a;
                letter-spacing: -0.02em;
            }
            h1 {
                font-size: 2.1rem;
                font-weight: 800;
            }
            h2, h3 {
                font-weight: 700;
            }
            [data-testid="stMetric"] {
                background: rgba(255, 255, 255, 0.94);
                border: 1px solid rgba(148, 163, 184, 0.18);
                border-radius: 16px;
                padding: 0.8rem 0.95rem;
                box-shadow: 0 8px 18px rgba(15, 23, 42, 0.04);
            }
            [data-testid="stMetricLabel"] {
                color: #6a8398;
                font-weight: 700;
                letter-spacing: 0.04em;
                text-transform: uppercase;
            }
            [data-testid="stMetricValue"] {
                color: #0f2940;
                font-weight: 800;
                line-height: 1.05;
                white-space: normal !important;
                overflow: visible !important;
                text-overflow: clip !important;
                word-break: break-word;
                overflow-wrap: anywhere;
                font-size: clamp(1.65rem, 2.3vw, 2.35rem);
            }
            [data-testid="stMetricValue"] > div,
            [data-testid="stMetricLabel"] > div {
                white-space: normal !important;
                overflow: visible !important;
                text-overflow: clip !important;
                word-break: break-word;
                overflow-wrap: anywhere;
            }
            div[data-testid="stDataFrame"] {
                border-radius: 18px;
                overflow: hidden;
                border: 1px solid rgba(148, 163, 184, 0.18);
                box-shadow: 0 10px 20px rgba(15, 23, 42, 0.04);
                background: rgba(255, 255, 255, 0.94);
            }
            [data-testid="stImage"] img {
                border-radius: 16px;
                box-shadow: 0 10px 22px rgba(15, 23, 42, 0.08);
            }
            [data-testid="stCaptionContainer"] {
                color: #688197;
                font-size: 0.83rem;
            }
            .section-heading {
                margin: 1.2rem 0 0.35rem 0;
            }
            .section-title {
                color: #12324a;
                font-size: 1.08rem;
                font-weight: 800;
                letter-spacing: -0.01em;
            }
            .section-subtitle {
                color: #627d91;
                font-size: 0.9rem;
                line-height: 1.5;
                margin-top: 0.12rem;
            }
            .media-shell {
                background: rgba(255, 255, 255, 0.92);
                border: 1px solid rgba(148, 163, 184, 0.18);
                border-radius: 20px;
                padding: 0.95rem 1rem 0.8rem 1rem;
                box-shadow: 0 10px 22px rgba(15, 23, 42, 0.05);
                margin-bottom: 0.85rem;
            }
            .media-shell .shell-title {
                color: #12324a;
                font-size: 1rem;
                font-weight: 800;
                margin-bottom: 0.2rem;
            }
            .media-shell .shell-subtitle {
                color: #647d91;
                font-size: 0.86rem;
                line-height: 1.45;
                margin-bottom: 0.75rem;
            }
            .phase-group {
                background: rgba(255, 255, 255, 0.9);
                border: 1px solid rgba(148, 163, 184, 0.16);
                border-radius: 22px;
                padding: 1rem 1rem 0.9rem 1rem;
                box-shadow: 0 10px 22px rgba(15, 23, 42, 0.04);
                margin-bottom: 0.95rem;
            }
            .phase-group-title {
                color: #173652;
                font-size: 1rem;
                font-weight: 800;
                margin-bottom: 0.2rem;
            }
            .phase-group-note {
                color: #688197;
                font-size: 0.83rem;
                line-height: 1.45;
                margin-bottom: 0.85rem;
            }
            .phase-card-head {
                display: flex;
                align-items: center;
                justify-content: space-between;
                gap: 0.75rem;
                margin-bottom: 0.55rem;
            }
            .phase-card-title {
                color: #18344d;
                font-size: 1rem;
                font-weight: 800;
                line-height: 1.3;
            }
            .phase-pill {
                border-radius: 999px;
                padding: 0.24rem 0.62rem;
                font-size: 0.72rem;
                font-weight: 800;
                letter-spacing: 0.04em;
                white-space: nowrap;
            }
            .phase-pill-ed {
                background: rgba(22, 163, 74, 0.12);
                color: #177245;
            }
            .phase-pill-es {
                background: rgba(220, 38, 38, 0.12);
                color: #b42318;
            }
            .phase-card-meta {
                display: flex;
                justify-content: space-between;
                gap: 0.75rem;
                color: #6b8297;
                font-size: 0.82rem;
                margin-top: 0.45rem;
                padding: 0 0.15rem 0.1rem 0.15rem;
            }
            .phase-frame-caption {
                color: #688197;
                font-size: 0.82rem;
                margin-top: 0.35rem;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _abs_path(path_value):
    if not path_value:
        return ""
    if os.path.isabs(path_value):
        return path_value
    return os.path.abspath(os.path.join(ROOT_DIR, path_value))


def _resolve_device(device_choice):
    if device_choice == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if device_choice == "cuda" and not torch.cuda.is_available():
        return "cpu"
    return device_choice


def _safe_checkpoint_state_dict(checkpoint_obj):
    if isinstance(checkpoint_obj, dict) and "model_state_dict" in checkpoint_obj:
        return checkpoint_obj["model_state_dict"], checkpoint_obj
    return checkpoint_obj, {}


def _attention_entropy(attn):
    if attn.size <= 1:
        return 0.0
    attn = np.clip(attn.astype(np.float64), 1e-12, 1.0)
    return float(-(attn * np.log(attn)).sum() / math.log(attn.shape[0]))


def _overlay_mask_rgb(frame_rgb, mask, color=(0, 255, 0), alpha=0.35):
    out = frame_rgb.copy()
    out[mask > 0] = np.array(color, dtype=np.uint8)
    blended = (alpha * out + (1.0 - alpha) * frame_rgb).astype(np.uint8)
    return blended


def _dice(mask_a, mask_b, eps=1e-6):
    if mask_a is None or mask_b is None:
        return float("nan")
    a = (mask_a > 0).astype(np.uint8)
    b = (mask_b > 0).astype(np.uint8)
    inter = float((a & b).sum())
    denom = float(a.sum() + b.sum())
    return (2.0 * inter + eps) / (denom + eps)


def _read_video_frames_rgb(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Failed to open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    frames = []
    while True:
        ok, frame_bgr = cap.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
    cap.release()

    if len(frames) == 0:
        raise ValueError(f"No frames found in video: {video_path}")

    return frames, float(fps if fps and fps > 0 else 50.0)


def _video_cache_dir():
    cache_dir = os.path.join(ROOT_DIR, "ui", ".video_cache")
    os.makedirs(cache_dir, exist_ok=True)
    return cache_dir


def _guess_video_mime(video_path):
    ext = os.path.splitext(str(video_path))[1].lower()
    if ext == ".mp4" or ext == ".m4v" or ext == ".mov":
        return "video/mp4"
    if ext == ".webm":
        return "video/webm"
    if ext in {".ogg", ".ogv"}:
        return "video/ogg"
    return "video/mp4"


def _ffmpeg_h264_transcode(input_path, output_path):
    ffmpeg_bin = shutil.which("ffmpeg")
    if not ffmpeg_bin:
        return False, "ffmpeg not found"

    cmd = [
        ffmpeg_bin,
        "-y",
        "-i",
        input_path,
        "-an",
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        output_path,
    ]

    try:
        proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
        if proc.returncode != 0:
            err = proc.stderr.decode("utf-8", errors="ignore").strip().splitlines()
            tail = err[-1] if err else "ffmpeg failed"
            return False, tail
        if not os.path.exists(output_path) or os.path.getsize(output_path) == 0:
            return False, "ffmpeg produced empty output"
        return True, ""
    except Exception as exc:
        return False, str(exc)


@st.cache_data(show_spinner=False)
def _prepare_browser_video(video_path):
    """
    Return (playable_path, was_converted, error_message, mime_type).

    Strategy:
    1) Use native web formats directly when possible.
    2) For non-web formats (or problematic files), transcode to cached MP4.
       Prefer ffmpeg/H.264 when available; fallback to OpenCV mp4v.
    """
    video_path = os.path.abspath(video_path)
    if not os.path.exists(video_path):
        return "", False, f"Video not found: {video_path}", "video/mp4"

    ext = os.path.splitext(video_path)[1].lower()
    if ext in {".mp4", ".webm", ".ogg", ".ogv", ".m4v", ".mov"}:
        return video_path, False, "", _guess_video_mime(video_path)

    try:
        stat = os.stat(video_path)
        cache_key = hashlib.md5(
            f"{video_path}|{stat.st_mtime_ns}|{stat.st_size}".encode("utf-8")
        ).hexdigest()[:16]
        output_path = os.path.join(_video_cache_dir(), f"{cache_key}.mp4")

        if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
            return output_path, True, "", "video/mp4"

        ok_ffmpeg, ffmpeg_err = _ffmpeg_h264_transcode(video_path, output_path)
        if ok_ffmpeg:
            return output_path, True, "", "video/mp4"

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return "", False, f"Failed to open source video for conversion ({ffmpeg_err}).", "video/mp4"

        src_fps = cap.get(cv2.CAP_PROP_FPS)
        fps = float(src_fps if src_fps and src_fps > 0 else 25.0)
        fps = max(10.0, min(30.0, fps))

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if width <= 0 or height <= 0:
            cap.release()
            return "", False, "Invalid source video size; cannot convert.", "video/mp4"

        max_w = 960
        if width > max_w:
            scale = max_w / float(width)
            out_w = int(round(width * scale))
            out_h = int(round(height * scale))
        else:
            out_w, out_h = width, height

        out_w = max(2, out_w - (out_w % 2))
        out_h = max(2, out_h - (out_h % 2))

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(output_path, fourcc, fps, (out_w, out_h))
        if not writer.isOpened():
            cap.release()
            return "", False, f"Failed to initialize MP4 writer (ffmpeg={ffmpeg_err}).", "video/mp4"

        frame_count = 0
        while True:
            ok, frame_bgr = cap.read()
            if not ok:
                break
            if frame_bgr.shape[1] != out_w or frame_bgr.shape[0] != out_h:
                frame_bgr = cv2.resize(frame_bgr, (out_w, out_h), interpolation=cv2.INTER_AREA)
            writer.write(frame_bgr)
            frame_count += 1

        writer.release()
        cap.release()

        if frame_count == 0 or not os.path.exists(output_path) or os.path.getsize(output_path) == 0:
            return "", False, "Video conversion produced an empty output.", "video/mp4"

        return output_path, True, "", "video/mp4"
    except Exception as exc:
        return "", False, f"Video conversion failed: {exc}", "video/mp4"


@st.cache_data(show_spinner=False)
def _prepare_gif_preview(video_path, max_frames=120, max_width=PREVIEW_MAX_WIDTH):
    """Build an animated GIF fallback preview for browsers with video codec issues."""
    try:
        from PIL import Image
    except Exception:
        return b"", "Pillow is not available for GIF fallback."

    video_path = os.path.abspath(video_path)
    if not os.path.exists(video_path):
        return b"", f"Video not found: {video_path}"

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return b"", "Failed to open source video for GIF preview."

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    fps = float(fps if fps and fps > 0 else 20.0)

    stride = max(1, int(total / max_frames)) if total > 0 else 2
    duration_ms = int(round(1000.0 / max(1.0, min(12.0, fps / stride))))

    pil_frames = []
    idx = 0
    while True:
        ok, frame_bgr = cap.read()
        if not ok:
            break
        if idx % stride != 0:
            idx += 1
            continue

        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        h, w = frame_rgb.shape[:2]
        if w > max_width:
            scale = max_width / float(w)
            frame_rgb = cv2.resize(frame_rgb, (int(round(w * scale)), int(round(h * scale))), interpolation=cv2.INTER_AREA)

        pil_frames.append(Image.fromarray(frame_rgb))
        if len(pil_frames) >= max_frames:
            break
        idx += 1

    cap.release()

    if not pil_frames:
        return b"", "Could not extract frames for GIF preview."

    buff = io.BytesIO()
    pil_frames[0].save(
        buff,
        format="GIF",
        save_all=True,
        append_images=pil_frames[1:],
        duration=max(40, duration_ms),
        loop=0,
        optimize=False,
    )
    return buff.getvalue(), ""


def _prepare_segmentation_gif(full_frames, model4, meta4, device, max_frames=80, max_width=PREVIEW_MAX_WIDTH):
    try:
        from PIL import Image
    except Exception:
        return b"", "Pillow is not available for segmentation GIF preview."

    if not full_frames:
        return b"", "No frames available for segmentation preview."

    stride = max(1, int(math.ceil(len(full_frames) / float(max_frames))))
    pil_frames = []

    for idx, frame_rgb in enumerate(full_frames):
        if idx % stride != 0:
            continue

        mask, _ = _predict_mask_stage4(
            model4,
            frame_rgb,
            image_size=int(meta4["image_size"]),
            normalize_mode=meta4["normalize"],
            pretrained_flag=bool(meta4.get("pretrained", False)),
            device=device,
        )
        overlay = _overlay_mask_rgb(frame_rgb, mask, color=(0, 255, 0), alpha=0.35)

        h, w = overlay.shape[:2]
        if w > max_width:
            scale = max_width / float(w)
            overlay = cv2.resize(overlay, (int(round(w * scale)), int(round(h * scale))), interpolation=cv2.INTER_AREA)

        pil_frames.append(Image.fromarray(overlay))
        if len(pil_frames) >= max_frames:
            break

    if not pil_frames:
        return b"", "Could not build segmentation GIF preview."

    buff = io.BytesIO()
    pil_frames[0].save(
        buff,
        format="GIF",
        save_all=True,
        append_images=pil_frames[1:],
        duration=80,
        loop=0,
        optimize=False,
    )
    return buff.getvalue(), ""


@st.cache_data(show_spinner=False)
def load_split_filelist(data_dir, split):
    filelist_path = os.path.join(data_dir, "FileList.csv")
    if not os.path.exists(filelist_path):
        raise FileNotFoundError(f"Missing FileList.csv at {filelist_path}")

    filelist = pd.read_csv(filelist_path)
    split_u = str(split).upper()
    filelist["Split"] = filelist["Split"].astype(str).str.upper()
    filelist = filelist[filelist["Split"] == split_u].copy()
    filelist = filelist.sort_values("FileName").reset_index(drop=True)
    return filelist


@st.cache_data(show_spinner=False)
def load_volume_tracings(data_dir):
    path = os.path.join(data_dir, "VolumeTracings.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing VolumeTracings.csv at {path}")
    return pd.read_csv(path)


@st.cache_data(show_spinner=False)
def load_stage67_predictions(output_dir, split):
    csv_path = os.path.join(output_dir, f"stage67_{str(split).lower()}_predictions.csv")
    if not os.path.exists(csv_path):
        return None, csv_path
    return pd.read_csv(csv_path), csv_path


@st.cache_data(show_spinner=False)
def load_stage67_summary(output_dir):
    path = os.path.join(output_dir, "stage67_summary.json")
    if not os.path.exists(path):
        return None, path
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f), path


def _format_display_number(value, digits=2, suffix=""):
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return "NA"
    if not np.isfinite(numeric):
        return "NA"
    return f"{numeric:.{digits}f}{suffix}"


def _doctor_severity_text(value):
    raw = str(value).strip().lower()
    mapping = {
        "normal_contraction": "Preserved LV systolic function",
        "reduced_contraction": "Reduced LV systolic function",
        "severe_dysfunction": "Severe LV systolic dysfunction",
    }
    return mapping.get(raw, str(value).replace("_", " ").title())


def _confidence_bucket(prob):
    p = float(prob)
    if p >= 0.80:
        return "High"
    if p >= 0.60:
        return "Moderate"
    return "Low"


def _agreement_bucket(disagreement_pct):
    if not np.isfinite(disagreement_pct):
        return "Stage5 unavailable"
    d = abs(float(disagreement_pct))
    if d <= 5.0:
        return "Good agreement"
    if d <= 10.0:
        return "Borderline agreement"
    return "High disagreement"


def _interval_bucket(width_pct):
    if not np.isfinite(width_pct):
        return "Unknown"
    if width_pct <= 8.0:
        return "Tight"
    if width_pct <= 15.0:
        return "Moderate"
    return "Wide"


def _agreement_explanation(disagreement_pct):
    if not np.isfinite(disagreement_pct):
        return "Agreement is unavailable because the Stage4/5 mask-based EF estimate is missing for this study."
    d = abs(float(disagreement_pct))
    if d <= 5.0:
        return f"Good agreement means the Stage1-3 EF head and the Stage4/5 mask-based EF differ by only {d:.1f} percentage points."
    if d <= 10.0:
        return f"Borderline agreement means the Stage1-3 EF head and the Stage4/5 mask-based EF differ by {d:.1f} percentage points."
    return f"High disagreement means the Stage1-3 EF head and the Stage4/5 mask-based EF differ by {d:.1f} percentage points, which is above the 10-point threshold."


def _severity_thresholds_from_summary(summary):
    stage67 = summary if isinstance(summary, dict) else {}
    thresholds = stage67.get("severity_thresholds", {}) if isinstance(stage67.get("severity_thresholds", {}), dict) else {}
    normal_threshold = float(thresholds.get("normal_threshold", 50.0))
    severe_threshold = float(thresholds.get("severe_threshold", 30.0))
    return normal_threshold, severe_threshold


def _severity_rule_text(normal_threshold, severe_threshold):
    return (
        f"Dataset reference label is derived from the dataset EF ground truth using the Stage6 severity thresholds: "
        f"Preserved if EF >= {normal_threshold:.1f}%, Reduced if {severe_threshold:.1f}% <= EF < {normal_threshold:.1f}%, "
        f"and Severe if EF < {severe_threshold:.1f}%."
    )


def _possible_severity_labels_text():
    return "Possible labels: Preserved LV systolic function, Reduced LV systolic function, Severe LV systolic dysfunction."


def _render_stage67_section(selected_video, split, stage67_output_dir):
    pred_df, pred_path = load_stage67_predictions(stage67_output_dir, split)
    summary, summary_path = load_stage67_summary(stage67_output_dir)

    st.subheader("Clinical Summary (Stage 6/7)")
    st.caption("Doctor-facing view of contraction class, fused EF estimate, and uncertainty based on the Stage 6 severity model and Stage 7 calibration.")

    if pred_df is None or pred_df.empty:
        st.info(f"Stage 6/7 predictions not found for {split} at {pred_path}")
        return

    row = pred_df[pred_df["file_name"].astype(str) == str(selected_video)]
    if row.empty and "file_name_ext" in pred_df.columns:
        row = pred_df[pred_df["file_name_ext"].astype(str) == f"{selected_video}.avi"]
    if row.empty:
        st.info(f"No Stage 6/7 row found for {selected_video} in {pred_path}")
        return

    row = row.iloc[0]
    pred_text = _doctor_severity_text(row.get("pred_text_cal", row.get("pred_text_raw", "Unknown")))
    gt_text = _doctor_severity_text(row.get("severity_text_gt", "Unknown"))
    fused_ef = float(row.get("ef_fused_pct", float("nan")))
    ci90_lo = float(row.get("ef_ci90_low", float("nan")))
    ci90_hi = float(row.get("ef_ci90_high", float("nan")))
    ci95_lo = float(row.get("ef_ci95_low", float("nan")))
    ci95_hi = float(row.get("ef_ci95_high", float("nan")))
    prob_cols = ["prob_cal_c0", "prob_cal_c1", "prob_cal_c2"]
    probs = [float(row.get(col, float("nan"))) for col in prob_cols]
    max_prob = max([p for p in probs if np.isfinite(p)], default=float("nan"))
    confidence_text = _confidence_bucket(max_prob) if np.isfinite(max_prob) else "Unknown"
    disagreement_pct = float(row.get("ef_disagreement_pct", float("nan")))
    agreement_text = _agreement_bucket(disagreement_pct)
    agreement_note = _agreement_explanation(disagreement_pct)
    ci90_width = float(ci90_hi - ci90_lo) if np.isfinite(ci90_hi) and np.isfinite(ci90_lo) else float("nan")
    uncertainty_text = _interval_bucket(ci90_width)
    normal_threshold, severe_threshold = _severity_thresholds_from_summary(summary)
    severity_rule_text = _severity_rule_text(normal_threshold, severe_threshold)

    c1, c2 = st.columns(2)
    c3, c4 = st.columns(2)
    c1.metric("Likely Function", pred_text)
    c2.metric("Fused EF (%)", f"{fused_ef:.1f}" if np.isfinite(fused_ef) else "NA")
    c3.metric("Confidence", confidence_text)
    c4.metric("Model Agreement", agreement_text)

    n1, n2 = st.columns([1.1, 0.9])
    with n1:
        summary_rows = [
            {"Clinical item": "Likely contractility class", "Value": pred_text},
            {"Clinical item": "EF 90% interval", "Value": f"{ci90_lo:.1f} to {ci90_hi:.1f}%" if np.isfinite(ci90_lo) and np.isfinite(ci90_hi) else "NA"},
            {"Clinical item": "EF 95% interval", "Value": f"{ci95_lo:.1f} to {ci95_hi:.1f}%" if np.isfinite(ci95_lo) and np.isfinite(ci95_hi) else "NA"},
            {"Clinical item": "Uncertainty width (90%)", "Value": f"{ci90_width:.1f}% ({uncertainty_text})" if np.isfinite(ci90_width) else "NA"},
            {"Clinical item": "Stage1-3 vs Stage4/5 EF gap", "Value": f"{disagreement_pct:.1f}%" if np.isfinite(disagreement_pct) else "NA"},
        ]
        if str(split).upper() in {"TEST", "VAL", "TRAIN"} and pd.notna(row.get("severity_text_gt", np.nan)):
            summary_rows.append({"Clinical item": "Dataset reference label", "Value": gt_text})
        st.dataframe(pd.DataFrame(summary_rows), width="stretch", hide_index=True)

    with n2:
        prob_table = pd.DataFrame(
            [
                {"Contraction class": "Preserved LV systolic function", "Probability": _format_display_number(row.get("prob_cal_c0", float("nan")), digits=3)},
                {"Contraction class": "Reduced LV systolic function", "Probability": _format_display_number(row.get("prob_cal_c1", float("nan")), digits=3)},
                {"Contraction class": "Severe LV systolic dysfunction", "Probability": _format_display_number(row.get("prob_cal_c2", float("nan")), digits=3)},
            ]
        )
        st.dataframe(prob_table, width="stretch", hide_index=True)

    st.markdown("**Reading Notes**")
    st.write(f"- {agreement_note}")
    if str(split).upper() in {"TEST", "VAL", "TRAIN"} and pd.notna(row.get("severity_text_gt", np.nan)):
        gt_ef_text = f"{float(row.get('ef_gt_pct')):.1f}%" if pd.notna(row.get("ef_gt_pct", np.nan)) else "the dataset EF value"
        st.write(f"- {severity_rule_text} This study's dataset EF is {gt_ef_text}, so the displayed reference label is {gt_text}.")
    st.write(f"- {_possible_severity_labels_text()}")

    interpretation = []
    interpretation.append(f"Stage 6 classifies this study as {pred_text.lower()}.")
    if np.isfinite(fused_ef):
        interpretation.append(f"Stage 7 fused EF estimate is {fused_ef:.1f}%.")
    if np.isfinite(ci90_lo) and np.isfinite(ci90_hi):
        interpretation.append(f"The 90% EF interval is {ci90_lo:.1f} to {ci90_hi:.1f}%, which is {uncertainty_text.lower()}.")
    interpretation.append(f"Model confidence is {confidence_text.lower()}.")
    interpretation.append(f"Cross-model agreement is {agreement_text.lower()}.")
    if summary is not None and isinstance(summary, dict):
        stage7 = summary.get("stage7", {}) if isinstance(summary.get("stage7", {}), dict) else {}
        if stage7:
            interpretation.append(
                f"Calibration uses temperature {float(stage7.get('temperature', 1.0)):.2f} and fusion alpha {float(stage7.get('fusion_alpha', 0.5)):.2f}."
            )
    for line in interpretation:
        st.write(f"- {line}")


@st.cache_resource(show_spinner=False)
def load_stage123_model(checkpoint_path, num_frames, device):
    model = EFModel(num_frames=int(num_frames)).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict, checkpoint_dict = _safe_checkpoint_state_dict(checkpoint)
    model_state = model.state_dict()
    filtered_state_dict = {
        key: value
        for key, value in state_dict.items()
        if key in model_state and tuple(value.shape) == tuple(model_state[key].shape)
    }
    incompatible = model.load_state_dict(filtered_state_dict, strict=False)
    model.eval()
    metadata = checkpoint_dict if isinstance(checkpoint_dict, dict) else {}
    return model, incompatible, metadata


@st.cache_resource(show_spinner=False)
def load_stage4_model(checkpoint_path, fallback_model_name, fallback_base_channels, device):
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
    incompatible = model.load_state_dict(state_dict, strict=False)
    model.eval()

    metadata = {
        "model_name": model_name,
        "image_size": int(args.get("image_size", 112)),
        "normalize": str(args.get("normalize", "none")),
        "pretrained": bool(args.get("pretrained", False)),
    }
    return model, metadata, incompatible


def _resolve_stage123_temporal_settings(checkpoint_meta):
    meta = checkpoint_meta if isinstance(checkpoint_meta, dict) else {}
    args = meta.get("args", {}) if isinstance(meta.get("args", {}), dict) else {}
    runtime_config = meta.get("runtime_config", {}) if isinstance(meta.get("runtime_config", {}), dict) else {}

    mode = args.get("phase_temporal_window_mode")
    if mode is None:
        mode = runtime_config.get("PHASE_TEMPORAL_WINDOW_MODE")
    if mode is None:
        mode = "tracing"

    margin = args.get("phase_temporal_window_margin_mult")
    if margin is None:
        margin = runtime_config.get("PHASE_TEMPORAL_WINDOW_MARGIN_MULT")
    if margin is None:
        margin = 1.0 if str(mode).lower() == "tracing" else float(getattr(config, "PHASE_TEMPORAL_WINDOW_MARGIN_MULT", 1.5))

    return {
        "mode": str(mode).lower(),
        "margin_mult": float(margin),
    }


@st.cache_resource(show_spinner=False)
def load_dataset_resource(data_dir, split, num_frames, temporal_window_mode, temporal_window_margin_mult):
    return EchoDataset(
        data_dir=data_dir,
        split=str(split).upper(),
        num_frames=int(num_frames),
        normalize_input=bool(getattr(config, "NORMALIZE_INPUT", True)),
        temporal_window_mode=str(temporal_window_mode).lower(),
        temporal_window_margin_mult=float(temporal_window_margin_mult),
        temporal_window_jitter_mult=0.0,
    )


def _normalize_stage4_input(image_tensor, normalize_mode, pretrained_flag=False):
    mode = str(normalize_mode).lower()
    if mode == "auto":
        mode = "imagenet" if bool(pretrained_flag) else "none"
    if mode == "imagenet":
        mean = torch.tensor([0.485, 0.456, 0.406], dtype=image_tensor.dtype).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], dtype=image_tensor.dtype).view(3, 1, 1)
        image_tensor = (image_tensor - mean) / std
    return image_tensor


def _predict_mask_stage4(model, frame_rgb, image_size, normalize_mode, pretrained_flag, device):
    h, w = frame_rgb.shape[:2]
    resized = cv2.resize(frame_rgb, (image_size, image_size), interpolation=cv2.INTER_LINEAR)
    image_t = torch.from_numpy(resized).permute(2, 0, 1).float() / 255.0
    image_t = _normalize_stage4_input(image_t, normalize_mode, pretrained_flag=pretrained_flag)

    with torch.no_grad():
        logits = model(image_t.unsqueeze(0).to(device))
        if isinstance(logits, dict):
            logits = logits["out"]
        prob = torch.sigmoid(logits)[0, 0].detach().cpu().numpy()

    mask_small = (prob >= 0.5).astype(np.uint8)
    mask_orig = cv2.resize(mask_small, (w, h), interpolation=cv2.INTER_NEAREST)
    return mask_orig, float(mask_orig.sum())


def _predict_area_curve_stage4_from_frames(model, frames_rgb, image_size, normalize_mode, pretrained_flag, device, eval_threshold=0.5, batch_size=16):
    if not frames_rgb:
        return np.zeros(0, dtype=np.int32), np.zeros(0, dtype=np.float64)

    frame_areas = []
    batch_images = []
    batch_sizes = []
    batch_ids = []

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

    for frame_idx, frame_rgb in enumerate(frames_rgb):
        h, w = frame_rgb.shape[:2]
        resized = cv2.resize(frame_rgb, (image_size, image_size), interpolation=cv2.INTER_LINEAR)
        image_t = torch.from_numpy(resized).permute(2, 0, 1).float() / 255.0
        image_t = _normalize_stage4_input(image_t, normalize_mode, pretrained_flag=pretrained_flag)
        batch_images.append(image_t)
        batch_sizes.append((h, w))
        batch_ids.append(frame_idx)
        if len(batch_images) >= int(max(1, batch_size)):
            flush_batch()

    flush_batch()
    if not frame_areas:
        return np.zeros(0, dtype=np.int32), np.zeros(0, dtype=np.float64)

    curve_frame_ids = np.array([fid for fid, _ in frame_areas], dtype=np.int32)
    curve_areas = np.array([area for _, area in frame_areas], dtype=np.float64)
    return curve_frame_ids, curve_areas


def _frame_from_list(frames, idx):
    if len(frames) == 0:
        return None, -1
    idx_clamped = int(np.clip(int(idx), 0, len(frames) - 1))
    return frames[idx_clamped], idx_clamped


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


def _annotate_temporal_frame(frame_rgb, _card_title, _phase_label, _frame_idx, accent_rgb):
    if frame_rgb is None:
        return None

    canvas = np.ascontiguousarray(frame_rgb.copy())
    height, width = canvas.shape[:2]
    accent = tuple(int(v) for v in accent_rgb)
    border = max(2, int(round(min(height, width) * 0.012)))
    cv2.rectangle(canvas, (0, 0), (width - 1, height - 1), accent, thickness=border)
    return canvas


def _frame_to_data_uri(frame_rgb):
    if frame_rgb is None:
        return ""
    bordered = np.ascontiguousarray(frame_rgb)
    success, encoded = cv2.imencode('.png', cv2.cvtColor(bordered, cv2.COLOR_RGB2BGR))
    if not success:
        return ""
    return "data:image/png;base64," + base64.b64encode(encoded.tobytes()).decode('ascii')


def _render_phase_card(frame_rgb, title, phase_code, frame_idx, accent_class, footnote):
    st.markdown(
        f"""
        <div class="phase-card-head">
          <div class="phase-card-title">{title}</div>
          <span class="phase-pill {accent_class}">{phase_code}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )
    if frame_rgb is not None:
        st.image(frame_rgb, width="stretch")
    else:
        st.caption("Frame unavailable")
    st.markdown(
        f"""
        <div class="phase-card-meta">
          <span>Frame {int(frame_idx)}</span>
          <span>{footnote}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_phase_group(group_title, card_specs):
    st.markdown(
        f"""
        <div class="phase-group">
          <div class="phase-group-title">{group_title}</div>
          <div class="phase-group-note">Two key landmarks shown side by side with labels outside the image for readability.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    card_cols = st.columns(2, gap="medium")
    for col, spec in zip(card_cols, card_specs):
        with col:
            _render_phase_card(*spec)


def _make_attention_plot(attn, gt_ed_idx, gt_es_idx, pred_ed_idx, pred_es_idx):
    fig, ax = plt.subplots(figsize=(8, 2.8))
    x = np.arange(attn.shape[0])
    ax.plot(x, attn, label="Stage2 attention", color="#1f77b4")
    ax.axvline(gt_ed_idx, color="green", linestyle="--", label="GT ED")
    ax.axvline(gt_es_idx, color="red", linestyle="--", label="GT ES")
    ax.axvline(pred_ed_idx, color="green", linestyle=":", label="Pred ED")
    ax.axvline(pred_es_idx, color="red", linestyle=":", label="Pred ES")
    ax.set_xlabel("Sampled frame index")
    ax.set_ylabel("Weight")
    ax.set_title("Stage 2 Temporal Weights")
    ax.grid(alpha=0.2)
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    return fig


def _make_phase_plot(phase_probs, gt_ed_idx, gt_es_idx, pred_ed_idx, pred_es_idx):
    fig, ax = plt.subplots(figsize=(8, 2.8))
    x = np.arange(phase_probs.shape[0])
    labels = ["ED", "ES", "Other"]
    colors = ["#2ca02c", "#d62728", "#7f7f7f"]
    for k in range(phase_probs.shape[1]):
        ax.plot(x, phase_probs[:, k], label=f"P({labels[k]})", color=colors[k])

    ax.axvline(gt_ed_idx, color="green", linestyle="--")
    ax.axvline(gt_es_idx, color="red", linestyle="--")
    ax.axvline(pred_ed_idx, color="green", linestyle=":")
    ax.axvline(pred_es_idx, color="red", linestyle=":")
    ax.set_xlabel("Sampled frame index")
    ax.set_ylabel("Probability")
    ax.set_title("Stage 3 Phase Probabilities")
    ax.set_ylim(0.0, 1.0)
    ax.grid(alpha=0.2)
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    return fig


def _expand_attention_to_full_frames(attn, sampled_indices, total_frames):
    sampled = np.asarray(sampled_indices, dtype=np.int32).reshape(-1)
    weights = _summarize_temporal_weights(attn, expected_length=sampled.size)
    total_frames = int(total_frames)

    if total_frames <= 0:
        return np.zeros(0, dtype=np.float64)
    if sampled.size == 0 or weights.size == 0:
        return np.zeros(total_frames, dtype=np.float64)

    n = min(sampled.size, weights.size)
    sampled = sampled[:n]
    weights = weights[:n]

    order = np.argsort(sampled)
    sampled = sampled[order]
    weights = weights[order]

    unique_sampled, unique_idx = np.unique(sampled, return_index=True)
    unique_weights = weights[unique_idx]

    if unique_sampled.size == 1:
        return np.full(total_frames, float(unique_weights[0]), dtype=np.float64)

    frame_axis = np.arange(total_frames, dtype=np.float64)
    full_weights = np.interp(
        frame_axis,
        unique_sampled.astype(np.float64),
        unique_weights.astype(np.float64),
        left=float(unique_weights[0]),
        right=float(unique_weights[-1]),
    )
    return np.clip(full_weights.astype(np.float64), 0.0, None)


def _summarize_temporal_weights(attn, expected_length=None):
    weights = np.asarray(attn, dtype=np.float64)
    if weights.size == 0:
        return np.zeros(0, dtype=np.float64)

    if weights.ndim == 1:
        summary = weights.reshape(-1)
    else:
        summary = None
        if expected_length is not None:
            for axis, axis_size in enumerate(weights.shape):
                if int(axis_size) != int(expected_length):
                    continue
                reduce_axes = tuple(i for i in range(weights.ndim) if i != axis)
                collapsed = weights.mean(axis=reduce_axes) if reduce_axes else weights
                summary = np.asarray(collapsed, dtype=np.float64).reshape(-1)
                break
        if summary is None:
            summary = weights.reshape(-1)

    summary = np.clip(summary.astype(np.float64), 0.0, None)
    return summary


def _aligned_sampled_temporal_weights(attn, sampled_indices):
    sampled = np.asarray(sampled_indices, dtype=np.int32).reshape(-1)
    weights = _summarize_temporal_weights(attn, expected_length=sampled.size)
    n = min(sampled.size, weights.size)
    if n <= 0:
        return np.zeros(0, dtype=np.int32), np.zeros(0, dtype=np.float64)
    return sampled[:n], weights[:n]


def _collapse_duplicate_sampled_frames(sampled_indices, sampled_weights):
    sampled = np.asarray(sampled_indices, dtype=np.int32).reshape(-1)
    weights = np.asarray(sampled_weights, dtype=np.float64).reshape(-1)
    n = min(sampled.size, weights.size)
    if n <= 0:
        return np.zeros(0, dtype=np.int32), np.zeros(0, dtype=np.float64)

    sampled = sampled[:n]
    weights = weights[:n]
    order = np.argsort(sampled, kind="stable")
    sampled = sampled[order]
    weights = weights[order]

    unique_frames, inverse = np.unique(sampled, return_inverse=True)
    collapsed_weights = np.zeros(unique_frames.shape[0], dtype=np.float64)
    np.add.at(collapsed_weights, inverse, weights)
    return unique_frames.astype(np.int32), collapsed_weights


def _top_temporal_frame_specs(result, top_k=4):
    sampled_indices, sampled_weights = _aligned_sampled_temporal_weights(
        result["stage2_attention"],
        result["sampled_indices"],
    )
    unique_indices, unique_weights = _collapse_duplicate_sampled_frames(sampled_indices, sampled_weights)
    if unique_indices.size == 0 or unique_weights.size == 0:
        return []

    order = np.argsort(-unique_weights, kind="stable")
    specs = []
    for rank, pos in enumerate(order[:max(1, int(top_k))], start=1):
        frame_idx = int(unique_indices[pos])
        frame_rgb, resolved_idx = _frame_from_list(result["full_frames"], frame_idx)
        if frame_rgb is None:
            continue
        specs.append(
            {
                "rank": rank,
                "frame_idx": int(resolved_idx),
                "weight": float(unique_weights[pos]),
                "image": _annotate_temporal_frame(frame_rgb, "", "", resolved_idx, (37, 92, 199)),
                "gt_context": _phase_window_label(resolved_idx, result["ed_orig"], result["es_orig"]),
                "pred_context": _phase_window_label(resolved_idx, result["pred_ed_orig"], result["pred_es_orig"]),
                "nearest_gt": _nearest_landmark_text(resolved_idx, result["ed_orig"], result["es_orig"], "GT"),
                "nearest_pred": _nearest_landmark_text(resolved_idx, result["pred_ed_orig"], result["pred_es_orig"], "Pred"),
            }
        )
    return specs


def _phase_window_label(frame_idx, ed_idx, es_idx):
    frame_idx = int(frame_idx)
    ed_idx = int(ed_idx)
    es_idx = int(es_idx)

    if ed_idx <= es_idx:
        if frame_idx < ed_idx:
            return "Pre-ED"
        if frame_idx == ed_idx:
            return "End-Diastole"
        if frame_idx < es_idx:
            return "Systolic window"
        if frame_idx == es_idx:
            return "End-Systole"
        return "Post-ES"

    lo, hi = sorted((ed_idx, es_idx))
    if frame_idx < lo:
        return "Before landmarks"
    if frame_idx > hi:
        return "After landmarks"
    if frame_idx == ed_idx:
        return "End-Diastole"
    if frame_idx == es_idx:
        return "End-Systole"
    return "Between landmarks"


def _nearest_landmark_text(frame_idx, ed_idx, es_idx, prefix):
    frame_idx = int(frame_idx)
    choices = [("ED", int(ed_idx)), ("ES", int(es_idx))]
    label, ref_idx = min(choices, key=lambda item: abs(frame_idx - item[1]))
    delta = frame_idx - ref_idx
    if delta == 0:
        return f"At {prefix} {label}"
    if delta < 0:
        return f"{abs(delta)} fr before {prefix} {label}"
    return f"{abs(delta)} fr after {prefix} {label}"


def _render_temporal_weight_video(result):
    total_frames = len(result["full_frames"])
    frame_weights = _expand_attention_to_full_frames(
        result["stage2_attention"],
        result["sampled_indices"],
        total_frames,
    )
    if frame_weights.size == 0:
        st.warning("Temporal weights unavailable for playback view.")
        return False

    sampled_indices = np.asarray(result["sampled_indices"], dtype=np.int32).reshape(-1)
    sampled_weights = _summarize_temporal_weights(result["stage2_attention"], expected_length=sampled_indices.size)
    sample_count = min(sampled_indices.size, sampled_weights.size)
    peak_frame = int(np.argmax(frame_weights)) if frame_weights.size > 0 else 0
    peak_weight = float(frame_weights[peak_frame]) if frame_weights.size > 0 else 0.0
    top_frame_specs = _top_temporal_frame_specs(result, top_k=4)

    gt_ed_frame_rgb, _ = _frame_from_list(result["full_frames"], result["ed_orig"])
    gt_es_frame_rgb, _ = _frame_from_list(result["full_frames"], result["es_orig"])
    pred_ed_frame_rgb, _ = _frame_from_list(result["full_frames"], result["pred_ed_orig"])
    pred_es_frame_rgb, _ = _frame_from_list(result["full_frames"], result["pred_es_orig"])

    st.markdown(
        """
        <div class="section-heading">
          <div class="section-title">Temporal Media Overview</div>
          <div class="section-subtitle">Animated preview, key ED/ES metrics, and temporal-weight interpretation for this study.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    playback_col, summary_col = st.columns([1.08, 0.92], gap="large")
    with playback_col:
        st.markdown(
            """
            <div class="media-shell">
              <div class="shell-title">Animated Source Preview</div>
              <div class="shell-subtitle">Compact cine preview for quick review. The graph below remains the main temporal analysis surface.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        gif_bytes, gif_err = _prepare_gif_preview(result["video_path"])
        if gif_bytes:
            left_pad, image_col, right_pad = st.columns([0.12, 0.76, 0.12])
            with image_col:
                st.image(gif_bytes, caption="Animated source preview", width=PREVIEW_DISPLAY_WIDTH)
        else:
            st.caption(f"Animated preview unavailable: {gif_err}")
    with summary_col:
        st.markdown(
            """
            <div class="media-shell">
              <div class="shell-title">Cardiac Phase Summary</div>
              <div class="shell-subtitle">Ground-truth and predicted frame landmarks are shown side by side for this case.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        stat_row_1 = st.columns(2)
        stat_row_1[0].metric("GT ED/ES", f"{int(result['ed_orig'])} / {int(result['es_orig'])}")
        stat_row_1[1].metric("Pred ED/ES", f"{int(result['pred_ed_orig'])} / {int(result['pred_es_orig'])}")
        stat_row_2 = st.columns(2)
        stat_row_2[0].metric("ED Error", f"{int(result['ed_err_orig'])} fr")
        stat_row_2[1].metric("ES Error", f"{int(result['es_err_orig'])} fr")
        st.caption("GT = dataset annotation for this case. Pred = model-predicted landmark mapped back to the original video frame number.")

    st.markdown(
        """
        <div class="section-heading">
          <div class="section-title">Cardiac Phase Frames</div>
          <div class="section-subtitle">Green indicates End-Diastole (ED) and red indicates End-Systole (ES). Labels stay outside the image so the echo frames remain unobstructed.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    compare_left, compare_right = st.columns(2, gap="large")
    with compare_left:
        _render_phase_group(
            "Ground Truth",
            [
                (
                    _annotate_temporal_frame(gt_ed_frame_rgb, "", "", result["ed_orig"], (22, 163, 74)),
                    "End-Diastole",
                    "ED",
                    result["ed_orig"],
                    "phase-pill-ed",
                    "Ground truth",
                ),
                (
                    _annotate_temporal_frame(gt_es_frame_rgb, "", "", result["es_orig"], (220, 38, 38)),
                    "End-Systole",
                    "ES",
                    result["es_orig"],
                    "phase-pill-es",
                    "Ground truth",
                ),
            ],
        )
    with compare_right:
        _render_phase_group(
            "Model Prediction",
            [
                (
                    _annotate_temporal_frame(pred_ed_frame_rgb, "", "", result["pred_ed_orig"], (22, 163, 74)),
                    "End-Diastole",
                    "ED",
                    result["pred_ed_orig"],
                    "phase-pill-ed",
                    "Predicted",
                ),
                (
                    _annotate_temporal_frame(pred_es_frame_rgb, "", "", result["pred_es_orig"], (220, 38, 38)),
                    "End-Systole",
                    "ES",
                    result["pred_es_orig"],
                    "phase-pill-es",
                    "Predicted",
                ),
            ],
        )

    st.markdown(
        """
        <div class="section-heading">
          <div class="section-title">Top Temporal-Weight Frames</div>
          <div class="section-subtitle">These are distinct sampled frames used by Stage 2, ranked by temporal weight after merging duplicate sampled slots that round to the same original frame.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    if top_frame_specs:
        top_cols = st.columns(len(top_frame_specs), gap="medium")
        for col, spec in zip(top_cols, top_frame_specs):
            with col:
                st.metric(f"Top {spec['rank']}", f"Frame {spec['frame_idx']}")
                st.image(
                    spec["image"],
                    caption=f"Exact sampled frame | weight {spec['weight']:.4f}",
                    width=FRAME_DISPLAY_WIDTH,
                )
                st.caption(f"GT context: {spec['gt_context']}")
                st.caption(f"Pred context: {spec['pred_context']}")
                st.caption(f"{spec['nearest_gt']} | {spec['nearest_pred']}")
    else:
        st.caption("Top temporal-weight frames are unavailable for this case.")

    component_id = hashlib.md5(
        f"{result['video_path']}|{total_frames}|{result['fps']}|{result['ed_orig']}|{result['pred_ed_orig']}".encode("utf-8")
    ).hexdigest()[:12]
    selected_frame = int(np.clip(peak_frame, 0, max(0, total_frames - 1)))
    zoom_start = 0
    zoom_end = max(0, total_frames - 1)

    selected_frame_rgb, _ = _frame_from_list(result["full_frames"], selected_frame)
    selected_preview = _annotate_temporal_frame(selected_frame_rgb, "", "", selected_frame, (37, 92, 199))
    selected_weight = float(frame_weights[selected_frame]) if frame_weights.size > 0 else 0.0
    gt_context = _phase_window_label(selected_frame, result["ed_orig"], result["es_orig"])
    pred_context = _phase_window_label(selected_frame, result["pred_ed_orig"], result["pred_es_orig"])
    nearest_gt = _nearest_landmark_text(selected_frame, result["ed_orig"], result["es_orig"], "GT")
    nearest_pred = _nearest_landmark_text(selected_frame, result["pred_ed_orig"], result["pred_es_orig"], "Pred")
    nearest_sampled = int(sampled_indices[np.argmin(np.abs(sampled_indices - selected_frame))]) if sample_count > 0 else int(selected_frame)
    if nearest_sampled == int(selected_frame):
        sampling_note = "Selected frame is one of the sampled frames used by Stage 2."
    else:
        sampling_note = f"Temporal weight is interpolated to this frame. Nearest sampled frame: {nearest_sampled}."

    st.markdown(
        """
        <div class="section-heading">
          <div class="section-title">Linked Frame Inspector</div>
          <div class="section-subtitle">Use the frame scrubber to connect the timeline marker with the actual echo frame and its phase context.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    inspect_col, context_col = st.columns([0.88, 1.12], gap="large")
    with inspect_col:
        st.markdown(
            """
            <div class="media-shell">
              <div class="shell-title">Selected Frame Preview</div>
              <div class="shell-subtitle">The blue border marks the exact frame currently highlighted on the temporal-weight timeline.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        if selected_preview is not None:
            st.image(selected_preview, caption=f"Frame {selected_frame} linked to the timeline", width=FRAME_DISPLAY_WIDTH)
        else:
            st.caption("Selected frame unavailable.")
    with context_col:
        st.markdown(
            """
            <div class="media-shell">
              <div class="shell-title">Phase Context</div>
              <div class="shell-subtitle">This summary ties the selected frame to both the ground-truth landmarks and the model's predicted ED/ES window.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        ctx_row_1 = st.columns(2)
        ctx_row_1[0].metric("GT Phase Context", gt_context)
        ctx_row_1[1].metric("Pred Phase Context", pred_context)
        ctx_row_2 = st.columns(2)
        ctx_row_2[0].metric("Selected Weight", f"{selected_weight:.4f}")
        ctx_row_2[1].metric("Nearest Sample", str(nearest_sampled))
        st.caption(f"Ground truth: {nearest_gt}")
        st.caption(f"Prediction: {nearest_pred}")
        st.caption(sampling_note)

    payload = {
        "frameWeights": [round(float(v), 6) for v in frame_weights.tolist()],
        "sampledIndices": [int(v) for v in sampled_indices[:sample_count].tolist()],
        "sampledWeights": [round(float(v), 6) for v in sampled_weights[:sample_count].tolist()],
        "gtEd": int(result["ed_orig"]),
        "gtEs": int(result["es_orig"]),
        "predEd": int(result["pred_ed_orig"]),
        "predEs": int(result["pred_es_orig"]),
        "peakFrame": peak_frame,
        "peakWeight": round(peak_weight, 6),
        "selectedFrame": int(selected_frame),
        "selectedWeight": round(selected_weight, 6),
        "gtContext": gt_context,
        "predContext": pred_context,
        "totalFrames": int(total_frames),
        "zoomStart": int(zoom_start),
        "zoomEnd": int(zoom_end),
        "edErr": int(result["ed_err_orig"]),
        "esErr": int(result["es_err_orig"]),
    }

    payload_json = json.dumps(payload)

    components.html(
        f"""
        <div id="tw-{component_id}" style="font-family: 'Trebuchet MS', 'Segoe UI', sans-serif; color: #102a43; margin-top: 10px;">
          <style>
            #tw-{component_id} {{
              --tw-ink: #102a43;
              --tw-muted: #5f7689;
              --tw-line: #255cc7;
              --tw-sample: #f97316;
              --tw-ed: #16a34a;
              --tw-es: #dc2626;
              --tw-pred-band: rgba(37, 92, 199, 0.05);
              --tw-gt-band: rgba(22, 163, 74, 0.05);
              --tw-border: rgba(148, 163, 184, 0.18);
            }}
            #tw-{component_id} .tw-shell {{
              background: linear-gradient(180deg, rgba(255,255,255,0.96) 0%, rgba(243,248,253,0.96) 100%);
              border: 1px solid var(--tw-border);
              border-radius: 24px;
              padding: 18px;
              box-shadow: 0 10px 22px rgba(15, 23, 42, 0.05);
            }}
            #tw-{component_id} .tw-header {{
              display: flex;
              justify-content: space-between;
              align-items: flex-start;
              gap: 12px;
              margin-bottom: 12px;
            }}
            #tw-{component_id} .tw-title {{
              margin: 0;
              font-size: 1.14rem;
              font-weight: 800;
              color: var(--tw-ink);
            }}
            #tw-{component_id} .tw-subtitle {{
              margin: 4px 0 0 0;
              color: var(--tw-muted);
              font-size: 0.88rem;
              line-height: 1.45;
              max-width: 620px;
            }}
            #tw-{component_id} .tw-chip-row {{
              display: flex;
              flex-wrap: wrap;
              gap: 8px;
              justify-content: flex-end;
            }}
            #tw-{component_id} .tw-chip {{
              border-radius: 999px;
              padding: 6px 10px;
              background: rgba(255,255,255,0.95);
              border: 1px solid var(--tw-border);
              color: var(--tw-muted);
              font-size: 11px;
              font-weight: 700;
              letter-spacing: 0.02em;
            }}
            #tw-{component_id} .tw-grid {{
              display: grid;
              grid-template-columns: repeat(4, minmax(0, 1fr));
              gap: 10px;
              margin-bottom: 12px;
            }}
            #tw-{component_id} .tw-stat {{
              background: rgba(255,255,255,0.92);
              border: 1px solid var(--tw-border);
              border-radius: 16px;
              padding: 12px 14px;
            }}
            #tw-{component_id} .tw-stat-label {{
              font-size: 10px;
              text-transform: uppercase;
              letter-spacing: 0.08em;
              color: #71879a;
              margin-bottom: 4px;
            }}
            #tw-{component_id} .tw-stat-value {{
              font-size: 1.05rem;
              font-weight: 800;
              color: var(--tw-ink);
              line-height: 1.25;
            }}
            #tw-{component_id} .tw-chart-card {{
              background: #ffffff;
              border: 1px solid var(--tw-border);
              border-radius: 20px;
              padding: 14px;
            }}
            #tw-{component_id} .tw-chart {{
              width: 100%;
              height: 344px;
              display: block;
            }}
            #tw-{component_id} .tw-footer {{
              margin-top: 10px;
              display: grid;
              grid-template-columns: 1fr auto;
              gap: 10px;
              align-items: center;
            }}
            #tw-{component_id} .tw-footer-note {{
              color: var(--tw-muted);
              font-size: 11.5px;
              line-height: 1.45;
            }}
            #tw-{component_id} .tw-legend {{
              display: flex;
              flex-wrap: wrap;
              justify-content: flex-end;
              gap: 10px;
              color: var(--tw-muted);
              font-size: 11px;
            }}
            #tw-{component_id} .tw-legend-item {{
              display: inline-flex;
              align-items: center;
              gap: 6px;
            }}
            #tw-{component_id} .tw-swatch {{
              width: 11px;
              height: 11px;
              border-radius: 999px;
              display: inline-block;
            }}
            @media (max-width: 900px) {{
              #tw-{component_id} .tw-grid {{
                grid-template-columns: repeat(2, minmax(0, 1fr));
              }}
              #tw-{component_id} .tw-header {{
                flex-direction: column;
              }}
              #tw-{component_id} .tw-chip-row {{
                justify-content: flex-start;
              }}
              #tw-{component_id} .tw-footer {{
                grid-template-columns: 1fr;
              }}
              #tw-{component_id} .tw-legend {{
                justify-content: flex-start;
              }}
            }}
          </style>
          <div class="tw-shell">
            <div class="tw-header">
              <div>
                <div class="tw-title">Temporal Weight Timeline</div>
                <div class="tw-subtitle">GT marks the ground-truth ED/ES landmarks for this case. Pred marks the model-predicted ED/ES landmarks mapped back to the original frame numbers.</div>
              </div>
              <div class="tw-chip-row">
                <span class="tw-chip">{payload['totalFrames']} frames</span>
                <span class="tw-chip">Peak frame {payload['selectedFrame']}</span>
              </div>
            </div>
            <div class="tw-grid">
              <div class="tw-stat">
                <div class="tw-stat-label">Selected frame</div>
                <div class="tw-stat-value">{payload['selectedFrame']} / {payload['totalFrames'] - 1}</div>
              </div>
              <div class="tw-stat">
                <div class="tw-stat-label">Temporal weight</div>
                <div class="tw-stat-value">{payload['selectedWeight']:.4f}</div>
              </div>
              <div class="tw-stat">
                <div class="tw-stat-label">GT phase context</div>
                <div class="tw-stat-value">{payload['gtContext']}</div>
              </div>
              <div class="tw-stat">
                <div class="tw-stat-label">Pred phase context</div>
                <div class="tw-stat-value">{payload['predContext']}</div>
              </div>
            </div>
            <div class="tw-chart-card">
              <svg id="tw-chart-{component_id}" class="tw-chart" viewBox="0 0 820 344" preserveAspectRatio="none"></svg>
              <div class="tw-footer">
                <div class="tw-footer-note">GT = ground-truth landmark for this case. Pred = model-predicted landmark. The selected marker, preview frame, and phase-context cards all refer to the same peak-weight frame.</div>
                <div class="tw-legend">
                  <span class="tw-legend-item"><span class="tw-swatch" style="background:#245fd9;"></span>Weight curve</span>
                  <span class="tw-legend-item"><span class="tw-swatch" style="background:#f97316;"></span>Sampled frames</span>
                  <span class="tw-legend-item"><span class="tw-swatch" style="background:rgba(22,163,74,0.55);"></span>GT systole</span>
                  <span class="tw-legend-item"><span class="tw-swatch" style="background:rgba(37,92,199,0.35);"></span>Pred systole</span>
                  <span class="tw-legend-item"><span class="tw-swatch" style="background:#0f172a;"></span>Selected frame</span>
                </div>
              </div>
            </div>
          </div>
        </div>
        <script>
          const payload = {payload_json};
          const svg = document.getElementById("tw-chart-{component_id}");
          const width = 820;
          const height = 344;
          const padLeft = 58;
          const padRight = 18;
          const padTop = 28;
          const padBottom = 58;
          const graphHeight = height - padTop - padBottom;
          const graphWidth = width - padLeft - padRight;
          const visibleStart = Math.max(0, Math.min(payload.zoomStart ?? 0, Math.max(0, payload.totalFrames - 1)));
          const visibleEnd = Math.max(visibleStart, Math.min(payload.zoomEnd ?? Math.max(0, payload.totalFrames - 1), Math.max(0, payload.totalFrames - 1)));
          const visibleSpan = Math.max(1, visibleEnd - visibleStart);
          const visibleWeights = payload.frameWeights.slice(visibleStart, visibleEnd + 1);
          const maxWeight = Math.max(...visibleWeights, 1e-6);

          function inView(frame) {{
            return frame >= visibleStart && frame <= visibleEnd;
          }}

          function xForFrame(frame) {{
            if (visibleEnd <= visibleStart) {{
              return padLeft + graphWidth / 2;
            }}
            return padLeft + ((frame - visibleStart) / visibleSpan) * graphWidth;
          }}

          function yForWeight(weight) {{
            return padTop + graphHeight - (weight / maxWeight) * graphHeight;
          }}

          function buildLinePath(points) {{
            return points.map((point, idx) => `${{idx === 0 ? 'M' : 'L'}}${{point[0].toFixed(2)}},${{point[1].toFixed(2)}}`).join(' ');
          }}

          function buildAreaPath(points) {{
            if (points.length === 0) {{
              return '';
            }}
            const linePath = buildLinePath(points);
            const last = points[points.length - 1];
            const first = points[0];
            return `${{linePath}} L${{last[0].toFixed(2)}},${{(padTop + graphHeight).toFixed(2)}} L${{first[0].toFixed(2)}},${{(padTop + graphHeight).toFixed(2)}} Z`;
          }}

          function bandMarkup(startFrame, endFrame, fill, opacity, label, labelY) {{
            const rawStart = Math.max(0, Math.min(startFrame, endFrame));
            const rawEnd = Math.min(payload.totalFrames - 1, Math.max(startFrame, endFrame));
            if (rawEnd < visibleStart || rawStart > visibleEnd) {{
              return '';
            }}
            const start = Math.max(visibleStart, rawStart);
            const end = Math.min(visibleEnd, rawEnd);
            const x1 = xForFrame(start);
            const x2 = xForFrame(end);
            const bandWidth = Math.max(2, x2 - x1);
            const labelMarkup = bandWidth > 42
              ? `<text x="${{Math.min(width - 18, Math.max(18, x1 + 8))}}" y="${{labelY}}" fill="#6b8297" font-size="10.5" font-weight="700">${{label}}</text>`
              : '';
            return `<rect x="${{x1}}" y="${{padTop}}" width="${{bandWidth}}" height="${{graphHeight}}" fill="${{fill}}" fill-opacity="${{opacity}}"></rect>${{labelMarkup}}`;
          }}

          function eventMarkup(frame, color, dash, label, anchor) {{
            if (!inView(frame)) {{
              return '';
            }}
            const x = xForFrame(frame);
            const pillWidth = Math.max(54, label.length * 6.6 + 14);
            const pillX = Math.min(width - pillWidth - 12, Math.max(12, x - pillWidth / 2));
            const pillY = anchor === 'top' ? padTop + 4 : padTop + graphHeight - 24;
            const textY = pillY + 13;
            return `
              <line x1="${{x}}" y1="${{padTop}}" x2="${{x}}" y2="${{padTop + graphHeight}}" stroke="${{color}}" stroke-width="1.8" stroke-dasharray="${{dash}}" opacity="0.82"></line>
              <rect x="${{pillX}}" y="${{pillY}}" width="${{pillWidth}}" height="18" rx="9" fill="white" stroke="${{color}}" stroke-width="1"></rect>
              <text x="${{pillX + pillWidth / 2}}" y="${{textY}}" fill="${{color}}" text-anchor="middle" font-size="10.5" font-weight="800">${{label}}</text>
            `;
          }}

          function selectedMarkup(frame, weight) {{
            if (!inView(frame)) {{
              return '';
            }}
            const x = xForFrame(frame);
            const y = yForWeight(weight);
            const pillWidth = 72;
            const pillX = Math.min(width - pillWidth - 12, Math.max(12, x - pillWidth / 2));
            return `
              <line x1="${{x}}" y1="${{padTop}}" x2="${{x}}" y2="${{padTop + graphHeight}}" stroke="#0f172a" stroke-width="2.2" stroke-dasharray="3 4" opacity="0.72"></line>
              <rect x="${{pillX}}" y="${{padTop + 22}}" width="${{pillWidth}}" height="18" rx="9" fill="#0f172a"></rect>
              <text x="${{pillX + pillWidth / 2}}" y="${{padTop + 35}}" fill="white" text-anchor="middle" font-size="10.5" font-weight="800">Selected</text>
              <circle cx="${{x}}" cy="${{y}}" r="5.6" fill="#0f172a" stroke="white" stroke-width="1.8"></circle>
            `;
          }}

          function curveEventDot(frame, color) {{
            const clamped = Math.max(0, Math.min(payload.totalFrames - 1, frame));
            if (!inView(clamped)) {{
              return '';
            }}
            const x = xForFrame(clamped);
            const y = yForWeight(payload.frameWeights[clamped] || 0);
            return `
              <circle cx="${{x}}" cy="${{y}}" r="5.2" fill="white" stroke="${{color}}" stroke-width="2.2"></circle>
              <circle cx="${{x}}" cy="${{y}}" r="2.0" fill="${{color}}"></circle>
            `;
          }}

          const points = visibleWeights.map((weight, offset) => [xForFrame(visibleStart + offset), yForWeight(weight)]);
          const linePath = buildLinePath(points);
          const areaPath = buildAreaPath(points);
          const sampledDots = payload.sampledIndices.map((frame, idx) => {{
            if (!inView(frame)) {{
              return '';
            }}
            return `
              <circle cx="${{xForFrame(frame)}}" cy="${{yForWeight(payload.sampledWeights[idx] ?? 0)}}" r="3.3" fill="#f97316" stroke="white" stroke-width="1"></circle>
            `;
          }}).join('');
          const xTickFrames = Array.from(new Set([
            visibleStart,
            Math.max(visibleStart, Math.floor(visibleStart + visibleSpan * 0.25)),
            Math.max(visibleStart, Math.floor(visibleStart + visibleSpan * 0.50)),
            Math.max(visibleStart, Math.floor(visibleStart + visibleSpan * 0.75)),
            visibleEnd,
          ])).sort((a, b) => a - b);
          const xTicks = xTickFrames.map((frame) => `
            <line x1="${{xForFrame(frame)}}" y1="${{padTop + graphHeight}}" x2="${{xForFrame(frame)}}" y2="${{padTop + graphHeight + 6}}" stroke="#9db2c7"></line>
            <text x="${{xForFrame(frame)}}" y="${{height - 10}}" fill="#627d98" text-anchor="middle" font-size="11.2">${{frame}}</text>
          `).join('');
          const yTickValues = [0, maxWeight / 2, maxWeight];
          const yTicks = yTickValues.map((weight) => `
            <line x1="${{padLeft}}" y1="${{yForWeight(weight)}}" x2="${{width - padRight}}" y2="${{yForWeight(weight)}}" stroke="#e7eef6"></line>
            <text x="14" y="${{yForWeight(weight) + 4}}" fill="#627d98" font-size="11.2">${{weight.toFixed(3)}}</text>
          `).join('');

          svg.innerHTML = `
            <defs>
              <linearGradient id="tw-fill-{component_id}" x1="0" y1="0" x2="0" y2="1">
                <stop offset="0%" stop-color="#245fd9" stop-opacity="0.18"></stop>
                <stop offset="100%" stop-color="#245fd9" stop-opacity="0.02"></stop>
              </linearGradient>
            </defs>
            <rect x="0" y="0" width="820" height="328" rx="18" fill="#ffffff"></rect>
            ${{bandMarkup(payload.gtEd, payload.gtEs, '#16a34a', 0.05, 'GT systole', padTop + 18)}}
            ${{bandMarkup(payload.predEd, payload.predEs, '#255cc7', 0.05, 'Pred systole', padTop + graphHeight - 8)}}
            ${{yTicks}}
            <path d="${{areaPath}}" fill="url(#tw-fill-{component_id})"></path>
            <path d="${{linePath}}" fill="none" stroke="#245fd9" stroke-width="3.3" stroke-linecap="round" stroke-linejoin="round"></path>
            ${{sampledDots}}
            ${{eventMarkup(payload.gtEd, '#16a34a', '4 5', 'GT-ED', 'top')}}
            ${{eventMarkup(payload.gtEs, '#dc2626', '4 5', 'GT-ES', 'top')}}
            ${{eventMarkup(payload.predEd, '#16a34a', '2 6', 'Pred-ED', 'bottom')}}
            ${{eventMarkup(payload.predEs, '#dc2626', '2 6', 'Pred-ES', 'bottom')}}
            ${{curveEventDot(payload.gtEd, '#16a34a')}}
            ${{curveEventDot(payload.gtEs, '#dc2626')}}
            ${{curveEventDot(payload.predEd, '#16a34a')}}
            ${{curveEventDot(payload.predEs, '#dc2626')}}
            ${{selectedMarkup(payload.selectedFrame, payload.selectedWeight)}}
            <line x1="${{padLeft}}" y1="${{padTop + graphHeight}}" x2="${{width - padRight}}" y2="${{padTop + graphHeight}}" stroke="#9db2c7" stroke-width="1.1"></line>
            <line x1="${{padLeft}}" y1="${{padTop}}" x2="${{padLeft}}" y2="${{padTop + graphHeight}}" stroke="#9db2c7" stroke-width="1.1"></line>
            ${{xTicks}}
            <text x="${{width / 2}}" y="${{height - 12}}" fill="#486581" text-anchor="middle" font-size="12" font-weight="700">Frame index</text>
            <text x="14" y="18" fill="#486581" font-size="12" font-weight="700">Weight</text>
          `;
        </script>
        """,
        height=540,
    )

    return True


def run_case(
    data_dir,
    split,
    video_name,
    stage123_checkpoint,
    num_frames,
    device,
    run_stage4,
    stage4_checkpoint,
    stage4_model_name,
    stage4_base_channels,
):
    stage45 = Stage45Pipeline()

    model123, incompat123, stage123_meta = load_stage123_model(stage123_checkpoint, int(num_frames), device)
    temporal_settings = _resolve_stage123_temporal_settings(stage123_meta)
    dataset = load_dataset_resource(
        data_dir=data_dir,
        split=split,
        num_frames=int(num_frames),
        temporal_window_mode=temporal_settings["mode"],
        temporal_window_margin_mult=temporal_settings["margin_mult"],
    )
    row_match = dataset.filelist[dataset.filelist["FileName"] == video_name]
    if row_match.empty:
        raise ValueError(f"Video {video_name} not found in split {split}")
    row = row_match.iloc[0]

    file_name_ext = f"{video_name}.avi"
    video_path = os.path.join(data_dir, "Videos", file_name_ext)
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video not found: {video_path}")

    full_frames, fps = _read_video_frames_rgb(video_path)

    ed_orig = int(dataset.phase_dict[file_name_ext]["ed"])
    es_orig = int(dataset.phase_dict[file_name_ext]["es"])

    clip, sampled_indices = dataset.load_video(video_path, ed_original=ed_orig, es_original=es_orig)

    gt_ed_idx = int(np.argmin(np.abs(sampled_indices - int(ed_orig)))) if ed_orig >= 0 else 0
    gt_es_idx = int(np.argmin(np.abs(sampled_indices - int(es_orig)))) if es_orig >= 0 else 0
    ef_gt_pct = float(row["EF"])

    with torch.no_grad():
        out = model123(clip.unsqueeze(0).to(device), return_stage_outputs=True)

    if isinstance(out, tuple) and len(out) == 4:
        ef_pred, attention, phase_logits, stage_outputs = out
    else:
        ef_pred, attention, phase_logits = out
        stage_outputs = {}

    pred_ed_idx_t, pred_es_idx_t = Stage3PhaseDetector.predict_indices(phase_logits)
    pred_ed_idx = int(pred_ed_idx_t[0].item())
    pred_es_idx = int(pred_es_idx_t[0].item())

    pred_ed_orig = int(sampled_indices[pred_ed_idx])
    pred_es_orig = int(sampled_indices[pred_es_idx])

    ef_pred_pct = float(ef_pred[0].item() * 100.0)

    attn = attention[0].detach().cpu().numpy()
    phase_probs = torch.softmax(phase_logits[0], dim=-1).detach().cpu().numpy()

    stage1_features = stage_outputs.get("stage1_features")
    if stage1_features is not None:
        s1 = stage1_features[0].detach().cpu().float()
        stage1_feat_norm = float(torch.linalg.vector_norm(s1.flatten(), dim=0).item())
        stage1_temp_std = float(s1.std(dim=1, unbiased=False).mean().item())
        stage1_tokens = int(s1.shape[1])
    else:
        stage1_feat_norm = float("nan")
        stage1_temp_std = float("nan")
        stage1_tokens = int(attn.shape[0])

    stage2_entropy = _attention_entropy(attn)
    stage2_peak = float(attn.max())
    stage2_peak_idx = int(np.argmax(attn))
    stage2_peak_to_event = float(min(abs(stage2_peak_idx - gt_ed_idx), abs(stage2_peak_idx - gt_es_idx)))

    phase_pred_ed = int(np.argmax(phase_probs[:, 1]))
    phase_pred_es = int(np.argmax(phase_probs[:, 2]))

    ed_err_sampled = int(abs(pred_ed_idx - gt_ed_idx))
    es_err_sampled = int(abs(pred_es_idx - gt_es_idx))
    ed_err_orig = int(abs(pred_ed_orig - ed_orig))
    es_err_orig = int(abs(pred_es_orig - es_orig))

    tracings = load_volume_tracings(data_dir)
    video_rows = tracings[tracings["FileName"] == file_name_ext].copy()

    gt_masks = {}
    gt_areas = {}
    if not video_rows.empty:
        h_full, w_full = full_frames[0].shape[:2]
        for frame_id in sorted(int(v) for v in video_rows["Frame"].unique().tolist()):
            rows_f = video_rows[video_rows["Frame"] == frame_id]
            mask = stage45.tracing_to_mask(rows_f, height=h_full, width=w_full)
            gt_masks[frame_id] = mask
            gt_areas[frame_id] = float(mask.sum())

    gt_ed_area = gt_areas.get(ed_orig, float("nan"))
    gt_es_area = gt_areas.get(es_orig, float("nan"))
    if np.isfinite(gt_ed_area) and gt_ed_area > 0 and np.isfinite(gt_es_area):
        ef_gt_area_pct = float(stage45.compute_ef_from_areas(gt_ed_area, gt_es_area) * 100.0)
    else:
        ef_gt_area_pct = float("nan")

    stage4_out = {
        "enabled": bool(run_stage4),
        "available": False,
        "error": None,
    }

    if run_stage4:
        if not stage4_checkpoint or not os.path.exists(stage4_checkpoint):
            stage4_out["error"] = f"Stage4 checkpoint not found: {stage4_checkpoint}"
        else:
            try:
                model4, meta4, incompat4 = load_stage4_model(
                    stage4_checkpoint,
                    fallback_model_name=stage4_model_name,
                    fallback_base_channels=int(stage4_base_channels),
                    device=device,
                )

                curve_frame_ids, curve_areas = _predict_area_curve_stage4_from_frames(
                    model4,
                    full_frames,
                    image_size=int(meta4["image_size"]),
                    normalize_mode=meta4["normalize"],
                    pretrained_flag=bool(meta4.get("pretrained", False)),
                    device=device,
                    eval_threshold=0.5,
                    batch_size=16,
                )

                if curve_frame_ids.size > 0:
                    detected = stage45.detect_ed_es_from_size_curve(
                        frame_ids=curve_frame_ids,
                        areas=curve_areas,
                        smooth_window=11,
                        enforce_es_after_ed=True,
                    )
                    curve_area_lookup = {int(fid): float(area) for fid, area in zip(curve_frame_ids.tolist(), curve_areas.tolist())}
                    ed_frame_idx = int(detected["ed_frame"])
                    es_frame_idx = int(detected["es_frame"])
                    pred_ed_area = float(curve_area_lookup.get(ed_frame_idx, float(np.max(curve_areas))))
                    pred_es_area = float(curve_area_lookup.get(es_frame_idx, float(np.min(curve_areas))))
                    pred_curve_method = "full_video_stage4_curve"
                else:
                    ed_frame_idx = int(pred_ed_orig)
                    es_frame_idx = int(pred_es_orig)
                    pred_curve_method = "stage123_frame_fallback"
                    _, pred_ed_area = _predict_mask_stage4(
                        model4,
                        full_frames[ed_frame_idx],
                        image_size=int(meta4["image_size"]),
                        normalize_mode=meta4["normalize"],
                        pretrained_flag=bool(meta4.get("pretrained", False)),
                        device=device,
                    )
                    _, pred_es_area = _predict_mask_stage4(
                        model4,
                        full_frames[es_frame_idx],
                        image_size=int(meta4["image_size"]),
                        normalize_mode=meta4["normalize"],
                        pretrained_flag=bool(meta4.get("pretrained", False)),
                        device=device,
                    )

                ed_frame_rgb, ed_frame_idx = _frame_from_list(full_frames, ed_frame_idx)
                es_frame_rgb, es_frame_idx = _frame_from_list(full_frames, es_frame_idx)

                pred_ed_mask, pred_ed_area_mask = _predict_mask_stage4(
                    model4,
                    ed_frame_rgb,
                    image_size=int(meta4["image_size"]),
                    normalize_mode=meta4["normalize"],
                    pretrained_flag=bool(meta4.get("pretrained", False)),
                    device=device,
                )
                pred_es_mask, pred_es_area_mask = _predict_mask_stage4(
                    model4,
                    es_frame_rgb,
                    image_size=int(meta4["image_size"]),
                    normalize_mode=meta4["normalize"],
                    pretrained_flag=bool(meta4.get("pretrained", False)),
                    device=device,
                )

                canonical_pair = _canonicalize_ed_es_pair_safe(
                    ed_frame_idx,
                    pred_ed_area,
                    es_frame_idx,
                    pred_es_area,
                )
                pred_pair_swapped = bool(canonical_pair["swapped"])
                if pred_pair_swapped:
                    ed_frame_idx, es_frame_idx = int(canonical_pair["ed_frame"]), int(canonical_pair["es_frame"])
                    pred_ed_area, pred_es_area = float(canonical_pair["ed_area"]), float(canonical_pair["es_area"])
                    pred_ed_mask, pred_es_mask = pred_es_mask, pred_ed_mask
                    pred_ed_area_mask, pred_es_area_mask = pred_es_area_mask, pred_ed_area_mask

                ef_stage5_pred_pct = float(stage45.compute_ef_from_areas(pred_ed_area, pred_es_area) * 100.0)
                seg_gif_bytes, seg_gif_err = _prepare_segmentation_gif(
                    full_frames,
                    model4,
                    meta4,
                    device,
                    max_frames=80,
                    max_width=640,
                )

                gt_mask_pred_ed = gt_masks.get(ed_frame_idx, gt_masks.get(int(pred_ed_orig)))
                gt_mask_pred_es = gt_masks.get(es_frame_idx, gt_masks.get(int(pred_es_orig)))
                gt_area_pred_ed = gt_areas.get(ed_frame_idx, float("nan"))
                gt_area_pred_es = gt_areas.get(es_frame_idx, float("nan"))

                stage4_out = {
                    "enabled": True,
                    "available": True,
                    "error": None,
                    "checkpoint_meta": meta4,
                    "incompatible": incompat4,
                    "pred_ed_frame_idx": int(ed_frame_idx),
                    "pred_es_frame_idx": int(es_frame_idx),
                    "pred_ed_mask": pred_ed_mask,
                    "pred_es_mask": pred_es_mask,
                    "pred_ed_area": float(pred_ed_area),
                    "pred_es_area": float(pred_es_area),
                    "pred_curve_method": pred_curve_method,
                    "pred_pair_swapped": bool(pred_pair_swapped),
                    "gt_mask_pred_ed": gt_mask_pred_ed,
                    "gt_mask_pred_es": gt_mask_pred_es,
                    "gt_area_pred_ed": gt_area_pred_ed,
                    "gt_area_pred_es": gt_area_pred_es,
                    "dice_pred_ed": _dice(pred_ed_mask, gt_mask_pred_ed),
                    "dice_pred_es": _dice(pred_es_mask, gt_mask_pred_es),
                    "ef_stage5_pred_pct": ef_stage5_pred_pct,
                    "seg_preview_gif": seg_gif_bytes,
                    "seg_preview_err": seg_gif_err,
                }
            except Exception as exc:
                stage4_out["error"] = str(exc)

    explanation = []
    explanation.append(
        f"Stage1-3 temporal inference mode: {temporal_settings['mode']} (margin={temporal_settings['margin_mult']:.2f})."
    )
    if stage4_out.get("available"):
        explanation.append(
            f"Stage5 EF from Stage4 masks = {stage4_out['ef_stage5_pred_pct']:.2f}%. Compare with Stage1-3 EF head = {ef_pred_pct:.2f}% and GT EF = {ef_gt_pct:.2f}%."
        )
        explanation.append(f"Stage4/5 frame selection method: {stage4_out.get('pred_curve_method', 'unknown')}.")
        if stage4_out.get("pred_pair_swapped"):
            explanation.append("Stage4 predicted areas were inverted on this case, so ED/ES were reordered before computing EF.")
    else:
        explanation.append("Stage4/5 result unavailable (missing checkpoint or load error).")


    return {
        "video_path": video_path,
        "fps": fps,
        "full_frames": full_frames,
        "sampled_indices": sampled_indices,
        "clip": clip,
        "ef_gt_pct": ef_gt_pct,
        "ef_pred_pct": ef_pred_pct,
        "ef_abs_err_pct": abs(ef_pred_pct - ef_gt_pct),
        "ed_orig": ed_orig,
        "es_orig": es_orig,
        "gt_ed_idx": gt_ed_idx,
        "gt_es_idx": gt_es_idx,
        "pred_ed_idx": pred_ed_idx,
        "pred_es_idx": pred_es_idx,
        "pred_ed_orig": pred_ed_orig,
        "pred_es_orig": pred_es_orig,
        "ed_err_sampled": ed_err_sampled,
        "es_err_sampled": es_err_sampled,
        "ed_err_orig": ed_err_orig,
        "es_err_orig": es_err_orig,
        "stage1_feat_norm": stage1_feat_norm,
        "stage1_temp_std": stage1_temp_std,
        "stage1_tokens": stage1_tokens,
        "stage2_attention": attn,
        "stage2_entropy": stage2_entropy,
        "stage2_peak": stage2_peak,
        "stage2_peak_idx": stage2_peak_idx,
        "stage2_peak_to_event": stage2_peak_to_event,
        "phase_probs": phase_probs,
        "phase_pred_ed": phase_pred_ed,
        "phase_pred_es": phase_pred_es,
        "gt_areas": gt_areas,
        "ef_gt_area_pct": ef_gt_area_pct,
        "stage4": stage4_out,
        "explanation": explanation,
        "incompatible_stage123": incompat123,
    }


def main():
    _inject_page_styles()
    st.title("CardioXplain: Dashboard")
    st.caption("Select a test video, run Stage1-5 inference, and inspect stage-wise outputs + metrics.")

    default_stage123_ckpt = _abs_path(getattr(config, "CHECKPOINT_PATH", "best_model_stage123_96f.pth"))
    default_stage4_ckpt = _abs_path(getattr(config, "STAGE4_CHECKPOINT_PATH", "best_stage4_segmentation_area.pth"))

    with st.sidebar:
        st.header("Run Settings")
        data_dir = st.text_input("Data directory", value=_abs_path(config.DATA_DIR))
        split = st.selectbox("Dataset split", options=["TEST", "VAL", "TRAIN"], index=0)
        num_frames = st.number_input("Stage1-3 num frames", min_value=8, max_value=128, value=int(getattr(config, "NUM_FRAMES", 96)), step=8)

        device_choice = st.selectbox("Device", options=["auto", "cuda", "cpu"], index=0)
        device = _resolve_device(device_choice)

        st.subheader("Stage1-3")
        stage123_checkpoint = st.text_input("Stage1-3 checkpoint", value=default_stage123_ckpt)

        st.subheader("Stage4/5")
        run_stage4 = st.checkbox("Run Stage4 + Stage5", value=True)
        stage4_checkpoint = st.text_input("Stage4 checkpoint", value=default_stage4_ckpt)
        stage4_model_name = st.selectbox("Stage4 fallback model", options=["deeplabv3_resnet50", "fcn_resnet50", "unet"], index=0)
        stage4_base_channels = st.number_input("Stage4 UNet base channels", min_value=8, max_value=128, value=32, step=8)

        st.subheader("Stage6/7")
        show_stage67 = st.checkbox("Show Stage6/7 clinical summary", value=True)
        stage67_output_dir = st.text_input("Stage6/7 output dir", value=_abs_path(os.path.join("validation", "outputs", "stage67")))

    if not os.path.exists(data_dir):
        st.error(f"Data directory not found: {data_dir}")
        return

    try:
        filelist = load_split_filelist(data_dir, split)
    except Exception as exc:
        st.error(f"Failed to load split metadata: {exc}")
        return

    if filelist.empty:
        st.warning(f"No videos found for split {split} in FileList.csv")
        return

    video_options = filelist["FileName"].astype(str).tolist()
    selected_video = st.selectbox("Select video from split", options=video_options, index=0)

    run_now = st.button("Run Inference", type="primary")
    if not run_now:
        st.info("Choose a video and click Run Inference.")
        return

    if not os.path.exists(stage123_checkpoint):
        st.error(f"Stage1-3 checkpoint not found: {stage123_checkpoint}")
        return

    with st.spinner("Running all stages..."):
        try:
            result = run_case(
                data_dir=data_dir,
                split=split,
                video_name=selected_video,
                stage123_checkpoint=stage123_checkpoint,
                num_frames=int(num_frames),
                device=device,
                run_stage4=bool(run_stage4),
                stage4_checkpoint=stage4_checkpoint,
                stage4_model_name=stage4_model_name,
                stage4_base_channels=int(stage4_base_channels),
            )
        except Exception as exc:
            st.error(f"Inference failed: {exc}")
            return

    st.subheader("Source Video")
    st.write(f"Path: `{result['video_path']}`")

    if not _render_temporal_weight_video(result):
        gif_bytes, gif_err = _prepare_gif_preview(result["video_path"])
        if gif_bytes:
            _render_centered_image(gif_bytes, "Animated source preview")
        else:
            st.caption(f"Animated preview unavailable: {gif_err}")

    st.subheader("Key Metrics")
    c1, c2, c3 = st.columns(3)
    c1.metric("GT EF (%)", f"{result['ef_gt_pct']:.2f}")
    c2.metric("Pred EF (%)", f"{result['ef_pred_pct']:.2f}")
    c3.metric("EF Abs Error (%)", f"{result['ef_abs_err_pct']:.2f}")

    st.subheader("Stage 1")
    s1c1, s1c2 = st.columns(2)
    s1c1.metric("Feature Norm", f"{result['stage1_feat_norm']:.3f}")
    s1c2.metric("Temporal Std", f"{result['stage1_temp_std']:.3f}")
    st.caption(f"Temporal tokens after Stage1 pooling (T'): {result['stage1_tokens']}")


   

    st.subheader("Stage 4 + Stage 5")
    if not result["stage4"].get("enabled"):
        st.info("Stage4/5 execution disabled in sidebar.")
    elif not result["stage4"].get("available"):
        st.warning(result["stage4"].get("error", "Stage4 unavailable"))
    else:
        s4 = result["stage4"]
        col_ed, col_es = st.columns(2)

        pred_ed_frame_rgb, _ = _frame_from_list(result["full_frames"], s4["pred_ed_frame_idx"])
        pred_es_frame_rgb, _ = _frame_from_list(result["full_frames"], s4["pred_es_frame_idx"])

        col_ed.image(
            _overlay_mask_rgb(pred_ed_frame_rgb, s4["pred_ed_mask"], color=(0, 255, 0), alpha=0.35),
            caption=f"Pred ED frame {s4['pred_ed_frame_idx']} + predicted mask",
            width=FRAME_DISPLAY_WIDTH,
        )
        col_es.image(
            _overlay_mask_rgb(pred_es_frame_rgb, s4["pred_es_mask"], color=(255, 165, 0), alpha=0.35),
            caption=f"Pred ES frame {s4['pred_es_frame_idx']} + predicted mask",
            width=FRAME_DISPLAY_WIDTH,
        )

        stage45_table = pd.DataFrame(
            [
                {"metric": "Pred ED area (px)", "value": _format_display_number(s4["pred_ed_area"], digits=2)},
                {"metric": "Pred ES area (px)", "value": _format_display_number(s4["pred_es_area"], digits=2)},
                {"metric": "Stage5 EF from predicted masks (%)", "value": _format_display_number(s4["ef_stage5_pred_pct"], digits=2)},
                {"metric": "GT EF from traced masks (%)", "value": _format_display_number(result["ef_gt_area_pct"], digits=2)},
                {"metric": "GT ED area at predicted-ED frame (px)", "value": _format_display_number(s4["gt_area_pred_ed"], digits=2)},
                {"metric": "GT ES area at predicted-ES frame (px)", "value": _format_display_number(s4["gt_area_pred_es"], digits=2)},
                {"metric": "Dice on predicted-ED frame", "value": _format_display_number(s4["dice_pred_ed"], digits=4)},
                {"metric": "Dice on predicted-ES frame", "value": _format_display_number(s4["dice_pred_es"], digits=4)},
                {"metric": "Stage4/5 frame selection", "value": str(s4.get("pred_curve_method", "NA"))},
            ]
        )
        st.dataframe(stage45_table, width="stretch", hide_index=True)
        seg_bytes = s4.get("seg_preview_gif")
        if seg_bytes:
            _render_centered_image(seg_bytes, "Segmentation overlay animation")
        else:
            st.caption(s4.get("seg_preview_err", "Segmentation preview unavailable."))

    if show_stage67:
        _render_stage67_section(selected_video=selected_video, split=split, stage67_output_dir=stage67_output_dir)

    st.subheader("Auto Explanation")
    for line in result["explanation"]:
        st.write(f"- {line}")

    if result["incompatible_stage123"].missing_keys or result["incompatible_stage123"].unexpected_keys:
        st.warning(
            "Stage1-3 checkpoint loaded with key mismatch. "
            f"Missing={len(result['incompatible_stage123'].missing_keys)}, "
            f"Unexpected={len(result['incompatible_stage123'].unexpected_keys)}"
        )


if __name__ == "__main__":
    main()


