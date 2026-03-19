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
PREVIEW_MAX_WIDTH = 420
PREVIEW_DISPLAY_WIDTH = 600
FRAME_DISPLAY_WIDTH = 360


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
                    radial-gradient(circle at top left, rgba(30, 136, 229, 0.10), transparent 28%),
                    radial-gradient(circle at top right, rgba(16, 185, 129, 0.08), transparent 24%),
                    linear-gradient(180deg, #f5f9ff 0%, #edf4fb 48%, #f8fbff 100%);
            }
            .block-container {
                padding-top: 1.4rem;
                padding-bottom: 2.5rem;
                max-width: 1320px;
            }
            [data-testid="stSidebar"] {
                background: linear-gradient(180deg, #f7fbff 0%, #eef5fd 100%);
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
                background: rgba(255, 255, 255, 0.88);
                border: 1px solid rgba(148, 163, 184, 0.22);
                border-radius: 18px;
                padding: 0.85rem 1rem;
                box-shadow: 0 10px 24px rgba(15, 23, 42, 0.06);
            }
            [data-testid="stMetricLabel"] {
                color: #5b7690;
                font-weight: 600;
                letter-spacing: 0.04em;
                text-transform: uppercase;
            }
            [data-testid="stMetricValue"] {
                color: #102a43;
                font-weight: 800;
            }
            div[data-testid="stDataFrame"] {
                border-radius: 18px;
                overflow: hidden;
                border: 1px solid rgba(148, 163, 184, 0.20);
                box-shadow: 0 12px 24px rgba(15, 23, 42, 0.05);
                background: rgba(255, 255, 255, 0.92);
            }
            [data-testid="stImage"] img {
                border-radius: 18px;
                box-shadow: 0 14px 34px rgba(15, 23, 42, 0.10);
            }
            [data-testid="stCaptionContainer"] {
                color: #587086;
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


@st.cache_resource(show_spinner=False)
def load_stage123_model(checkpoint_path, num_frames, device):
    model = EFModel(num_frames=int(num_frames)).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict, _ = _safe_checkpoint_state_dict(checkpoint)
    incompatible = model.load_state_dict(state_dict, strict=False)
    model.eval()
    return model, incompatible


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


@st.cache_resource(show_spinner=False)
def load_dataset_resource(data_dir, split, num_frames):
    return EchoDataset(
        data_dir=data_dir,
        split=str(split).upper(),
        num_frames=int(num_frames),
        normalize_input=bool(getattr(config, "NORMALIZE_INPUT", True)),
        temporal_window_mode=str(getattr(config, "PHASE_TEMPORAL_WINDOW_MODE", "full")),
        temporal_window_margin_mult=float(getattr(config, "PHASE_TEMPORAL_WINDOW_MARGIN_MULT", 1.5)),
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


def _frame_from_list(frames, idx):
    if len(frames) == 0:
        return None, -1
    idx_clamped = int(np.clip(int(idx), 0, len(frames) - 1))
    return frames[idx_clamped], idx_clamped


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
    weights = np.asarray(attn, dtype=np.float64).reshape(-1)
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
    sampled_weights = np.asarray(result["stage2_attention"], dtype=np.float64).reshape(-1)
    sample_count = min(sampled_indices.size, sampled_weights.size)
    peak_frame = int(np.argmax(frame_weights)) if frame_weights.size > 0 else 0
    peak_weight = float(frame_weights[peak_frame]) if frame_weights.size > 0 else 0.0

    playback_col, summary_col = st.columns([1.65, 1.0])
    with playback_col:
        gif_bytes, gif_err = _prepare_gif_preview(result["video_path"])
        if gif_bytes:
            st.image(gif_bytes, caption="Animated source preview", use_container_width=True)
        else:
            st.caption(f"Animated preview unavailable: {gif_err}")
    with summary_col:
        st.metric("Peak Weight", f"{peak_weight:.4f}")
        st.metric("Peak Frame", str(peak_frame))
        st.metric("Frames", str(total_frames))
        st.caption("Only the animated preview is shown here. The graph below stays available for frame-wise temporal-weight inspection.")

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
        "totalFrames": int(total_frames),
    }

    component_id = hashlib.md5(
        f"{result['video_path']}|{total_frames}|{result['fps']}".encode("utf-8")
    ).hexdigest()[:12]
    payload_json = json.dumps(payload)

    components.html(
        f"""
        <div id="tw-{component_id}" style="font-family: 'Trebuchet MS', 'Segoe UI', sans-serif; color: #102a43;">
          <style>
            #tw-{component_id} {{
              --tw-ink: #102a43;
              --tw-muted: #587086;
              --tw-line: #245fd9;
              --tw-sample: #f97316;
              --tw-ed: #16a34a;
              --tw-es: #dc2626;
              --tw-border: rgba(148, 163, 184, 0.22);
            }}
            #tw-{component_id} .tw-shell {{
              background: linear-gradient(180deg, rgba(255,255,255,0.92) 0%, rgba(240,247,255,0.94) 100%);
              border: 1px solid var(--tw-border);
              border-radius: 26px;
              padding: 18px;
              box-shadow: 0 18px 40px rgba(15, 23, 42, 0.08);
            }}
            #tw-{component_id} .tw-header {{
              display: flex;
              justify-content: space-between;
              align-items: flex-end;
              gap: 12px;
              margin-bottom: 14px;
            }}
            #tw-{component_id} .tw-title {{
              margin: 0;
              font-size: 1.25rem;
              font-weight: 800;
              color: var(--tw-ink);
            }}
            #tw-{component_id} .tw-subtitle {{
              margin: 4px 0 0 0;
              color: var(--tw-muted);
              font-size: 0.92rem;
              line-height: 1.45;
            }}
            #tw-{component_id} .tw-chip-row {{
              display: flex;
              flex-wrap: wrap;
              gap: 8px;
              justify-content: flex-end;
            }}
            #tw-{component_id} .tw-chip {{
              border-radius: 999px;
              padding: 8px 12px;
              background: rgba(255,255,255,0.92);
              border: 1px solid var(--tw-border);
              color: var(--tw-muted);
              font-size: 12px;
              font-weight: 700;
              letter-spacing: 0.03em;
            }}
            #tw-{component_id} .tw-grid {{
              display: grid;
              grid-template-columns: repeat(4, minmax(0, 1fr));
              gap: 10px;
              margin-bottom: 14px;
            }}
            #tw-{component_id} .tw-stat {{
              background: rgba(255,255,255,0.88);
              border: 1px solid var(--tw-border);
              border-radius: 16px;
              padding: 12px 14px;
              box-shadow: inset 0 1px 0 rgba(255,255,255,0.65);
            }}
            #tw-{component_id} .tw-stat-label {{
              font-size: 11px;
              text-transform: uppercase;
              letter-spacing: 0.08em;
              color: #6b8297;
              margin-bottom: 5px;
            }}
            #tw-{component_id} .tw-stat-value {{
              font-size: 1.14rem;
              font-weight: 800;
              color: var(--tw-ink);
            }}
            #tw-{component_id} .tw-chart-card {{
              background: radial-gradient(circle at top right, rgba(36,95,217,0.08), transparent 26%), #ffffff;
              border: 1px solid var(--tw-border);
              border-radius: 22px;
              padding: 14px;
            }}
            #tw-{component_id} .tw-chart {{
              width: 100%;
              height: 310px;
              display: block;
            }}
            #tw-{component_id} .tw-slider-row {{
              margin-top: 10px;
            }}
            #tw-{component_id} .tw-slider-label {{
              display: flex;
              justify-content: space-between;
              color: var(--tw-muted);
              font-size: 12px;
              margin-bottom: 6px;
            }}
            #tw-{component_id} .tw-slider {{
              width: 100%;
              accent-color: #245fd9;
            }}
            #tw-{component_id} .tw-legend {{
              display: flex;
              flex-wrap: wrap;
              gap: 14px;
              margin-top: 12px;
              color: var(--tw-muted);
              font-size: 12px;
            }}
            #tw-{component_id} .tw-legend-item {{
              display: inline-flex;
              align-items: center;
              gap: 7px;
            }}
            #tw-{component_id} .tw-swatch {{
              width: 12px;
              height: 12px;
              border-radius: 999px;
              display: inline-block;
            }}
            @media (max-width: 900px) {{
              #tw-{component_id} .tw-grid {{
                grid-template-columns: repeat(2, minmax(0, 1fr));
              }}
              #tw-{component_id} .tw-header {{
                flex-direction: column;
                align-items: flex-start;
              }}
              #tw-{component_id} .tw-chip-row {{
                justify-content: flex-start;
              }}
            }}
          </style>
          <div class="tw-shell">
            <div class="tw-header">
              <div>
                <div class="tw-title">Temporal Weight Graph</div>
                <div class="tw-subtitle">Use the native video player above for playback. This graph remains available for frame-by-frame temporal-weight inspection.</div>
              </div>
              <div class="tw-chip-row">
                <span class="tw-chip">{payload['totalFrames']} frames</span>
                <span class="tw-chip">Peak frame {payload['peakFrame']}</span>
                <span class="tw-chip">Peak weight {payload['peakWeight']:.4f}</span>
              </div>
            </div>
            <div class="tw-grid">
              <div class="tw-stat">
                <div class="tw-stat-label">Selected frame</div>
                <div class="tw-stat-value" id="tw-frame-{component_id}">0 / {payload['totalFrames'] - 1}</div>
              </div>
              <div class="tw-stat">
                <div class="tw-stat-label">Temporal weight</div>
                <div class="tw-stat-value" id="tw-weight-{component_id}">0.0000</div>
              </div>
              <div class="tw-stat">
                <div class="tw-stat-label">Ground truth ED/ES</div>
                <div class="tw-stat-value">{int(result['ed_orig'])} / {int(result['es_orig'])}</div>
              </div>
              <div class="tw-stat">
                <div class="tw-stat-label">Predicted ED/ES</div>
                <div class="tw-stat-value">{int(result['pred_ed_orig'])} / {int(result['pred_es_orig'])}</div>
              </div>
            </div>
            <div class="tw-chart-card">
              <svg id="tw-chart-{component_id}" class="tw-chart" viewBox="0 0 820 310" preserveAspectRatio="none"></svg>
              <div class="tw-slider-row">
                <div class="tw-slider-label">
                  <span>Inspect graph by frame</span>
                  <span id="tw-slider-readout-{component_id}">Frame 0</span>
                </div>
                <input id="tw-slider-{component_id}" class="tw-slider" type="range" min="0" max="{max(0, payload['totalFrames'] - 1)}" value="0" step="1" />
              </div>
              <div class="tw-legend">
                <span class="tw-legend-item"><span class="tw-swatch" style="background:#245fd9;"></span>Temporal weight curve</span>
                <span class="tw-legend-item"><span class="tw-swatch" style="background:#f97316;"></span>Sampled frames</span>
                <span class="tw-legend-item"><span class="tw-swatch" style="background:#16a34a;"></span>ED markers</span>
                <span class="tw-legend-item"><span class="tw-swatch" style="background:#dc2626;"></span>ES markers</span>
              </div>
            </div>
          </div>
        </div>
        <script>
          const payload = {payload_json};
          const svg = document.getElementById("tw-chart-{component_id}");
          const frameLabel = document.getElementById("tw-frame-{component_id}");
          const weightLabel = document.getElementById("tw-weight-{component_id}");
          const slider = document.getElementById("tw-slider-{component_id}");
          const sliderReadout = document.getElementById("tw-slider-readout-{component_id}");
          const width = 820;
          const height = 310;
          const padLeft = 58;
          const padRight = 20;
          const padTop = 22;
          const padBottom = 42;
          const graphHeight = height - padTop - padBottom;
          const graphWidth = width - padLeft - padRight;
          const maxWeight = Math.max(...payload.frameWeights, 1e-6);

          function xForFrame(frame) {{
            if (payload.totalFrames <= 1) {{
              return padLeft + graphWidth / 2;
            }}
            return padLeft + (frame / (payload.totalFrames - 1)) * graphWidth;
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

          function eventMarkup(frame, color, dash, label, row) {{
            const x = xForFrame(frame);
            const labelWidth = Math.max(62, label.length * 7.4 + 20);
            const rectX = Math.min(width - labelWidth - 8, Math.max(8, x - labelWidth / 2));
            const rectY = padTop + 8 + row * 20;
            return `
              <line x1="${{x}}" y1="${{padTop}}" x2="${{x}}" y2="${{padTop + graphHeight}}" stroke="${{color}}" stroke-width="2.2" stroke-dasharray="${{dash}}" opacity="0.85"></line>
              <rect x="${{rectX}}" y="${{rectY}}" width="${{labelWidth}}" height="18" rx="9" fill="white" fill-opacity="0.92" stroke="${{color}}" stroke-opacity="0.35"></rect>
              <text x="${{rectX + labelWidth / 2}}" y="${{rectY + 12.5}}" fill="${{color}}" text-anchor="middle" font-size="11.5" font-weight="800">${{label}}</text>
            `;
          }}

          const points = payload.frameWeights.map((weight, frame) => [xForFrame(frame), yForWeight(weight)]);
          const linePath = buildLinePath(points);
          const areaPath = buildAreaPath(points);
          const sampledDots = payload.sampledIndices.map((frame, idx) => `
            <circle cx="${{xForFrame(frame)}}" cy="${{yForWeight(payload.sampledWeights[idx] ?? 0)}}" r="4.2" fill="#f97316" stroke="white" stroke-width="1.4"></circle>
          `).join('');
          const xTickFrames = [0, Math.max(0, Math.floor((payload.totalFrames - 1) / 2)), Math.max(0, payload.totalFrames - 1)];
          const xTicks = xTickFrames.map((frame) => `
            <line x1="${{xForFrame(frame)}}" y1="${{padTop + graphHeight}}" x2="${{xForFrame(frame)}}" y2="${{padTop + graphHeight + 6}}" stroke="#9db2c7"></line>
            <text x="${{xForFrame(frame)}}" y="${{height - 10}}" fill="#627d98" text-anchor="middle" font-size="11.5">${{frame}}</text>
          `).join('');
          const yTickValues = [0, maxWeight / 2, maxWeight];
          const yTicks = yTickValues.map((weight) => `
            <line x1="${{padLeft}}" y1="${{yForWeight(weight)}}" x2="${{width - padRight}}" y2="${{yForWeight(weight)}}" stroke="#e5edf6"></line>
            <text x="14" y="${{yForWeight(weight) + 4}}" fill="#627d98" font-size="11.5">${{weight.toFixed(3)}}</text>
          `).join('');

          svg.innerHTML = `
            <defs>
              <linearGradient id="tw-fill-{component_id}" x1="0" y1="0" x2="0" y2="1">
                <stop offset="0%" stop-color="#245fd9" stop-opacity="0.28"></stop>
                <stop offset="100%" stop-color="#245fd9" stop-opacity="0.02"></stop>
              </linearGradient>
              <filter id="tw-glow-{component_id}" x="-10%" y="-10%" width="120%" height="120%">
                <feDropShadow dx="0" dy="5" stdDeviation="6" flood-color="#245fd9" flood-opacity="0.20"></feDropShadow>
              </filter>
            </defs>
            <rect x="0" y="0" width="820" height="310" rx="18" fill="#ffffff"></rect>
            ${{yTicks}}
            <path d="${{areaPath}}" fill="url(#tw-fill-{component_id})"></path>
            <path d="${{linePath}}" fill="none" stroke="#245fd9" stroke-width="4" stroke-linecap="round" stroke-linejoin="round" filter="url(#tw-glow-{component_id})"></path>
            ${{sampledDots}}
            ${{eventMarkup(payload.gtEd, '#16a34a', '5 5', 'GT ED', 0)}}
            ${{eventMarkup(payload.gtEs, '#dc2626', '5 5', 'GT ES', 1)}}
            ${{eventMarkup(payload.predEd, '#16a34a', '2 5', 'Pred ED', 2)}}
            ${{eventMarkup(payload.predEs, '#dc2626', '2 5', 'Pred ES', 3)}}
            <line x1="${{padLeft}}" y1="${{padTop + graphHeight}}" x2="${{width - padRight}}" y2="${{padTop + graphHeight}}" stroke="#9db2c7" stroke-width="1.3"></line>
            <line x1="${{padLeft}}" y1="${{padTop}}" x2="${{padLeft}}" y2="${{padTop + graphHeight}}" stroke="#9db2c7" stroke-width="1.3"></line>
            ${{xTicks}}
            <text x="${{width / 2}}" y="${{height - 4}}" fill="#486581" text-anchor="middle" font-size="12.5" font-weight="700">Frame index</text>
            <text x="14" y="16" fill="#486581" font-size="12.5" font-weight="700">Weight</text>
            <line id="tw-marker-{component_id}" x1="${{padLeft}}" y1="${{padTop}}" x2="${{padLeft}}" y2="${{padTop + graphHeight}}" stroke="#0f172a" stroke-width="2.7"></line>
            <circle id="tw-marker-dot-{component_id}" cx="${{padLeft}}" cy="${{yForWeight(payload.frameWeights[0] || 0)}}" r="6" fill="#0f172a" stroke="white" stroke-width="2"></circle>
          `;

          const marker = document.getElementById("tw-marker-{component_id}");
          const markerDot = document.getElementById("tw-marker-dot-{component_id}");

          function updateMarker(frame) {{
            const clampedFrame = Math.max(0, Math.min(payload.totalFrames - 1, frame));
            const x = xForFrame(clampedFrame);
            const weight = payload.frameWeights[clampedFrame] || 0;
            marker.setAttribute('x1', x);
            marker.setAttribute('x2', x);
            markerDot.setAttribute('cx', x);
            markerDot.setAttribute('cy', yForWeight(weight));
            frameLabel.textContent = `${{clampedFrame}} / ${{Math.max(0, payload.totalFrames - 1)}}`;
            weightLabel.textContent = weight.toFixed(4);
            slider.value = String(clampedFrame);
            sliderReadout.textContent = `Frame ${{clampedFrame}}`;
          }}

          slider.addEventListener('input', (event) => {{
            updateMarker(Number(event.target.value || 0));
          }});

          updateMarker(0);
        </script>
        """,
        height=560,
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

    dataset = load_dataset_resource(data_dir=data_dir, split=split, num_frames=int(num_frames))
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

    model123, incompat123 = load_stage123_model(stage123_checkpoint, int(num_frames), device)

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

    phase_pred_ed = int(np.argmax(phase_probs[:, 0]))
    phase_pred_es = int(np.argmax(phase_probs[:, 1]))

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

                ed_frame_rgb, ed_frame_idx = _frame_from_list(full_frames, pred_ed_orig)
                es_frame_rgb, es_frame_idx = _frame_from_list(full_frames, pred_es_orig)

                pred_ed_mask, pred_ed_area = _predict_mask_stage4(
                    model4,
                    ed_frame_rgb,
                    image_size=int(meta4["image_size"]),
                    normalize_mode=meta4["normalize"],
                    pretrained_flag=bool(meta4.get("pretrained", False)),
                    device=device,
                )
                pred_es_mask, pred_es_area = _predict_mask_stage4(
                    model4,
                    es_frame_rgb,
                    image_size=int(meta4["image_size"]),
                    normalize_mode=meta4["normalize"],
                    pretrained_flag=bool(meta4.get("pretrained", False)),
                    device=device,
                )

                ef_stage5_pred_pct = float(stage45.compute_ef_from_areas(pred_ed_area, pred_es_area) * 100.0)
                seg_gif_bytes, seg_gif_err = _prepare_segmentation_gif(
                    full_frames,
                    model4,
                    meta4,
                    device,
                    max_frames=80,
                    max_width=640,
                )

                gt_mask_pred_ed = gt_masks.get(ed_frame_idx)
                gt_mask_pred_es = gt_masks.get(es_frame_idx)
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
    if stage4_out.get("available"):
        explanation.append(
            f"Stage5 EF from Stage4 masks = {stage4_out['ef_stage5_pred_pct']:.2f}%. Compare with Stage1-3 EF head = {ef_pred_pct:.2f}% and GT EF = {ef_gt_pct:.2f}%."
        )
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

    default_stage123_ckpt = _abs_path(getattr(config, "CHECKPOINT_PATH", "best_model.pth"))
    default_stage4_ckpt = _abs_path("best_stage4_segmentation.pth")

    with st.sidebar:
        st.header("Run Settings")
        data_dir = st.text_input("Data directory", value=_abs_path(config.DATA_DIR))
        split = st.selectbox("Dataset split", options=["TEST", "VAL", "TRAIN"], index=0)
        num_frames = st.number_input("Stage1-3 num frames", min_value=8, max_value=128, value=int(getattr(config, "NUM_FRAMES", 32)), step=8)

        device_choice = st.selectbox("Device", options=["auto", "cuda", "cpu"], index=0)
        device = _resolve_device(device_choice)

        st.subheader("Stage1-3")
        stage123_checkpoint = st.text_input("Stage1-3 checkpoint", value=default_stage123_ckpt)

        st.subheader("Stage4/5")
        run_stage4 = st.checkbox("Run Stage4 + Stage5", value=True)
        stage4_checkpoint = st.text_input("Stage4 checkpoint", value=default_stage4_ckpt)
        stage4_model_name = st.selectbox("Stage4 fallback model", options=["deeplabv3_resnet50", "fcn_resnet50", "unet"], index=0)
        stage4_base_channels = st.number_input("Stage4 UNet base channels", min_value=8, max_value=128, value=32, step=8)

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
                {"metric": "Pred ED area (px)", "value": round(float(s4["pred_ed_area"]), 2)},
                {"metric": "Pred ES area (px)", "value": round(float(s4["pred_es_area"]), 2)},
                {"metric": "Stage5 EF from predicted masks (%)", "value": round(float(s4["ef_stage5_pred_pct"]), 2)},
                {"metric": "GT EF from traced masks (%)", "value": round(float(result["ef_gt_area_pct"]), 2) if np.isfinite(result["ef_gt_area_pct"]) else "NA"},
                {"metric": "GT ED area at predicted-ED frame (px)", "value": round(float(s4["gt_area_pred_ed"]), 2) if np.isfinite(s4["gt_area_pred_ed"]) else "NA"},
                {"metric": "GT ES area at predicted-ES frame (px)", "value": round(float(s4["gt_area_pred_es"]), 2) if np.isfinite(s4["gt_area_pred_es"]) else "NA"},
                {"metric": "Dice on predicted-ED frame", "value": round(float(s4["dice_pred_ed"]), 4) if np.isfinite(s4["dice_pred_ed"]) else "NA"},
                {"metric": "Dice on predicted-ES frame", "value": round(float(s4["dice_pred_es"]), 4) if np.isfinite(s4["dice_pred_es"]) else "NA"},
            ]
        )
        st.dataframe(stage45_table, use_container_width=True, hide_index=True)
        seg_bytes = s4.get("seg_preview_gif")
        if seg_bytes:
            _render_centered_image(seg_bytes, "Segmentation overlay animation")
        else:
            st.caption(s4.get("seg_preview_err", "Segmentation preview unavailable."))

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


