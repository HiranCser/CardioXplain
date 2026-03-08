import os
import sys
import math

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
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
                    device=device,
                )
                pred_es_mask, pred_es_area = _predict_mask_stage4(
                    model4,
                    es_frame_rgb,
                    image_size=int(meta4["image_size"]),
                    normalize_mode=meta4["normalize"],
                    device=device,
                )

                ef_stage5_pred_pct = float(stage45.compute_ef_from_areas(pred_ed_area, pred_es_area) * 100.0)

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
                }
            except Exception as exc:
                stage4_out["error"] = str(exc)

    explanation = []
    tol = int(getattr(config, "TOLERANCE", 1))
    if ed_err_sampled <= tol and es_err_sampled <= tol:
        explanation.append(f"Stage3 is within tolerance (+/-{tol}) on sampled ED/ES indices.")
    else:
        explanation.append(
            f"Stage3 misses tolerance (+/-{tol}): ED err={ed_err_sampled}, ES err={es_err_sampled} (sampled index)."
        )

    explanation.append(
        f"Stage2 attention peak is at sampled frame {stage2_peak_idx} with weight {stage2_peak:.3f}; closest ED/ES distance is {stage2_peak_to_event:.2f} frames."
    )

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
    st.title("CardioXplain: One-Page Stage Dashboard")
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
    st.video(result["video_path"])

    st.subheader("Key Metrics")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("GT EF (%)", f"{result['ef_gt_pct']:.2f}")
    c2.metric("Pred EF (%)", f"{result['ef_pred_pct']:.2f}")
    c3.metric("EF Abs Error (%)", f"{result['ef_abs_err_pct']:.2f}")
    c4.metric("Stage2 Entropy", f"{result['stage2_entropy']:.3f}")

    c5, c6, c7, c8 = st.columns(4)
    c5.metric("ED Error (sampled)", f"{result['ed_err_sampled']}")
    c6.metric("ES Error (sampled)", f"{result['es_err_sampled']}")
    c7.metric("Pred ED (orig frame)", f"{result['pred_ed_orig']}")
    c8.metric("Pred ES (orig frame)", f"{result['pred_es_orig']}")

    st.subheader("Stage 1")
    s1c1, s1c2 = st.columns(2)
    s1c1.metric("Feature Norm", f"{result['stage1_feat_norm']:.3f}")
    s1c2.metric("Temporal Std", f"{result['stage1_temp_std']:.3f}")
    st.caption(f"Temporal tokens after Stage1 pooling (T'): {result['stage1_tokens']}")

    st.subheader("Stage 2")
    fig_attn = _make_attention_plot(
        result["stage2_attention"],
        result["gt_ed_idx"],
        result["gt_es_idx"],
        result["pred_ed_idx"],
        result["pred_es_idx"],
    )
    st.pyplot(fig_attn)
    plt.close(fig_attn)

    st.write(
        {
            "peak_index": result["stage2_peak_idx"],
            "peak_weight": round(result["stage2_peak"], 4),
            "peak_to_nearest_event_frames": round(result["stage2_peak_to_event"], 3),
        }
    )

    st.subheader("Stage 3")
    fig_phase = _make_phase_plot(
        result["phase_probs"],
        result["gt_ed_idx"],
        result["gt_es_idx"],
        result["pred_ed_idx"],
        result["pred_es_idx"],
    )
    st.pyplot(fig_phase)
    plt.close(fig_phase)

    s3_table = pd.DataFrame(
        [
            {
                "metric": "GT ED idx (sampled)",
                "value": result["gt_ed_idx"],
            },
            {
                "metric": "GT ES idx (sampled)",
                "value": result["gt_es_idx"],
            },
            {
                "metric": "Pred ED idx (sampled)",
                "value": result["pred_ed_idx"],
            },
            {
                "metric": "Pred ES idx (sampled)",
                "value": result["pred_es_idx"],
            },
            {
                "metric": "ED abs err (sampled idx)",
                "value": result["ed_err_sampled"],
            },
            {
                "metric": "ES abs err (sampled idx)",
                "value": result["es_err_sampled"],
            },
            {
                "metric": "ED abs err (orig frame)",
                "value": result["ed_err_orig"],
            },
            {
                "metric": "ES abs err (orig frame)",
                "value": result["es_err_orig"],
            },
        ]
    )
    st.dataframe(s3_table, use_container_width=True, hide_index=True)

    st.subheader("Frames and Overlays")
    view_idx = st.slider("Inspect original frame", 0, len(result["full_frames"]) - 1, int(result["pred_ed_orig"]))
    frame_view, _ = _frame_from_list(result["full_frames"], view_idx)
    st.image(frame_view, caption=f"Original frame {view_idx}", use_column_width=True)

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
            use_column_width=True,
        )
        col_es.image(
            _overlay_mask_rgb(pred_es_frame_rgb, s4["pred_es_mask"], color=(255, 165, 0), alpha=0.35),
            caption=f"Pred ES frame {s4['pred_es_frame_idx']} + predicted mask",
            use_column_width=True,
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
