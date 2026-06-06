import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import config
from data.dataset import EchoDataset
from models.ef_model import EFModel, infer_ef_head_arch
from pipeline.stage3_phase_detector import Stage3PhaseDetector
from pipeline.stage45_pipeline import Stage45Pipeline
from pipeline.stage67_similarity import (
    LABEL_TO_TEXT,
    Stage6SimilarityEngine,
    Stage7UncertaintyCalibrator,
    accuracy_np,
    confusion_matrix_np,
    ef_to_severity_label,
    macro_f1_np,
    softmax_np,
)


FEATURE_COLUMNS = [
    "ef_stage123_pct",
    "ef_stage5_pct",
    "ef_disagreement_pct",
    "attention_entropy",
    "attention_peak",
    "phase_ed_conf",
    "phase_es_conf",
    "pred_gap_norm",
    "ed_trace_offset",
    "es_trace_offset",
]


class Stage6MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, dropout=0.1, n_classes=3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(int(input_dim), int(hidden_dim)),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(int(hidden_dim), int(hidden_dim)),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(int(hidden_dim), int(n_classes)),
        )

    def forward(self, x):
        return self.net(x)


def parse_args():
    parser = argparse.ArgumentParser(description="Train Stage6 similarity/MLP and Stage7 uncertainty calibration using Stage1-5 outputs.")
    parser.add_argument("--data-dir", type=str, default=config.DATA_DIR)
    parser.add_argument("--stage123-checkpoint", type=str, default=getattr(config, "CHECKPOINT_PATH", "best_model.pth"))
    parser.add_argument("--num-frames", type=int, default=int(getattr(config, "NUM_FRAMES", 32)))
    parser.add_argument("--max-videos", type=int, default=None, help="Optional cap per split for faster runs")
    parser.add_argument("--device", type=str, default=("cuda" if torch.cuda.is_available() else "cpu"))
    parser.add_argument("--normal-threshold", type=float, default=50.0, help="EF >= threshold -> normal class")
    parser.add_argument("--severe-threshold", type=float, default=30.0, help="EF < threshold -> severe class")
    parser.add_argument("--output-dir", type=str, default=os.path.join("validation", "outputs", "stage67"))
    parser.add_argument("--clip-period", type=int, default=int(getattr(config, "CLIP_PERIOD", 1)))
    parser.add_argument("--clip-eval-mode", type=str, choices=["center", "all"], default="all")
    parser.add_argument("--clip-batch-size", type=int, default=8)
    parser.add_argument("--save-per-split-csv", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--fit-split", type=str, default="TRAIN", choices=["TRAIN", "VAL", "TEST"], help="Split used to fit Stage6.")
    parser.add_argument("--calibration-split", type=str, default="VAL", choices=["TRAIN", "VAL", "TEST"], help="Split used to fit Stage7 calibration.")
    parser.add_argument("--predict-splits", type=str, default="TRAIN,VAL,TEST", help="Comma-separated splits to write predictions for.")

    parser.add_argument("--stage6-backend", type=str, choices=["similarity", "mlp"], default="similarity")
    parser.add_argument("--stage6-mlp-hidden-dim", type=int, default=64)
    parser.add_argument("--stage6-mlp-dropout", type=float, default=0.1)
    parser.add_argument("--stage6-mlp-epochs", type=int, default=80)
    parser.add_argument("--stage6-mlp-batch-size", type=int, default=128)
    parser.add_argument("--stage6-mlp-learning-rate", type=float, default=1e-3)
    parser.add_argument("--stage6-mlp-weight-decay", type=float, default=1e-4)
    parser.add_argument("--stage6-mlp-patience", type=int, default=10)
    parser.add_argument("--stage6-mlp-label-smoothing", type=float, default=0.0)
    parser.add_argument("--stage6-mlp-log-every", type=int, default=10)
    return parser.parse_args()


def _safe_entropy(weights):
    w = np.asarray(weights, dtype=np.float64)
    if w.ndim == 2 and w.shape[1] > 0:
        w = w.mean(axis=1)
    else:
        w = w.reshape(-1)
    if w.size <= 1:
        return 0.0
    w = np.clip(w, 1e-12, 1.0)
    return float(-(w * np.log(w)).sum() / np.log(w.shape[0]))


def _phase_confidences(phase_output_0, pred_ed_idx, pred_es_idx):
    """Return ED/ES temporal confidence for current and legacy phase outputs."""
    if phase_output_0.ndim == 2 and phase_output_0.shape[-1] >= 3:
        ed_time_prob = torch.softmax(phase_output_0[:, 1], dim=0).detach().cpu().numpy()
        es_time_prob = torch.softmax(phase_output_0[:, 2], dim=0).detach().cpu().numpy()
    elif phase_output_0.ndim == 1:
        phase = phase_output_0.detach().float()
        ed_score = -torch.minimum(torch.abs(phase), torch.abs(phase - 1.0))
        es_score = -torch.abs(phase - 0.5)
        ed_time_prob = torch.softmax(ed_score, dim=0).cpu().numpy()
        es_time_prob = torch.softmax(es_score, dim=0).cpu().numpy()
    else:
        flat = phase_output_0.detach().float().reshape(phase_output_0.shape[0], -1)
        score = flat.mean(dim=1)
        prob = torch.softmax(score, dim=0).cpu().numpy()
        ed_time_prob = prob
        es_time_prob = prob

    pred_ed_idx = int(np.clip(pred_ed_idx, 0, len(ed_time_prob) - 1))
    pred_es_idx = int(np.clip(pred_es_idx, 0, len(es_time_prob) - 1))
    return float(ed_time_prob[pred_ed_idx]), float(es_time_prob[pred_es_idx])


def _get_video_dims_map(data_dir):
    filelist_path = os.path.join(data_dir, "FileList.csv")
    df = pd.read_csv(filelist_path)
    dims_map = {}
    for _, row in df.iterrows():
        fname = str(row["FileName"]).strip() + ".avi"
        dims_map[fname] = (int(row["FrameHeight"]), int(row["FrameWidth"]))
    return dims_map


def _build_frame_area_lookup(data_dir):
    tracings_path = os.path.join(data_dir, "VolumeTracings.csv")
    tracings = pd.read_csv(tracings_path)

    dims_map = _get_video_dims_map(data_dir)
    stage45 = Stage45Pipeline()

    area_lookup = {}

    grouped = tracings.groupby(["FileName", "Frame"])
    for (file_name_ext, frame_id), grp in grouped:
        file_name_ext = str(file_name_ext)
        frame_id = int(frame_id)

        if file_name_ext in dims_map:
            h, w = dims_map[file_name_ext]
        else:
            max_x = float(max(grp["X1"].max(), grp["X2"].max()))
            max_y = float(max(grp["Y1"].max(), grp["Y2"].max()))
            w = int(max(2, np.ceil(max_x + 2)))
            h = int(max(2, np.ceil(max_y + 2)))

        mask = stage45.tracing_to_mask(grp.sort_index(), height=h, width=w)
        area = stage45.mask_area(mask)

        if file_name_ext not in area_lookup:
            area_lookup[file_name_ext] = {}
        area_lookup[file_name_ext][frame_id] = float(area)

    return area_lookup


def _nearest_frame(frame_ids, target):
    if not frame_ids:
        return None
    t = int(target)
    arr = np.asarray(frame_ids, dtype=np.int32)
    idx = int(np.argmin(np.abs(arr - t)))
    return int(arr[idx])


def _compute_stage5_proxy(area_lookup, file_name_ext, pred_ed_orig, pred_es_orig):
    frame_areas = area_lookup.get(file_name_ext, {})
    if not frame_areas:
        return float("nan"), float("nan"), float("nan")

    frame_ids = sorted(frame_areas.keys())
    use_ed = _nearest_frame(frame_ids, pred_ed_orig)
    use_es = _nearest_frame(frame_ids, pred_es_orig)
    if use_ed is None or use_es is None:
        return float("nan"), float("nan"), float("nan")

    ed_area = float(frame_areas[use_ed])
    es_area = float(frame_areas[use_es])

    if ed_area <= 0:
        return float("nan"), float(abs(pred_ed_orig - use_ed)), float(abs(pred_es_orig - use_es))

    ef = 100.0 * Stage45Pipeline.compute_ef_from_areas(ed_area, es_area)
    return float(ef), float(abs(pred_ed_orig - use_ed)), float(abs(pred_es_orig - use_es))


def _load_stage123_model(checkpoint_path, num_frames, device):
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Stage1-3 checkpoint not found: {checkpoint_path}")

    ckpt = torch.load(checkpoint_path, map_location=device)
    state_dict = ckpt.get("model_state_dict", ckpt)
    model = EFModel(num_frames=int(num_frames), ef_head_arch=infer_ef_head_arch(state_dict)).to(device)
    model_state = model.state_dict()
    filtered_state_dict = {
        key: value
        for key, value in state_dict.items()
        if key in model_state and tuple(value.shape) == tuple(model_state[key].shape)
    }
    incompatible = model.load_state_dict(filtered_state_dict, strict=False)
    model.eval()
    return model, incompatible


def _collect_split_rows(split, args, model, device, area_lookup):
    dataset = EchoDataset(
        data_dir=args.data_dir,
        split=str(split).upper(),
        num_frames=int(args.num_frames),
        max_videos=args.max_videos,
        normalize_input=bool(getattr(config, "NORMALIZE_INPUT", True)),
        clip_period=int(args.clip_period),
        clip_eval_mode=str(args.clip_eval_mode),
    )

    rows = []

    for i in range(len(dataset)):
        row = dataset.filelist.iloc[i]
        file_name = str(row["FileName"]).strip()
        file_name_ext = file_name + ".avi"
        video_path = os.path.join(args.data_dir, "Videos", file_name_ext)

        ed_orig = int(dataset.phase_dict[file_name_ext]["ed"])
        es_orig = int(dataset.phase_dict[file_name_ext]["es"])

        if str(args.clip_eval_mode) == "all":
            clips, sampled_indices_batch = dataset.load_video_clips(video_path, mode="all")
        else:
            clip, sampled_indices = dataset.load_video(video_path)
            clips = clip.unsqueeze(0)
            sampled_indices_batch = np.expand_dims(sampled_indices, axis=0)

        ef_values = []
        attn_entropy_values = []
        attn_peak_values = []
        phase_candidates = []

        clip_batch_size = max(1, int(args.clip_batch_size))
        with torch.no_grad():
            for batch_start in range(0, clips.shape[0], clip_batch_size):
                clip_batch = clips[batch_start : batch_start + clip_batch_size].to(device)
                model_out = model(clip_batch, return_stage_outputs=True)

                if isinstance(model_out, tuple) and len(model_out) == 4:
                    ef_pred, attention, phase_logits, _ = model_out
                else:
                    ef_pred, attention, phase_logits = model_out

                ef_values.extend((ef_pred.detach().cpu().reshape(-1).numpy() * 100.0).tolist())
                pred_ed_idx_t, pred_es_idx_t = Stage3PhaseDetector.predict_indices(phase_logits)

                for local_idx in range(int(phase_logits.shape[0])):
                    global_idx = batch_start + local_idx
                    sampled_indices = sampled_indices_batch[global_idx]

                    attn_np = attention[local_idx].detach().cpu().numpy().astype(np.float64)
                    if attn_np.ndim == 2 and attn_np.shape[1] > 0:
                        attn_for_metrics = attn_np.mean(axis=1)
                    else:
                        attn_for_metrics = attn_np.reshape(-1)
                    attn_peak_values.append(float(np.max(attn_for_metrics)))
                    attn_entropy_values.append(_safe_entropy(attn_for_metrics))

                    pred_ed_idx = int(pred_ed_idx_t[local_idx].item())
                    pred_es_idx = int(pred_es_idx_t[local_idx].item())
                    pred_ed_orig = int(sampled_indices[pred_ed_idx])
                    pred_es_orig = int(sampled_indices[pred_es_idx])

                    ed_conf, es_conf = _phase_confidences(phase_logits[local_idx], pred_ed_idx, pred_es_idx)
                    phase_candidates.append(
                        {
                            "score": ed_conf + es_conf,
                            "pred_ed_idx": pred_ed_idx,
                            "pred_es_idx": pred_es_idx,
                            "pred_ed_orig": pred_ed_orig,
                            "pred_es_orig": pred_es_orig,
                            "ed_conf": ed_conf,
                            "es_conf": es_conf,
                        }
                    )

        ef_stage123_pct = float(np.mean(ef_values))
        ef_gt_pct = float(row["EF"])

        attn_peak = float(np.mean(attn_peak_values))
        attn_entropy = float(np.mean(attn_entropy_values))

        best_phase = max(phase_candidates, key=lambda item: item["score"])
        pred_ed_idx = int(best_phase["pred_ed_idx"])
        pred_es_idx = int(best_phase["pred_es_idx"])
        pred_ed_orig = int(best_phase["pred_ed_orig"])
        pred_es_orig = int(best_phase["pred_es_orig"])
        ed_conf = float(best_phase["ed_conf"])
        es_conf = float(best_phase["es_conf"])

        ef_stage5_pct, ed_offset, es_offset = _compute_stage5_proxy(
            area_lookup=area_lookup,
            file_name_ext=file_name_ext,
            pred_ed_orig=pred_ed_orig,
            pred_es_orig=pred_es_orig,
        )

        if np.isfinite(ef_stage5_pct):
            ef_disagreement = float(abs(ef_stage123_pct - ef_stage5_pct))
        else:
            ef_disagreement = float("nan")

        gap = max(0, pred_es_idx - pred_ed_idx)
        pred_gap_norm = float(gap / max(1, int(args.num_frames) - 1))

        label = ef_to_severity_label(
            ef_pct=ef_gt_pct,
            normal_threshold=float(args.normal_threshold),
            severe_threshold=float(args.severe_threshold),
        )

        rows.append(
            {
                "split": str(split).upper(),
                "file_name": file_name,
                "file_name_ext": file_name_ext,
                "ef_gt_pct": ef_gt_pct,
                "ef_stage123_pct": ef_stage123_pct,
                "ef_stage5_pct": ef_stage5_pct,
                "ef_disagreement_pct": ef_disagreement,
                "attention_entropy": attn_entropy,
                "attention_peak": attn_peak,
                "phase_ed_conf": ed_conf,
                "phase_es_conf": es_conf,
                "pred_gap_norm": pred_gap_norm,
                "clip_count": int(clips.shape[0]),
                "clip_eval_mode": str(args.clip_eval_mode),
                "ed_trace_offset": ed_offset,
                "es_trace_offset": es_offset,
                "severity_label": int(label),
                "severity_text_gt": LABEL_TO_TEXT[int(label)],
            }
        )

    return pd.DataFrame(rows)


def _impute_and_scale(train_df, val_df, test_df, feature_cols):
    train_x = train_df[feature_cols].to_numpy(dtype=np.float64)
    med = np.nanmedian(train_x, axis=0)
    med = np.where(np.isfinite(med), med, 0.0)

    def prep(df):
        x = df[feature_cols].to_numpy(dtype=np.float64)
        x = np.where(np.isfinite(x), x, med)
        return x

    x_train = prep(train_df)
    x_val = prep(val_df)
    x_test = prep(test_df)

    mean = np.mean(x_train, axis=0)
    std = np.std(x_train, axis=0)
    std = np.where(std < 1e-8, 1.0, std)

    x_train = (x_train - mean) / std
    x_val = (x_val - mean) / std
    x_test = (x_test - mean) / std

    return x_train, x_val, x_test, med, mean, std


def _coverage(y, lo, hi):
    y = np.asarray(y, dtype=np.float64)
    lo = np.asarray(lo, dtype=np.float64)
    hi = np.asarray(hi, dtype=np.float64)
    if y.size == 0:
        return float("nan")
    return float(((y >= lo) & (y <= hi)).mean())


def _attach_predictions(df, probs_raw, probs_cal, pred_raw, pred_cal, ef_fused, lo90, hi90, lo95, hi95):
    out = df.copy()
    out["pred_label_raw"] = pred_raw.astype(int)
    out["pred_label_cal"] = pred_cal.astype(int)
    out["pred_text_raw"] = [LABEL_TO_TEXT[int(v)] for v in pred_raw.tolist()]
    out["pred_text_cal"] = [LABEL_TO_TEXT[int(v)] for v in pred_cal.tolist()]

    for c in range(3):
        out[f"prob_raw_c{c}"] = probs_raw[:, c]
        out[f"prob_cal_c{c}"] = probs_cal[:, c]

    out["ef_fused_pct"] = ef_fused
    out["ef_ci90_low"] = lo90
    out["ef_ci90_high"] = hi90
    out["ef_ci95_low"] = lo95
    out["ef_ci95_high"] = hi95
    out["ef_abs_error_fused_pct"] = np.abs(out["ef_fused_pct"].to_numpy() - out["ef_gt_pct"].to_numpy())
    return out


def _predict_logits_mlp(model, x_np, device, batch_size=4096):
    model.eval()
    outs = []
    x_t = torch.from_numpy(np.asarray(x_np, dtype=np.float32))
    with torch.no_grad():
        for i in range(0, x_t.shape[0], int(batch_size)):
            xb = x_t[i : i + int(batch_size)].to(device)
            logits = model(xb)
            outs.append(logits.detach().cpu())
    if not outs:
        return np.zeros((0, 3), dtype=np.float64)
    return torch.cat(outs, dim=0).numpy().astype(np.float64)


def _train_stage6_mlp(x_train, y_train, x_val, y_val, args, device):
    input_dim = int(x_train.shape[1])
    model = Stage6MLP(
        input_dim=input_dim,
        hidden_dim=int(args.stage6_mlp_hidden_dim),
        dropout=float(args.stage6_mlp_dropout),
        n_classes=3,
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(args.stage6_mlp_learning_rate),
        weight_decay=float(args.stage6_mlp_weight_decay),
    )
    criterion = nn.CrossEntropyLoss(label_smoothing=float(max(0.0, args.stage6_mlp_label_smoothing)))

    x_train_t = torch.from_numpy(np.asarray(x_train, dtype=np.float32).copy())
    y_train_t = torch.from_numpy(np.asarray(y_train, dtype=np.int64).copy())
    x_val_t = torch.from_numpy(np.asarray(x_val, dtype=np.float32).copy()).to(device)
    y_val_t = torch.from_numpy(np.asarray(y_val, dtype=np.int64).copy()).to(device)

    best_state = None
    best_val_loss = float("inf")
    best_epoch = 0
    bad_epochs = 0

    batch_size = int(max(8, args.stage6_mlp_batch_size))
    n_train = int(x_train_t.shape[0])

    history = []

    for epoch in range(1, int(args.stage6_mlp_epochs) + 1):
        model.train()
        perm = torch.randperm(n_train)

        train_loss_sum = 0.0
        train_seen = 0

        for i in range(0, n_train, batch_size):
            idx = perm[i : i + batch_size]
            xb = x_train_t[idx].to(device)
            yb = y_train_t[idx].to(device)

            optimizer.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()

            bs = int(xb.shape[0])
            train_loss_sum += float(loss.item()) * bs
            train_seen += bs

        train_loss = train_loss_sum / max(1, train_seen)

        model.eval()
        with torch.no_grad():
            val_logits = model(x_val_t)
            val_loss = float(criterion(val_logits, y_val_t).item())
            val_pred = torch.argmax(val_logits, dim=1).detach().cpu().numpy()
            val_acc = accuracy_np(val_pred, y_val)
            val_f1 = macro_f1_np(val_pred, y_val)

        history.append(
            {
                "epoch": int(epoch),
                "train_loss": float(train_loss),
                "val_loss": float(val_loss),
                "val_acc": float(val_acc),
                "val_f1": float(val_f1),
            }
        )

        if epoch == 1 or epoch % int(max(1, args.stage6_mlp_log_every)) == 0:
            print(
                f"Stage6-MLP epoch {epoch:03d}/{int(args.stage6_mlp_epochs)} | "
                f"train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | val_acc={val_acc*100:.2f}% | val_f1={val_f1:.4f}"
            )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = int(epoch)
            bad_epochs = 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad_epochs += 1
            if bad_epochs >= int(max(1, args.stage6_mlp_patience)):
                print(f"Stage6-MLP early stopping at epoch {epoch} (best epoch={best_epoch}, val_loss={best_val_loss:.4f})")
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    return model, {
        "best_epoch": int(best_epoch),
        "best_val_loss": float(best_val_loss),
        "history": history,
    }


def _parse_split_list(value):
    splits = []
    for item in str(value).split(","):
        split = item.strip().upper()
        if not split:
            continue
        if split not in {"TRAIN", "VAL", "TEST"}:
            raise ValueError(f"Unsupported split: {split}")
        if split not in splits:
            splits.append(split)
    if not splits:
        raise ValueError("At least one prediction split is required")
    return splits


def _impute_and_scale_from_fit(split_dfs, fit_split, feature_cols):
    fit_x = split_dfs[fit_split][feature_cols].to_numpy(dtype=np.float64)
    med = np.nanmedian(fit_x, axis=0)
    med = np.where(np.isfinite(med), med, 0.0)

    fit_prepped = np.where(np.isfinite(fit_x), fit_x, med)
    mean = np.mean(fit_prepped, axis=0)
    std = np.std(fit_prepped, axis=0)
    std = np.where(std < 1e-8, 1.0, std)

    x_by_split = {}
    for split, df in split_dfs.items():
        x = df[feature_cols].to_numpy(dtype=np.float64)
        x = np.where(np.isfinite(x), x, med)
        x_by_split[split] = (x - mean) / std

    return x_by_split, med, mean, std


def _split_metrics(df, y, pred_raw, pred_cal, fused, lo90, hi90, lo95, hi95):
    gt = df["ef_gt_pct"].to_numpy(dtype=np.float64)
    return {
        "stage6_acc_raw": accuracy_np(pred_raw, y),
        "stage6_macro_f1_raw": macro_f1_np(pred_raw, y),
        "stage6_acc_cal": accuracy_np(pred_cal, y),
        "stage6_macro_f1_cal": macro_f1_np(pred_cal, y),
        "stage5_fused_ef_mae_pct": float(np.mean(np.abs(fused - gt))),
        "ef_ci90_coverage": _coverage(gt, lo90, hi90),
        "ef_ci95_coverage": _coverage(gt, lo95, hi95),
        "confusion_raw": confusion_matrix_np(pred_raw, y).tolist(),
        "confusion_cal": confusion_matrix_np(pred_cal, y).tolist(),
    }


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    t0 = time.perf_counter()

    device = str(args.device)
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"

    print("=" * 96)
    print("STAGE 6/7 TRAINING (SIMILARITY + UNCERTAINTY)")
    print("=" * 96)
    print(f"Device: {device}")
    print(f"Data dir: {args.data_dir}")
    print(f"Stage1-3 checkpoint: {args.stage123_checkpoint}")
    print(f"Num frames: {args.num_frames}")
    print(f"Clip sampling: period={args.clip_period}, eval_mode={args.clip_eval_mode}, batch_size={args.clip_batch_size}")
    print(f"Max videos per split: {args.max_videos if args.max_videos else 'All'}")
    print(f"Severity thresholds: severe<{args.severe_threshold}, normal>={args.normal_threshold}")
    print(f"Stage6 backend: {args.stage6_backend}")
    print(f"Output dir: {os.path.abspath(args.output_dir)}")
    print("=" * 96)

    model, incompatible = _load_stage123_model(args.stage123_checkpoint, args.num_frames, device)
    if len(incompatible.missing_keys) or len(incompatible.unexpected_keys):
        print(
            "Warning: checkpoint loaded with key mismatch | "
            f"missing={len(incompatible.missing_keys)} unexpected={len(incompatible.unexpected_keys)}"
        )

    fit_split = str(args.fit_split).upper()
    calibration_split = str(args.calibration_split).upper()
    predict_splits = _parse_split_list(args.predict_splits)
    required_splits = []
    for split in [fit_split, calibration_split, *predict_splits]:
        if split not in required_splits:
            required_splits.append(split)

    print(f"Stage6 fit split: {fit_split}")
    print(f"Stage7 calibration split: {calibration_split}")
    print(f"Prediction splits: {','.join(predict_splits)}")

    area_lookup = _build_frame_area_lookup(args.data_dir)

    split_dfs = {
        split: _collect_split_rows(split, args, model, device, area_lookup)
        for split in required_splits
    }
    print("Feature rows -> " + " ".join(f"{split.lower()}={len(df)}" for split, df in split_dfs.items()))

    x_by_split, med, mean, std = _impute_and_scale_from_fit(
        split_dfs=split_dfs,
        fit_split=fit_split,
        feature_cols=FEATURE_COLUMNS,
    )
    y_by_split = {
        split: df["severity_label"].to_numpy(dtype=np.int64)
        for split, df in split_dfs.items()
    }

    x_fit = x_by_split[fit_split]
    y_fit = y_by_split[fit_split]
    x_cal = x_by_split[calibration_split]
    y_cal = y_by_split[calibration_split]

    stage6_artifact = None
    stage6_extra = {}

    if str(args.stage6_backend) == "similarity":
        stage6 = Stage6SimilarityEngine()
        stage6.fit(x_fit, y_fit)

        logits_by_split = {
            split: stage6.predict_logits(x)
            for split, x in x_by_split.items()
        }
        probs_raw_by_split = {
            split: stage6.predict_proba(x, temperature=1.0)
            for split, x in x_by_split.items()
        }

        stage6_npz = os.path.join(args.output_dir, "stage6_similarity_engine.npz")
        stage6.save_npz(stage6_npz)
        stage6_artifact = stage6_npz
    else:
        stage6_mlp, mlp_info = _train_stage6_mlp(
            x_train=x_fit,
            y_train=y_fit,
            x_val=x_cal,
            y_val=y_cal,
            args=args,
            device=device,
        )

        logits_by_split = {
            split: _predict_logits_mlp(stage6_mlp, x, device=device)
            for split, x in x_by_split.items()
        }
        probs_raw_by_split = {
            split: softmax_np(logits, temperature=1.0)
            for split, logits in logits_by_split.items()
        }

        stage6_pth = os.path.join(args.output_dir, "stage6_mlp_model.pth")
        torch.save(
            {
                "model_state_dict": stage6_mlp.state_dict(),
                "input_dim": int(x_fit.shape[1]),
                "hidden_dim": int(args.stage6_mlp_hidden_dim),
                "dropout": float(args.stage6_mlp_dropout),
                "feature_columns": FEATURE_COLUMNS,
                "train_args": vars(args),
                "best_epoch": int(mlp_info["best_epoch"]),
                "best_val_loss": float(mlp_info["best_val_loss"]),
            },
            stage6_pth,
        )
        stage6_artifact = stage6_pth
        stage6_extra = {
            "mlp_best_epoch": int(mlp_info["best_epoch"]),
            "mlp_best_val_loss": float(mlp_info["best_val_loss"]),
        }

    pred_raw_by_split = {
        split: np.argmax(probs, axis=1)
        for split, probs in probs_raw_by_split.items()
    }

    stage7 = Stage7UncertaintyCalibrator()
    calibration_df = split_dfs[calibration_split]
    stage7.fit(
        val_logits=logits_by_split[calibration_split],
        val_labels=y_cal,
        ef_stage123_pct=calibration_df["ef_stage123_pct"].to_numpy(dtype=np.float64),
        ef_stage5_pct=calibration_df["ef_stage5_pct"].to_numpy(dtype=np.float64),
        ef_gt_pct=calibration_df["ef_gt_pct"].to_numpy(dtype=np.float64),
    )

    pred_dfs = {}
    metrics_by_split = {}
    for split, df in split_dfs.items():
        probs_cal = stage7.calibrated_proba(logits_by_split[split])
        pred_cal = np.argmax(probs_cal, axis=1)
        fused = stage7.fuse_ef(
            df["ef_stage123_pct"].to_numpy(dtype=np.float64),
            df["ef_stage5_pct"].to_numpy(dtype=np.float64),
        )
        lo90, hi90, lo95, hi95 = stage7.intervals(fused)
        if split in predict_splits:
            pred_dfs[split] = _attach_predictions(
                df,
                probs_raw_by_split[split],
                probs_cal,
                pred_raw_by_split[split],
                pred_cal,
                fused,
                lo90,
                hi90,
                lo95,
                hi95,
            )
        metrics_by_split[split.lower()] = _split_metrics(
            df=df,
            y=y_by_split[split],
            pred_raw=pred_raw_by_split[split],
            pred_cal=pred_cal,
            fused=fused,
            lo90=lo90,
            hi90=hi90,
            lo95=lo95,
            hi95=hi95,
        )

    metrics = {
        "stage6": {
            "backend": str(args.stage6_backend),
            "artifact": os.path.abspath(stage6_artifact) if stage6_artifact else None,
            **stage6_extra,
        },
        **metrics_by_split,
        "stage7": {
            "temperature": float(stage7.temperature),
            "fusion_alpha": float(stage7.fusion_alpha),
            "q90_abs_error": float(stage7.q90_abs_error),
            "q95_abs_error": float(stage7.q95_abs_error),
        },
        "feature_columns": FEATURE_COLUMNS,
        "severity_thresholds": {
            "normal_threshold": float(args.normal_threshold),
            "severe_threshold": float(args.severe_threshold),
        },
        "n_samples": {
            split.lower(): int(len(df))
            for split, df in split_dfs.items()
        },
        "split_config": {
            "fit_split": fit_split,
            "calibration_split": calibration_split,
            "predict_splits": predict_splits,
        },
    }

    stage7_json = os.path.join(args.output_dir, "stage7_calibration.json")
    stage7.save_json(stage7_json)

    preprocess_json = os.path.join(args.output_dir, "stage67_feature_preprocess.json")
    with open(preprocess_json, "w", encoding="utf-8") as f:
        json.dump(
            {
                "feature_columns": FEATURE_COLUMNS,
                "impute_median": med.tolist(),
                "standardize_mean": mean.tolist(),
                "standardize_std": std.tolist(),
            },
            f,
            indent=2,
        )

    if args.save_per_split_csv:
        for split, pred_df in pred_dfs.items():
            pred_df.to_csv(os.path.join(args.output_dir, f"stage67_{split.lower()}_predictions.csv"), index=False)

    summary_json = os.path.join(args.output_dir, "stage67_summary.json")
    with open(summary_json, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    dt = time.perf_counter() - t0

    print("=" * 96)
    print("STAGE 6/7 SUMMARY")
    print("=" * 96)
    print(f"Stage6 backend: {metrics['stage6']['backend']}")
    print(f"Stage6 artifact: {metrics['stage6']['artifact']}")
    for split in required_splits:
        split_key = split.lower()
        split_metrics = metrics.get(split_key)
        if split_metrics:
            print(
                f"{split} Stage6 acc raw/cal: {split_metrics['stage6_acc_raw']*100:.2f}% / {split_metrics['stage6_acc_cal']*100:.2f}% | "
                f"macro-F1 raw/cal: {split_metrics['stage6_macro_f1_raw']:.4f} / {split_metrics['stage6_macro_f1_cal']:.4f}"
            )
    print(
        f"Stage7 temperature={metrics['stage7']['temperature']:.3f} | "
        f"fusion_alpha={metrics['stage7']['fusion_alpha']:.3f} | "
        f"q90={metrics['stage7']['q90_abs_error']:.2f} | q95={metrics['stage7']['q95_abs_error']:.2f}"
    )
    for split in predict_splits:
        split_metrics = metrics.get(split.lower())
        if split_metrics:
            print(
                f"{split} EF fused MAE: {split_metrics['stage5_fused_ef_mae_pct']:.2f}% | "
                f"CI90 coverage: {split_metrics['ef_ci90_coverage']*100:.2f}% | "
                f"CI95 coverage: {split_metrics['ef_ci95_coverage']*100:.2f}%"
            )
    print(f"Artifacts: {os.path.abspath(args.output_dir)}")
    print(f"Total duration: {dt:.1f}s")
    print("=" * 96)


if __name__ == "__main__":
    main()

