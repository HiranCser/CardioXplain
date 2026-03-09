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
from models.ef_model import EFModel
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
    parser.add_argument("--temporal-window-mode", type=str, choices=["full", "tracing"], default="tracing")
    parser.add_argument("--temporal-window-margin-mult", type=float, default=1.0)
    parser.add_argument("--save-per-split-csv", action=argparse.BooleanOptionalAction, default=True)

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
    if w.size <= 1:
        return 0.0
    w = np.clip(w, 1e-12, 1.0)
    return float(-(w * np.log(w)).sum() / np.log(w.shape[0]))


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

    model = EFModel(num_frames=int(num_frames)).to(device)
    ckpt = torch.load(checkpoint_path, map_location=device)
    state_dict = ckpt.get("model_state_dict", ckpt)
    incompatible = model.load_state_dict(state_dict, strict=False)
    model.eval()
    return model, incompatible


def _collect_split_rows(split, args, model, device, area_lookup):
    dataset = EchoDataset(
        data_dir=args.data_dir,
        split=str(split).upper(),
        num_frames=int(args.num_frames),
        max_videos=args.max_videos,
        normalize_input=bool(getattr(config, "NORMALIZE_INPUT", True)),
        temporal_window_mode=str(args.temporal_window_mode),
        temporal_window_margin_mult=float(args.temporal_window_margin_mult),
        temporal_window_jitter_mult=0.0,
    )

    rows = []

    for i in range(len(dataset)):
        row = dataset.filelist.iloc[i]
        file_name = str(row["FileName"]).strip()
        file_name_ext = file_name + ".avi"
        video_path = os.path.join(args.data_dir, "Videos", file_name_ext)

        ed_orig = int(dataset.phase_dict[file_name_ext]["ed"])
        es_orig = int(dataset.phase_dict[file_name_ext]["es"])

        clip, sampled_indices = dataset.load_video(
            video_path,
            ed_original=ed_orig,
            es_original=es_orig,
        )

        with torch.no_grad():
            model_out = model(clip.unsqueeze(0).to(device), return_stage_outputs=True)

        if isinstance(model_out, tuple) and len(model_out) == 4:
            ef_pred, attention, phase_logits, _ = model_out
        else:
            ef_pred, attention, phase_logits = model_out

        ef_stage123_pct = float(ef_pred[0].item() * 100.0)
        ef_gt_pct = float(row["EF"])

        attn_np = attention[0].detach().cpu().numpy().astype(np.float64)
        attn_peak = float(np.max(attn_np))
        attn_entropy = _safe_entropy(attn_np)

        pred_ed_idx_t, pred_es_idx_t = Stage3PhaseDetector.predict_indices(phase_logits)
        pred_ed_idx = int(pred_ed_idx_t[0].item())
        pred_es_idx = int(pred_es_idx_t[0].item())

        pred_ed_orig = int(sampled_indices[pred_ed_idx])
        pred_es_orig = int(sampled_indices[pred_es_idx])

        phase_logits_0 = phase_logits[0]
        ed_time_prob = torch.softmax(phase_logits_0[:, 1], dim=0).detach().cpu().numpy()
        es_time_prob = torch.softmax(phase_logits_0[:, 2], dim=0).detach().cpu().numpy()
        ed_conf = float(ed_time_prob[pred_ed_idx])
        es_conf = float(es_time_prob[pred_es_idx])

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
    print(f"Temporal window: {args.temporal_window_mode} (margin={args.temporal_window_margin_mult})")
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

    area_lookup = _build_frame_area_lookup(args.data_dir)

    train_df = _collect_split_rows("TRAIN", args, model, device, area_lookup)
    val_df = _collect_split_rows("VAL", args, model, device, area_lookup)
    test_df = _collect_split_rows("TEST", args, model, device, area_lookup)

    print(f"Feature rows -> train={len(train_df)} val={len(val_df)} test={len(test_df)}")

    x_train, x_val, x_test, med, mean, std = _impute_and_scale(
        train_df=train_df,
        val_df=val_df,
        test_df=test_df,
        feature_cols=FEATURE_COLUMNS,
    )

    y_train = train_df["severity_label"].to_numpy(dtype=np.int64)
    y_val = val_df["severity_label"].to_numpy(dtype=np.int64)
    y_test = test_df["severity_label"].to_numpy(dtype=np.int64)

    stage6_artifact = None
    stage6_extra = {}

    if str(args.stage6_backend) == "similarity":
        stage6 = Stage6SimilarityEngine()
        stage6.fit(x_train, y_train)

        logits_train = stage6.predict_logits(x_train)
        logits_val = stage6.predict_logits(x_val)
        logits_test = stage6.predict_logits(x_test)

        probs_train_raw = stage6.predict_proba(x_train, temperature=1.0)
        probs_val_raw = stage6.predict_proba(x_val, temperature=1.0)
        probs_test_raw = stage6.predict_proba(x_test, temperature=1.0)

        stage6_npz = os.path.join(args.output_dir, "stage6_similarity_engine.npz")
        stage6.save_npz(stage6_npz)
        stage6_artifact = stage6_npz
    else:
        stage6_mlp, mlp_info = _train_stage6_mlp(
            x_train=x_train,
            y_train=y_train,
            x_val=x_val,
            y_val=y_val,
            args=args,
            device=device,
        )

        logits_train = _predict_logits_mlp(stage6_mlp, x_train, device=device)
        logits_val = _predict_logits_mlp(stage6_mlp, x_val, device=device)
        logits_test = _predict_logits_mlp(stage6_mlp, x_test, device=device)

        probs_train_raw = softmax_np(logits_train, temperature=1.0)
        probs_val_raw = softmax_np(logits_val, temperature=1.0)
        probs_test_raw = softmax_np(logits_test, temperature=1.0)

        stage6_pth = os.path.join(args.output_dir, "stage6_mlp_model.pth")
        torch.save(
            {
                "model_state_dict": stage6_mlp.state_dict(),
                "input_dim": int(x_train.shape[1]),
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

    pred_train_raw = np.argmax(probs_train_raw, axis=1)
    pred_val_raw = np.argmax(probs_val_raw, axis=1)
    pred_test_raw = np.argmax(probs_test_raw, axis=1)

    stage7 = Stage7UncertaintyCalibrator()
    stage7.fit(
        val_logits=logits_val,
        val_labels=y_val,
        ef_stage123_pct=val_df["ef_stage123_pct"].to_numpy(dtype=np.float64),
        ef_stage5_pct=val_df["ef_stage5_pct"].to_numpy(dtype=np.float64),
        ef_gt_pct=val_df["ef_gt_pct"].to_numpy(dtype=np.float64),
    )

    probs_train_cal = stage7.calibrated_proba(logits_train)
    probs_val_cal = stage7.calibrated_proba(logits_val)
    probs_test_cal = stage7.calibrated_proba(logits_test)

    pred_train_cal = np.argmax(probs_train_cal, axis=1)
    pred_val_cal = np.argmax(probs_val_cal, axis=1)
    pred_test_cal = np.argmax(probs_test_cal, axis=1)

    ef_train_fused = stage7.fuse_ef(train_df["ef_stage123_pct"].to_numpy(), train_df["ef_stage5_pct"].to_numpy())
    ef_val_fused = stage7.fuse_ef(val_df["ef_stage123_pct"].to_numpy(), val_df["ef_stage5_pct"].to_numpy())
    ef_test_fused = stage7.fuse_ef(test_df["ef_stage123_pct"].to_numpy(), test_df["ef_stage5_pct"].to_numpy())

    tr_lo90, tr_hi90, tr_lo95, tr_hi95 = stage7.intervals(ef_train_fused)
    va_lo90, va_hi90, va_lo95, va_hi95 = stage7.intervals(ef_val_fused)
    te_lo90, te_hi90, te_lo95, te_hi95 = stage7.intervals(ef_test_fused)

    train_pred_df = _attach_predictions(
        train_df, probs_train_raw, probs_train_cal, pred_train_raw, pred_train_cal,
        ef_train_fused, tr_lo90, tr_hi90, tr_lo95, tr_hi95
    )
    val_pred_df = _attach_predictions(
        val_df, probs_val_raw, probs_val_cal, pred_val_raw, pred_val_cal,
        ef_val_fused, va_lo90, va_hi90, va_lo95, va_hi95
    )
    test_pred_df = _attach_predictions(
        test_df, probs_test_raw, probs_test_cal, pred_test_raw, pred_test_cal,
        ef_test_fused, te_lo90, te_hi90, te_lo95, te_hi95
    )

    metrics = {
        "stage6": {
            "backend": str(args.stage6_backend),
            "artifact": os.path.abspath(stage6_artifact) if stage6_artifact else None,
            **stage6_extra,
        },
        "train": {
            "stage6_acc_raw": accuracy_np(pred_train_raw, y_train),
            "stage6_macro_f1_raw": macro_f1_np(pred_train_raw, y_train),
            "stage6_acc_cal": accuracy_np(pred_train_cal, y_train),
            "stage6_macro_f1_cal": macro_f1_np(pred_train_cal, y_train),
            "stage5_fused_ef_mae_pct": float(np.mean(np.abs(ef_train_fused - train_df["ef_gt_pct"].to_numpy()))),
            "ef_ci90_coverage": _coverage(train_df["ef_gt_pct"].to_numpy(), tr_lo90, tr_hi90),
            "ef_ci95_coverage": _coverage(train_df["ef_gt_pct"].to_numpy(), tr_lo95, tr_hi95),
            "confusion_raw": confusion_matrix_np(pred_train_raw, y_train).tolist(),
            "confusion_cal": confusion_matrix_np(pred_train_cal, y_train).tolist(),
        },
        "val": {
            "stage6_acc_raw": accuracy_np(pred_val_raw, y_val),
            "stage6_macro_f1_raw": macro_f1_np(pred_val_raw, y_val),
            "stage6_acc_cal": accuracy_np(pred_val_cal, y_val),
            "stage6_macro_f1_cal": macro_f1_np(pred_val_cal, y_val),
            "stage5_fused_ef_mae_pct": float(np.mean(np.abs(ef_val_fused - val_df["ef_gt_pct"].to_numpy()))),
            "ef_ci90_coverage": _coverage(val_df["ef_gt_pct"].to_numpy(), va_lo90, va_hi90),
            "ef_ci95_coverage": _coverage(val_df["ef_gt_pct"].to_numpy(), va_lo95, va_hi95),
            "confusion_raw": confusion_matrix_np(pred_val_raw, y_val).tolist(),
            "confusion_cal": confusion_matrix_np(pred_val_cal, y_val).tolist(),
        },
        "test": {
            "stage6_acc_raw": accuracy_np(pred_test_raw, y_test),
            "stage6_macro_f1_raw": macro_f1_np(pred_test_raw, y_test),
            "stage6_acc_cal": accuracy_np(pred_test_cal, y_test),
            "stage6_macro_f1_cal": macro_f1_np(pred_test_cal, y_test),
            "stage5_fused_ef_mae_pct": float(np.mean(np.abs(ef_test_fused - test_df["ef_gt_pct"].to_numpy()))),
            "ef_ci90_coverage": _coverage(test_df["ef_gt_pct"].to_numpy(), te_lo90, te_hi90),
            "ef_ci95_coverage": _coverage(test_df["ef_gt_pct"].to_numpy(), te_lo95, te_hi95),
            "confusion_raw": confusion_matrix_np(pred_test_raw, y_test).tolist(),
            "confusion_cal": confusion_matrix_np(pred_test_cal, y_test).tolist(),
        },
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
            "train": int(len(train_df)),
            "val": int(len(val_df)),
            "test": int(len(test_df)),
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
        train_pred_df.to_csv(os.path.join(args.output_dir, "stage67_train_predictions.csv"), index=False)
        val_pred_df.to_csv(os.path.join(args.output_dir, "stage67_val_predictions.csv"), index=False)
        test_pred_df.to_csv(os.path.join(args.output_dir, "stage67_test_predictions.csv"), index=False)

    summary_json = os.path.join(args.output_dir, "stage67_summary.json")
    with open(summary_json, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    dt = time.perf_counter() - t0

    print("=" * 96)
    print("STAGE 6/7 SUMMARY")
    print("=" * 96)
    print(f"Stage6 backend: {metrics['stage6']['backend']}")
    print(f"Stage6 artifact: {metrics['stage6']['artifact']}")
    print(
        f"VAL Stage6 acc raw/cal: {metrics['val']['stage6_acc_raw']*100:.2f}% / {metrics['val']['stage6_acc_cal']*100:.2f}% | "
        f"macro-F1 raw/cal: {metrics['val']['stage6_macro_f1_raw']:.4f} / {metrics['val']['stage6_macro_f1_cal']:.4f}"
    )
    print(
        f"TEST Stage6 acc raw/cal: {metrics['test']['stage6_acc_raw']*100:.2f}% / {metrics['test']['stage6_acc_cal']*100:.2f}% | "
        f"macro-F1 raw/cal: {metrics['test']['stage6_macro_f1_raw']:.4f} / {metrics['test']['stage6_macro_f1_cal']:.4f}"
    )
    print(
        f"Stage7 temperature={metrics['stage7']['temperature']:.3f} | "
        f"fusion_alpha={metrics['stage7']['fusion_alpha']:.3f} | "
        f"q90={metrics['stage7']['q90_abs_error']:.2f} | q95={metrics['stage7']['q95_abs_error']:.2f}"
    )
    print(
        f"TEST EF fused MAE: {metrics['test']['stage5_fused_ef_mae_pct']:.2f}% | "
        f"CI90 coverage: {metrics['test']['ef_ci90_coverage']*100:.2f}% | "
        f"CI95 coverage: {metrics['test']['ef_ci95_coverage']*100:.2f}%"
    )
    print(f"Artifacts: {os.path.abspath(args.output_dir)}")
    print(f"Total duration: {dt:.1f}s")
    print("=" * 96)


if __name__ == "__main__":
    main()

