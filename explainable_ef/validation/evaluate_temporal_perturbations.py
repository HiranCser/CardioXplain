import argparse
import csv
import json
import math
import os
import random
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import config
from data.dataset import EchoDataset
from models.ef_model import EFModel
from pipeline.stage3_phase_detector import Stage3PhaseDetector
from validation.temporal_perturbations import (
    AVAILABLE_PERTURBATIONS,
    apply_temporal_perturbation,
    summarize_temporal_attention,
)
from visualization.visualize_attention import plot_attention, plot_phase_curves


def parse_args():
    parser = argparse.ArgumentParser(description="Run paired temporal perturbation experiments on a trained checkpoint.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--split", type=str, default="VAL", choices=["TRAIN", "VAL", "TEST"])
    parser.add_argument("--num-frames", type=int, default=int(getattr(config, "NUM_FRAMES", 32)))
    parser.add_argument("--dataset-period", type=int, default=int(getattr(config, "DATASET_PERIOD", 1)))
    parser.add_argument("--dataset-max-length", type=int, default=getattr(config, "DATASET_MAX_LENGTH", None))
    parser.add_argument("--max-videos", type=int, default=50)
    parser.add_argument("--phase-temporal-window-mode", type=str, default=str(getattr(config, "PHASE_TEMPORAL_WINDOW_MODE", "full")))
    parser.add_argument("--phase-temporal-window-margin-mult", type=float, default=float(getattr(config, "PHASE_TEMPORAL_WINDOW_MARGIN_MULT", 1.5)))
    parser.add_argument("--phase-temporal-window-jitter-mult", type=float, default=0.0)
    parser.add_argument("--perturbations", type=str, default="random_mask,attention_guided_mask,contiguous_mask,temporal_shift,local_shuffle,reverse_window,frame_drop")
    parser.add_argument("--severities", type=str, default="0.1,0.2,0.3")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--bootstrap-samples", type=int, default=500)
    parser.add_argument("--plots", type=int, default=6, help="Maximum number of paired clean/perturbed plot sets to save")
    parser.add_argument("--output-dir", type=str, default=os.path.join("validation", "outputs", "temporal_perturbations"))
    parser.add_argument("--device", type=str, default=None, help="cuda or cpu; default uses config")
    return parser.parse_args()


def parse_csv_list(raw_value, cast_fn):
    items = []
    for token in str(raw_value).split(","):
        token = token.strip()
        if not token:
            continue
        items.append(cast_fn(token))
    return items


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_model(checkpoint_path, num_frames, device):
    model = EFModel(num_frames=int(num_frames)).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint["model_state_dict"] if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint else checkpoint
    model_state = model.state_dict()
    filtered_state_dict = {
        key: value
        for key, value in state_dict.items()
        if key in model_state and tuple(value.shape) == tuple(model_state[key].shape)
    }
    incompatible = model.load_state_dict(filtered_state_dict, strict=False)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        print(
            f"Warning: checkpoint loaded with key mismatch | missing={len(incompatible.missing_keys)} unexpected={len(incompatible.unexpected_keys)}"
        )
    model.eval()
    return model


def run_inference(model, clip, device):
    with torch.no_grad():
        ef_pred, attention, phase_logits = model(clip.unsqueeze(0).to(device))

    ef_pred_pct = float(ef_pred[0].item() * 100.0)
    attention_np = attention[0].detach().cpu().numpy()
    attn_summary = summarize_temporal_attention(attention_np)
    phase_probs = torch.softmax(phase_logits[0], dim=-1).detach().cpu().numpy()

    pred_ed_idx, pred_es_idx = Stage3PhaseDetector.predict_indices(phase_logits)
    pred_ed_idx = int(pred_ed_idx[0].item())
    pred_es_idx = int(pred_es_idx[0].item())

    return {
        "ef_pred_pct": ef_pred_pct,
        "attention_summary": attn_summary,
        "phase_probs": phase_probs,
        "pred_ed_idx": pred_ed_idx,
        "pred_es_idx": pred_es_idx,
        "pred_ed_conf": float(phase_probs[pred_ed_idx, 1]) if phase_probs.shape[0] > 0 else float("nan"),
        "pred_es_conf": float(phase_probs[pred_es_idx, 2]) if phase_probs.shape[0] > 0 else float("nan"),
    }


def bootstrap_mean_ci(values, seed, n_boot=500, alpha=0.05):
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return [float("nan"), float("nan")]
    if values.size == 1:
        v = float(values[0])
        return [v, v]

    rng = np.random.default_rng(int(seed))
    means = np.empty(int(max(10, n_boot)), dtype=np.float64)
    for i in range(means.shape[0]):
        sample = values[rng.integers(0, values.size, size=values.size)]
        means[i] = float(np.mean(sample))
    lo = float(np.quantile(means, alpha / 2.0))
    hi = float(np.quantile(means, 1.0 - alpha / 2.0))
    return [lo, hi]


def make_dataset(args):
    return EchoDataset(
        data_dir=config.DATA_DIR,
        split=args.split,
        num_frames=int(args.num_frames),
        period=int(args.dataset_period),
        max_length=args.dataset_max_length,
        max_videos=args.max_videos,
        clips=1,
        normalize_input=bool(getattr(config, "NORMALIZE_INPUT", True)),
        temporal_window_mode=args.phase_temporal_window_mode,
        temporal_window_margin_mult=args.phase_temporal_window_margin_mult,
        temporal_window_jitter_mult=args.phase_temporal_window_jitter_mult,
    )


def save_plot_set(output_dir, plot_index, row, clean_result, pert_result):
    stem = (
        f"{plot_index:03d}_"
        f"{row['perturbation']}_sev{row['severity']:.2f}_"
        f"sample{int(row['sample_index']):04d}_{row['file_name'].replace('.avi', '')}"
    )
    safe_stem = stem.replace(" ", "_")
    plot_dir = os.path.join(output_dir, "plots")

    plot_attention(
        clean_result["attention_summary"],
        ed_idx=int(row["gt_ed_clip_idx"]),
        es_idx=int(row["gt_es_clip_idx"]),
        pred_ed_idx=int(row["clean_pred_ed_idx"]),
        pred_es_idx=int(row["clean_pred_es_idx"]),
        save_path=os.path.join(plot_dir, f"{safe_stem}_clean_attention.png"),
    )
    plot_attention(
        pert_result["attention_summary"],
        ed_idx=int(row["gt_ed_clip_idx"]),
        es_idx=int(row["gt_es_clip_idx"]),
        pred_ed_idx=int(row["perturbed_pred_ed_idx"]),
        pred_es_idx=int(row["perturbed_pred_es_idx"]),
        save_path=os.path.join(plot_dir, f"{safe_stem}_perturbed_attention.png"),
    )
    plot_phase_curves(
        clean_result["phase_probs"],
        ed_idx=int(row["gt_ed_clip_idx"]),
        es_idx=int(row["gt_es_clip_idx"]),
        pred_ed_idx=int(row["clean_pred_ed_idx"]),
        pred_es_idx=int(row["clean_pred_es_idx"]),
        save_path=os.path.join(plot_dir, f"{safe_stem}_clean_phase.png"),
    )
    plot_phase_curves(
        pert_result["phase_probs"],
        ed_idx=int(row["gt_ed_clip_idx"]),
        es_idx=int(row["gt_es_clip_idx"]),
        pred_ed_idx=int(row["perturbed_pred_ed_idx"]),
        pred_es_idx=int(row["perturbed_pred_es_idx"]),
        save_path=os.path.join(plot_dir, f"{safe_stem}_perturbed_phase.png"),
    )


def build_summary(rows, bootstrap_samples, seed):
    groups = defaultdict(list)
    for row in rows:
        groups[(row["perturbation"], float(row["severity"]))].append(row)

    summary = {"groups": []}
    for (perturbation, severity), group_rows in sorted(groups.items(), key=lambda x: (x[0][0], x[0][1])):
        delta_abs = np.array([r["delta_ef_abs_error_pct"] for r in group_rows], dtype=np.float64)
        delta_ed = np.array([r["delta_ed_abs_error_frames"] for r in group_rows], dtype=np.float64)
        delta_es = np.array([r["delta_es_abs_error_frames"] for r in group_rows], dtype=np.float64)
        clean_abs = np.array([r["clean_ef_abs_error_pct"] for r in group_rows], dtype=np.float64)
        pert_abs = np.array([r["perturbed_ef_abs_error_pct"] for r in group_rows], dtype=np.float64)
        guided = {
            "perturbation": perturbation,
            "severity": float(severity),
            "samples": len(group_rows),
            "clean_ef_mae_pct": float(np.mean(clean_abs)),
            "perturbed_ef_mae_pct": float(np.mean(pert_abs)),
            "delta_ef_mae_pct": float(np.mean(delta_abs)),
            "delta_ef_mae_ci95": bootstrap_mean_ci(delta_abs, seed=seed + len(summary["groups"]) * 97, n_boot=bootstrap_samples),
            "clean_ed_mae_frames": float(np.mean([r["clean_ed_abs_error_frames"] for r in group_rows])),
            "perturbed_ed_mae_frames": float(np.mean([r["perturbed_ed_abs_error_frames"] for r in group_rows])),
            "delta_ed_mae_frames": float(np.mean(delta_ed)),
            "delta_ed_mae_ci95": bootstrap_mean_ci(delta_ed, seed=seed + len(summary["groups"]) * 131, n_boot=bootstrap_samples),
            "clean_es_mae_frames": float(np.mean([r["clean_es_abs_error_frames"] for r in group_rows])),
            "perturbed_es_mae_frames": float(np.mean([r["perturbed_es_abs_error_frames"] for r in group_rows])),
            "delta_es_mae_frames": float(np.mean(delta_es)),
            "delta_es_mae_ci95": bootstrap_mean_ci(delta_es, seed=seed + len(summary["groups"]) * 173, n_boot=bootstrap_samples),
            "mean_abs_prediction_shift_pct": float(np.mean([abs(r["perturbed_ef_pred_pct"] - r["clean_ef_pred_pct"]) for r in group_rows])),
            "ef_prediction_flip_rate_5pct": float(np.mean([abs(r["perturbed_ef_pred_pct"] - r["clean_ef_pred_pct"]) >= 5.0 for r in group_rows])),
            "phase_joint_within_tol_clean": float(np.mean([r["clean_joint_within_tol"] for r in group_rows])),
            "phase_joint_within_tol_perturbed": float(np.mean([r["perturbed_joint_within_tol"] for r in group_rows])),
        }
        summary["groups"].append(guided)

    attention_vs_random = []
    by_name = defaultdict(dict)
    for group in summary["groups"]:
        by_name[group["severity"]][group["perturbation"]] = group
    for severity, groups_at_severity in sorted(by_name.items(), key=lambda x: x[0]):
        if "attention_guided_mask" in groups_at_severity and "random_mask" in groups_at_severity:
            guided = groups_at_severity["attention_guided_mask"]
            random_group = groups_at_severity["random_mask"]
            attention_vs_random.append(
                {
                    "severity": float(severity),
                    "guided_delta_ef_mae_pct": guided["delta_ef_mae_pct"],
                    "random_delta_ef_mae_pct": random_group["delta_ef_mae_pct"],
                    "guided_minus_random_delta_ef_mae_pct": guided["delta_ef_mae_pct"] - random_group["delta_ef_mae_pct"],
                }
            )
    summary["attention_guided_vs_random"] = attention_vs_random
    return summary


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    set_seed(args.seed)
    device = args.device if args.device else config.DEVICE
    if str(device).startswith("cuda") and not torch.cuda.is_available():
        device = "cpu"

    perturbations = parse_csv_list(args.perturbations, str)
    severities = parse_csv_list(args.severities, float)

    invalid = [p for p in perturbations if p not in AVAILABLE_PERTURBATIONS]
    if invalid:
        raise ValueError(f"Unknown perturbations requested: {invalid}")
    if not severities:
        raise ValueError("At least one severity must be provided")

    dataset = make_dataset(args)
    model = load_model(args.checkpoint, args.num_frames, device)

    rows = []
    plots_saved = 0

    print("=" * 96)
    print("TEMPORAL PERTURBATION EVALUATION")
    print("=" * 96)
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Split: {args.split}")
    print(f"Samples: {len(dataset)}")
    print(f"Frames: {args.num_frames}")
    print(f"Dataset period: {args.dataset_period}")
    print(f"Perturbations: {', '.join(perturbations)}")
    print(f"Severities: {', '.join(f'{s:.2f}' for s in severities)}")
    print(f"Device: {device}")
    print("=" * 96)

    for sample_index in range(len(dataset)):
        row = dataset.filelist.iloc[sample_index]
        file_name = str(row["FileName"]).strip() + ".avi"
        video_path = os.path.join(dataset.data_dir, "Videos", file_name)
        ed_orig = int(dataset.phase_dict[file_name]["ed"])
        es_orig = int(dataset.phase_dict[file_name]["es"])
        ef_gt_pct = float(row["EF"])

        clip, sampled_indices = dataset.load_video(video_path, ed_original=ed_orig, es_original=es_orig)
        sampled_indices = np.asarray(sampled_indices, dtype=np.int32)

        gt_ed_clip_idx = int(np.argmin(np.abs(sampled_indices - ed_orig))) if ed_orig >= 0 else 0
        gt_es_clip_idx = int(np.argmin(np.abs(sampled_indices - es_orig))) if es_orig >= 0 else 0

        clean_result = run_inference(model, clip, device)
        clean_ef_abs_error = abs(clean_result["ef_pred_pct"] - ef_gt_pct)
        clean_ed_abs_error = abs(clean_result["pred_ed_idx"] - gt_ed_clip_idx)
        clean_es_abs_error = abs(clean_result["pred_es_idx"] - gt_es_clip_idx)
        clean_joint_within_tol = int(clean_ed_abs_error <= config.TOLERANCE and clean_es_abs_error <= config.TOLERANCE)

        for perturbation in perturbations:
            for severity in severities:
                rng = np.random.default_rng(args.seed + sample_index * 1009 + int(round(severity * 1000)) * 17 + sum(ord(c) for c in perturbation))
                perturbed_clip, metadata = apply_temporal_perturbation(
                    clip,
                    perturbation=perturbation,
                    severity=severity,
                    rng=rng,
                    frame_scores=clean_result["attention_summary"],
                )
                pert_result = run_inference(model, perturbed_clip, device)

                pert_ef_abs_error = abs(pert_result["ef_pred_pct"] - ef_gt_pct)
                pert_ed_abs_error = abs(pert_result["pred_ed_idx"] - gt_ed_clip_idx)
                pert_es_abs_error = abs(pert_result["pred_es_idx"] - gt_es_clip_idx)
                pert_joint_within_tol = int(pert_ed_abs_error <= config.TOLERANCE and pert_es_abs_error <= config.TOLERANCE)

                result_row = {
                    "sample_index": int(sample_index),
                    "file_name": file_name,
                    "perturbation": perturbation,
                    "severity": float(severity),
                    "ef_gt_pct": float(ef_gt_pct),
                    "gt_ed_clip_idx": int(gt_ed_clip_idx),
                    "gt_es_clip_idx": int(gt_es_clip_idx),
                    "gt_ed_orig_frame": int(ed_orig),
                    "gt_es_orig_frame": int(es_orig),
                    "clean_pred_ed_idx": int(clean_result["pred_ed_idx"]),
                    "clean_pred_es_idx": int(clean_result["pred_es_idx"]),
                    "clean_pred_ed_orig_frame": int(sampled_indices[clean_result["pred_ed_idx"]]),
                    "clean_pred_es_orig_frame": int(sampled_indices[clean_result["pred_es_idx"]]),
                    "perturbed_pred_ed_idx": int(pert_result["pred_ed_idx"]),
                    "perturbed_pred_es_idx": int(pert_result["pred_es_idx"]),
                    "perturbed_pred_ed_orig_frame": int(sampled_indices[pert_result["pred_ed_idx"]]),
                    "perturbed_pred_es_orig_frame": int(sampled_indices[pert_result["pred_es_idx"]]),
                    "clean_ef_pred_pct": float(clean_result["ef_pred_pct"]),
                    "perturbed_ef_pred_pct": float(pert_result["ef_pred_pct"]),
                    "clean_ef_abs_error_pct": float(clean_ef_abs_error),
                    "perturbed_ef_abs_error_pct": float(pert_ef_abs_error),
                    "delta_ef_abs_error_pct": float(pert_ef_abs_error - clean_ef_abs_error),
                    "clean_ed_abs_error_frames": float(clean_ed_abs_error),
                    "clean_es_abs_error_frames": float(clean_es_abs_error),
                    "perturbed_ed_abs_error_frames": float(pert_ed_abs_error),
                    "perturbed_es_abs_error_frames": float(pert_es_abs_error),
                    "delta_ed_abs_error_frames": float(pert_ed_abs_error - clean_ed_abs_error),
                    "delta_es_abs_error_frames": float(pert_es_abs_error - clean_es_abs_error),
                    "clean_joint_within_tol": int(clean_joint_within_tol),
                    "perturbed_joint_within_tol": int(pert_joint_within_tol),
                    "clean_pred_ed_conf": float(clean_result["pred_ed_conf"]),
                    "clean_pred_es_conf": float(clean_result["pred_es_conf"]),
                    "perturbed_pred_ed_conf": float(pert_result["pred_ed_conf"]),
                    "perturbed_pred_es_conf": float(pert_result["pred_es_conf"]),
                    "clean_attention_peak_idx": int(np.argmax(clean_result["attention_summary"])) if clean_result["attention_summary"].size > 0 else -1,
                    "perturbed_attention_peak_idx": int(np.argmax(pert_result["attention_summary"])) if pert_result["attention_summary"].size > 0 else -1,
                    "perturbation_metadata_json": json.dumps(metadata, ensure_ascii=True, sort_keys=True),
                }
                rows.append(result_row)

                if plots_saved < int(max(0, args.plots)):
                    save_plot_set(str(output_dir), plots_saved, result_row, clean_result, pert_result)
                    plots_saved += 1

    if not rows:
        raise RuntimeError("No perturbation rows were produced")

    csv_path = output_dir / "temporal_perturbation_results.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    summary = build_summary(rows, bootstrap_samples=int(args.bootstrap_samples), seed=int(args.seed))
    summary["config"] = {
        "checkpoint": os.path.abspath(args.checkpoint),
        "split": str(args.split),
        "num_frames": int(args.num_frames),
        "dataset_period": int(args.dataset_period),
        "dataset_max_length": args.dataset_max_length,
        "max_videos": int(args.max_videos) if args.max_videos is not None else None,
        "perturbations": perturbations,
        "severities": [float(s) for s in severities],
        "seed": int(args.seed),
        "device": str(device),
        "plots_saved": int(plots_saved),
    }

    summary_path = output_dir / "temporal_perturbation_summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("=" * 96)
    print("TEMPORAL PERTURBATION SUMMARY")
    print("=" * 96)
    for group in summary["groups"]:
        print(
            f"{group['perturbation']} | severity={group['severity']:.2f} | "
            f"clean EF MAE={group['clean_ef_mae_pct']:.3f} | "
            f"pert EF MAE={group['perturbed_ef_mae_pct']:.3f} | "
            f"delta={group['delta_ef_mae_pct']:.3f}"
        )
    if summary["attention_guided_vs_random"]:
        print("-" * 96)
        for item in summary["attention_guided_vs_random"]:
            print(
                f"attention_guided vs random | severity={item['severity']:.2f} | "
                f"guided-random delta EF MAE={item['guided_minus_random_delta_ef_mae_pct']:.3f}"
            )
    print(f"Saved CSV: {csv_path}")
    print(f"Saved summary: {summary_path}")
    print("=" * 96)


if __name__ == "__main__":
    main()
