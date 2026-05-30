import argparse
import csv
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import config
from data.dataset import EchoDataset
from models.ef_model import EFModel, infer_ef_head_arch
from pipeline.stage3_phase_detector import Stage3PhaseDetector


def parse_args():
    parser = argparse.ArgumentParser(description="Validate Stage1-3 EF and ED/ES phase predictions.")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--data-dir", type=str, default=config.DATA_DIR)
    parser.add_argument("--split", type=str, default="TEST", choices=["TRAIN", "VAL", "TEST"])
    parser.add_argument("--num-frames", type=int, default=int(getattr(config, "NUM_FRAMES", 64)))
    parser.add_argument("--max-videos", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--clip-period", type=int, default=int(getattr(config, "CLIP_PERIOD", 1)))
    parser.add_argument("--clip-eval-mode", type=str, choices=["center", "all"], default="center")
    parser.add_argument("--phase-tolerance", type=int, default=4)
    parser.add_argument("--output-dir", type=str, default=os.path.join("validation", "outputs", "stage123_validation"))
    parser.add_argument("--device", type=str, default=("cuda" if torch.cuda.is_available() else "cpu"))
    return parser.parse_args()


def load_model(checkpoint_path, num_frames, device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint.get("model_state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint
    model = EFModel(num_frames=int(num_frames), ef_head_arch=infer_ef_head_arch(state_dict)).to(device)
    model_state = model.state_dict()
    filtered = {
        key: value
        for key, value in state_dict.items()
        if key in model_state and tuple(value.shape) == tuple(model_state[key].shape)
    }
    incompatible = model.load_state_dict(filtered, strict=False)
    model.eval()
    return model, incompatible


def _phase_confidence(phase_output, pred_idx, channel):
    if phase_output.ndim == 2 and phase_output.shape[-1] >= 3:
        probs = torch.softmax(phase_output[:, channel], dim=0)
        return float(probs[int(pred_idx)].detach().cpu().item())
    if phase_output.ndim == 1:
        phase = phase_output.detach().float()
        score = -torch.minimum(torch.abs(phase), torch.abs(phase - 1.0)) if channel == 1 else -torch.abs(phase - 0.5)
        probs = torch.softmax(score, dim=0)
        return float(probs[int(pred_idx)].cpu().item())
    return float("nan")


def main():
    args = parse_args()
    device = torch.device(args.device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset = EchoDataset(
        data_dir=args.data_dir,
        split=args.split,
        num_frames=args.num_frames,
        max_videos=args.max_videos,
        normalize_input=bool(getattr(config, "NORMALIZE_INPUT", True)),
        clip_period=args.clip_period,
        clip_eval_mode=args.clip_eval_mode,
    )
    loader = DataLoader(
        dataset,
        batch_size=max(1, int(args.batch_size)),
        shuffle=False,
        num_workers=max(0, int(args.workers)),
        pin_memory=torch.cuda.is_available() and str(device).startswith("cuda"),
    )

    model, incompatible = load_model(args.checkpoint, args.num_frames, device)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        print(
            f"Warning: checkpoint loaded with key mismatch | missing={len(incompatible.missing_keys)} unexpected={len(incompatible.unexpected_keys)}"
        )

    rows = []
    sample_offset = 0
    with torch.no_grad():
        for videos, efs, ed_idx, es_idx in loader:
            videos = videos.to(device)
            efs = efs.to(device)
            ed_idx = ed_idx.to(device)
            es_idx = es_idx.to(device)

            ef_pred, attention, phase_pred = model(videos)
            pred_ed, pred_es = Stage3PhaseDetector.predict_indices(phase_pred)

            for b in range(videos.shape[0]):
                row_meta = dataset.filelist.iloc[sample_offset + b]
                file_name = str(row_meta["FileName"]).strip() + ".avi"
                ef_gt_pct = float(efs[b].detach().cpu().item() * 100.0)
                ef_pred_pct = float(ef_pred[b].detach().cpu().item() * 100.0)
                ed_err = abs(int(pred_ed[b].item()) - int(ed_idx[b].item()))
                es_err = abs(int(pred_es[b].item()) - int(es_idx[b].item()))
                attn_b = attention[b].detach().float()
                if attn_b.ndim == 2:
                    attn_summary = attn_b.mean(dim=-1)
                else:
                    attn_summary = attn_b.reshape(-1)

                rows.append(
                    {
                        "sample_index": sample_offset + b,
                        "file_name": file_name,
                        "ef_gt_pct": ef_gt_pct,
                        "ef_pred_pct": ef_pred_pct,
                        "ef_abs_error_pct": abs(ef_pred_pct - ef_gt_pct),
                        "gt_ed_idx": int(ed_idx[b].item()),
                        "gt_es_idx": int(es_idx[b].item()),
                        "pred_ed_idx": int(pred_ed[b].item()),
                        "pred_es_idx": int(pred_es[b].item()),
                        "ed_abs_error_frames": float(ed_err),
                        "es_abs_error_frames": float(es_err),
                        "joint_within_tol": int(ed_err <= args.phase_tolerance and es_err <= args.phase_tolerance),
                        "ed_conf": _phase_confidence(phase_pred[b], pred_ed[b], 1),
                        "es_conf": _phase_confidence(phase_pred[b], pred_es[b], 2),
                        "attention_peak_idx": int(torch.argmax(attn_summary).detach().cpu().item()),
                    }
                )
            sample_offset += videos.shape[0]

    if not rows:
        raise RuntimeError("No validation samples were processed")

    ef_errors = np.array([r["ef_abs_error_pct"] for r in rows], dtype=np.float64)
    ed_errors = np.array([r["ed_abs_error_frames"] for r in rows], dtype=np.float64)
    es_errors = np.array([r["es_abs_error_frames"] for r in rows], dtype=np.float64)
    joint = np.array([r["joint_within_tol"] for r in rows], dtype=np.float64)

    summary = {
        "checkpoint": os.path.abspath(args.checkpoint),
        "split": args.split,
        "num_frames": int(args.num_frames),
        "clip_period": int(args.clip_period),
        "clip_eval_mode": args.clip_eval_mode,
        "samples": len(rows),
        "ef_mae_pct": float(ef_errors.mean()),
        "ef_rmse_pct": float(np.sqrt(np.mean(ef_errors ** 2))),
        "ed_mae_frames": float(ed_errors.mean()),
        "es_mae_frames": float(es_errors.mean()),
        "ed_acc_within_tol": float(np.mean(ed_errors <= args.phase_tolerance)),
        "es_acc_within_tol": float(np.mean(es_errors <= args.phase_tolerance)),
        "joint_acc_within_tol": float(joint.mean()),
        "phase_tolerance": int(args.phase_tolerance),
    }

    csv_path = output_dir / f"stage123_{args.split.lower()}_validation.csv"
    json_path = output_dir / f"stage123_{args.split.lower()}_summary.json"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("=" * 90)
    print("STAGE1-3 VALIDATION SUMMARY")
    print("=" * 90)
    print(f"Split: {args.split} | Samples: {len(rows)} | Frames: {args.num_frames}")
    print(f"EF MAE/RMSE: {summary['ef_mae_pct']:.2f}% / {summary['ef_rmse_pct']:.2f}%")
    print(f"ED/ES MAE(fr): {summary['ed_mae_frames']:.3f} / {summary['es_mae_frames']:.3f}")
    print(
        f"Phase Acc @ +/-{args.phase_tolerance}: ED {summary['ed_acc_within_tol'] * 100:.2f}% | "
        f"ES {summary['es_acc_within_tol'] * 100:.2f}% | Joint {summary['joint_acc_within_tol'] * 100:.2f}%"
    )
    print(f"Saved rows: {csv_path}")
    print(f"Saved summary: {json_path}")


if __name__ == "__main__":
    main()
