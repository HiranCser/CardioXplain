import argparse
import csv
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

import config
from data.dataset import EchoDataset
from models.ef_model import load_ef_model_from_checkpoint


def _window_bounds(center, radius, num_frames):
    left = max(0, int(center) - int(radius))
    right = min(int(num_frames) - 1, int(center) + int(radius))
    return left, right


def _window_mass(attn, center, radius):
    left, right = _window_bounds(center, radius, attn.shape[0])
    return float(attn[left : right + 1].sum())


def _rank_of_index(attn, index):
    order = np.argsort(-attn)
    return int(np.where(order == int(index))[0][0]) + 1


def parse_args():
    parser = argparse.ArgumentParser(description="Validate Stage-2 temporal attention against ED/ES labels.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--split", type=str, default="VAL", choices=["TRAIN", "VAL", "TEST"], help="Dataset split")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--num-frames", type=int, default=config.NUM_FRAMES)
    parser.add_argument("--dataset-period", type=int, default=int(getattr(config, "DATASET_PERIOD", 1)))
    parser.add_argument("--dataset-max-length", type=int, default=getattr(config, "DATASET_MAX_LENGTH", None))
    parser.add_argument("--max-videos", type=int, default=None)
    parser.add_argument("--radius", type=int, default=1, help="Neighborhood radius around ED/ES for attention mass")
    parser.add_argument("--topk", type=int, default=3, help="Top-k for hit-rate metrics")
    parser.add_argument("--phase-temporal-window-mode", type=str, default="tracing", choices=["full", "tracing"])
    parser.add_argument("--phase-temporal-window-margin-mult", type=float, default=1.5)
    parser.add_argument("--phase-temporal-window-jitter-mult", type=float, default=0.0)
    parser.add_argument("--output", type=str, default="validation/outputs/stage2_attention_validation.csv")
    parser.add_argument("--device", type=str, default=None, help="cuda or cpu; default uses config")
    return parser.parse_args()


def main():
    args = parse_args()
    device = args.device if args.device else config.DEVICE

    dataset = EchoDataset(
        data_dir=config.DATA_DIR,
        split=args.split,
        num_frames=args.num_frames,
        period=int(args.dataset_period),
        max_length=args.dataset_max_length,
        max_videos=args.max_videos,
        normalize_input=bool(getattr(config, "NORMALIZE_INPUT", True)),
        temporal_window_mode=args.phase_temporal_window_mode,
        temporal_window_margin_mult=args.phase_temporal_window_margin_mult,
        temporal_window_jitter_mult=args.phase_temporal_window_jitter_mult,
    )

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=max(0, int(args.workers)),
        pin_memory=torch.cuda.is_available() and str(device).startswith("cuda"),
    )

    model, incompatible, _ = load_ef_model_from_checkpoint(
        checkpoint_path=args.checkpoint,
        num_frames=args.num_frames,
        device=device,
        default_preserve_temporal_stride=bool(getattr(config, "STAGE1_PRESERVE_TEMPORAL_STRIDE", True)),
    )
    if incompatible.missing_keys or incompatible.unexpected_keys:
        print(
            f"Warning: checkpoint loaded with key mismatch | missing={len(incompatible.missing_keys)} unexpected={len(incompatible.unexpected_keys)}"
        )

    rows = []
    sample_offset = 0

    with torch.no_grad():
        for videos, _efs, ed_idx, es_idx in loader:
            videos = videos.to(device)
            _ef, attention, _phase = model(videos)

            attn_np = attention.detach().cpu().numpy()
            if attn_np.ndim == 3 and attn_np.shape[-1] > 1:
                attn_np = attn_np.mean(axis=2)
            elif attn_np.ndim == 3 and attn_np.shape[-1] == 1:
                attn_np = attn_np[:, :, 0]
            ed_np = ed_idx.cpu().numpy().astype(int)
            es_np = es_idx.cpu().numpy().astype(int)

            for i in range(attn_np.shape[0]):
                attn = attn_np[i]
                ed = int(ed_np[i])
                es = int(es_np[i])
                topk_idx = np.argsort(-attn)[: max(1, int(args.topk))]

                ed_mass = _window_mass(attn, ed, args.radius)
                es_mass = _window_mass(attn, es, args.radius)

                target_mask = np.zeros_like(attn, dtype=bool)
                ed_l, ed_r = _window_bounds(ed, args.radius, attn.shape[0])
                es_l, es_r = _window_bounds(es, args.radius, attn.shape[0])
                target_mask[ed_l : ed_r + 1] = True
                target_mask[es_l : es_r + 1] = True
                target_mass = float(attn[target_mask].sum())

                file_name = str(dataset.filelist.iloc[sample_offset + i]["FileName"]) + ".avi"

                rows.append(
                    {
                        "sample_index": sample_offset + i,
                        "file_name": file_name,
                        "ed_idx": ed,
                        "es_idx": es,
                        "attn_ed": float(attn[ed]),
                        "attn_es": float(attn[es]),
                        "attn_ed_window": ed_mass,
                        "attn_es_window": es_mass,
                        "attn_target_union": target_mass,
                        "rank_ed": _rank_of_index(attn, ed),
                        "rank_es": _rank_of_index(attn, es),
                        "top1": int(np.argmax(attn)),
                        "top1_is_ed_or_es": int(int(np.argmax(attn)) in (ed, es)),
                        f"top{int(args.topk)}_has_ed": int(ed in topk_idx),
                        f"top{int(args.topk)}_has_es": int(es in topk_idx),
                        f"top{int(args.topk)}_has_either": int((ed in topk_idx) or (es in topk_idx)),
                    }
                )

            sample_offset += attn_np.shape[0]

    if not rows:
        raise RuntimeError("No samples processed. Check dataset/filter settings.")

    arr = {k: np.array([row[k] for row in rows]) for k in rows[0].keys() if k not in ("file_name",)}

    target_slots = []
    for row in rows:
        t = len(set(range(max(0, row["ed_idx"] - args.radius), min(args.num_frames - 1, row["ed_idx"] + args.radius) + 1)).union(
            set(range(max(0, row["es_idx"] - args.radius), min(args.num_frames - 1, row["es_idx"] + args.radius) + 1))
        ))
        target_slots.append(t)
    baseline_mass = float(np.mean(np.array(target_slots, dtype=np.float64) / float(args.num_frames)))

    print("=" * 90)
    print("STAGE-2 ATTENTION VALIDATION SUMMARY")
    print("=" * 90)
    print(f"Samples: {len(rows)}")
    print(f"Split: {args.split}")
    print(f"Frames: {args.num_frames} | Radius: +/-{args.radius} | Top-k: {args.topk}")
    print(f"Mean attention at ED idx: {arr['attn_ed'].mean():.4f}")
    print(f"Mean attention at ES idx: {arr['attn_es'].mean():.4f}")
    print(f"Mean ED window mass: {arr['attn_ed_window'].mean():.4f}")
    print(f"Mean ES window mass: {arr['attn_es_window'].mean():.4f}")
    print(f"Mean ED/ES union mass: {arr['attn_target_union'].mean():.4f}")
    print(f"Random baseline union mass: {baseline_mass:.4f}")
    print(f"Top-1 is ED or ES: {arr['top1_is_ed_or_es'].mean() * 100:.2f}%")
    print(f"Top-{args.topk} has ED: {arr[f'top{int(args.topk)}_has_ed'].mean() * 100:.2f}%")
    print(f"Top-{args.topk} has ES: {arr[f'top{int(args.topk)}_has_es'].mean() * 100:.2f}%")
    print(f"Top-{args.topk} has either ED/ES: {arr[f'top{int(args.topk)}_has_either'].mean() * 100:.2f}%")
    print(f"Median rank ED: {np.median(arr['rank_ed']):.1f}")
    print(f"Median rank ES: {np.median(arr['rank_es']):.1f}")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved per-sample metrics: {output_path}")


if __name__ == "__main__":
    main()
