import argparse
import os
import random
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader

# Allow running from the visualization folder directly
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import config
from data.dataset import EchoDataset
from models.ef_model import load_ef_model_from_checkpoint
from visualization.visualize_attention import plot_attention, plot_phase_curves


def compute_phase_metrics(model, loader, device):
    """Return aggregate ED/ES error and tolerance-based accuracy metrics."""
    model.eval()

    ed_errors = []
    es_errors = []

    with torch.no_grad():
        for videos, _, ed_idx, es_idx in loader:
            videos = videos.to(device)
            ed_idx = ed_idx.to(device)
            es_idx = es_idx.to(device)

            _, _, phase_logits = model(videos)

            pred_ed = torch.argmax(phase_logits[:, :, 1], dim=1)
            pred_es = torch.argmax(phase_logits[:, :, 2], dim=1)

            ed_errors.extend(torch.abs(pred_ed - ed_idx).cpu().numpy().tolist())
            es_errors.extend(torch.abs(pred_es - es_idx).cpu().numpy().tolist())

    ed_errors = np.array(ed_errors, dtype=np.float32)
    es_errors = np.array(es_errors, dtype=np.float32)

    def within_tol(errors, tol):
        return float(np.mean(errors <= tol)) if len(errors) else 0.0

    metrics = {
        "num_samples": int(len(ed_errors)),
        "ed_mae_frames": float(np.mean(ed_errors)) if len(ed_errors) else 0.0,
        "es_mae_frames": float(np.mean(es_errors)) if len(es_errors) else 0.0,
        "joint_within_1": float(np.mean((ed_errors <= 1) & (es_errors <= 1))) if len(ed_errors) else 0.0,
        "joint_within_2": float(np.mean((ed_errors <= 2) & (es_errors <= 2))) if len(ed_errors) else 0.0,
        "ed_within_0": within_tol(ed_errors, 0),
        "ed_within_1": within_tol(ed_errors, 1),
        "ed_within_2": within_tol(ed_errors, 2),
        "es_within_0": within_tol(es_errors, 0),
        "es_within_1": within_tol(es_errors, 1),
        "es_within_2": within_tol(es_errors, 2),
    }
    return metrics


def save_sample_visualizations(model, dataset, out_dir, num_samples, device, seed=42):
    """Save attention and phase-curve plots for random test samples."""
    os.makedirs(out_dir, exist_ok=True)

    indices = list(range(len(dataset)))
    random.Random(seed).shuffle(indices)
    indices = indices[: min(num_samples, len(indices))]

    model.eval()
    with torch.no_grad():
        for sample_id, idx in enumerate(indices):
            video, _, ed_idx, es_idx = dataset[idx]
            video = video.unsqueeze(0).to(device)

            _, attn, phase_logits = model(video)
            phase_probs = torch.softmax(phase_logits, dim=-1)[0].cpu().numpy()
            attn = attn[0].cpu().numpy()

            pred_ed = int(np.argmax(phase_probs[:, 1]))
            pred_es = int(np.argmax(phase_probs[:, 2]))

            sample_prefix = os.path.join(out_dir, f"sample_{sample_id:03d}_idx_{idx}")

            plot_attention(
                attn=attn,
                ed_idx=int(ed_idx),
                es_idx=int(es_idx),
                pred_ed_idx=pred_ed,
                pred_es_idx=pred_es,
                save_path=f"{sample_prefix}_attention.png",
            )

            plot_phase_curves(
                phase_probs=phase_probs,
                ed_idx=int(ed_idx),
                es_idx=int(es_idx),
                pred_ed_idx=pred_ed,
                pred_es_idx=pred_es,
                save_path=f"{sample_prefix}_phase_curves.png",
            )


def load_model(checkpoint_path, device):
    model, _, _ = load_ef_model_from_checkpoint(
        checkpoint_path=checkpoint_path,
        num_frames=config.NUM_FRAMES,
        device=device,
        default_preserve_temporal_stride=bool(getattr(config, "STAGE1_PRESERVE_TEMPORAL_STRIDE", True)),
    )
    return model


def main():
    parser = argparse.ArgumentParser(description="Validate and visualize phase detection quality.")
    parser.add_argument("--split", type=str, default="TEST", help="Dataset split: TRAIN/VAL/TEST")
    parser.add_argument("--checkpoint", type=str, default=config.CHECKPOINT_PATH, help="Model checkpoint path")
    parser.add_argument("--batch-size", type=int, default=config.BATCH_SIZE, help="Batch size for validation")
    parser.add_argument("--dataset-period", type=int, default=int(getattr(config, "DATASET_PERIOD", 1)), help="Temporal stride within each sampled clip")
    parser.add_argument("--dataset-max-length", type=int, default=getattr(config, "DATASET_MAX_LENGTH", None), help="Optional max clip length when length is not fixed")
    parser.add_argument("--num-samples", type=int, default=8, help="Number of samples to visualize")
    parser.add_argument(
        "--output-dir",
        type=str,
        default=os.path.join("visualization", "outputs", "phase_validation"),
        help="Directory to save plots",
    )
    args = parser.parse_args()

    device = config.DEVICE

    dataset = EchoDataset(
        data_dir=config.DATA_DIR,
        split=args.split,
        num_frames=config.NUM_FRAMES,
        max_videos=config.MAX_VIDEOS,
        period=int(args.dataset_period),
        max_length=args.dataset_max_length,
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    model = load_model(args.checkpoint, device)

    metrics = compute_phase_metrics(model, loader, device)
    save_sample_visualizations(
        model=model,
        dataset=dataset,
        out_dir=args.output_dir,
        num_samples=args.num_samples,
        device=device,
    )

    print("=" * 70)
    print("PHASE VALIDATION SUMMARY")
    print("=" * 70)
    print(f"Samples:           {metrics['num_samples']}")
    print(f"ED MAE (frames):   {metrics['ed_mae_frames']:.3f}")
    print(f"ES MAE (frames):   {metrics['es_mae_frames']:.3f}")
    print(f"ED within +/-0:    {metrics['ed_within_0'] * 100:.2f}%")
    print(f"ED within +/-1:    {metrics['ed_within_1'] * 100:.2f}%")
    print(f"ED within +/-2:    {metrics['ed_within_2'] * 100:.2f}%")
    print(f"ES within +/-0:    {metrics['es_within_0'] * 100:.2f}%")
    print(f"ES within +/-1:    {metrics['es_within_1'] * 100:.2f}%")
    print(f"ES within +/-2:    {metrics['es_within_2'] * 100:.2f}%")
    print(f"Joint +/-1:        {metrics['joint_within_1'] * 100:.2f}%")
    print(f"Joint +/-2:        {metrics['joint_within_2'] * 100:.2f}%")
    print(f"Plots saved to:    {os.path.abspath(args.output_dir)}")
    print("=" * 70)


if __name__ == "__main__":
    main()
