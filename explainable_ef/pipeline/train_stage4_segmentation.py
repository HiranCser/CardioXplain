import argparse
import contextlib
import os
import sys
import time
import shutil

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import config
from data.stage4_segmentation_dataset import Stage4SegmentationDataset
from models.stage4_segmentation_model import build_stage4_segmentation_model


def extract_logits(model_output):
    if isinstance(model_output, dict):
        if "out" not in model_output:
            raise ValueError("Segmentation model output dict does not contain 'out'")
        return model_output["out"]
    return model_output


def autocast_context(device, amp_enabled):
    if amp_enabled and str(device).startswith("cuda") and torch.cuda.is_available():
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    return contextlib.nullcontext()


def dice_from_logits(logits, targets, eps=1e-6, threshold=0.5, soft=False):
    probs = torch.sigmoid(logits)
    preds = probs if soft else (probs >= float(threshold)).float()
    inter = (preds * targets).sum(dim=(1, 2, 3))
    union = preds.sum(dim=(1, 2, 3)) + targets.sum(dim=(1, 2, 3))
    return ((2.0 * inter + eps) / (union + eps)).mean()


def _resolve_pos_weight(targets, manual_pos_weight=None, pos_weight_max=20.0):
    if manual_pos_weight is not None:
        v = max(1.0, float(manual_pos_weight))
        return torch.tensor(v, device=targets.device, dtype=targets.dtype)

    pos = targets.sum()
    total = torch.tensor(float(targets.numel()), device=targets.device, dtype=targets.dtype)
    neg = total - pos
    ratio = torch.sqrt(neg / (pos + 1e-6))
    ratio = torch.clamp(ratio, min=1.0, max=float(pos_weight_max))
    return ratio.detach()


def segmentation_loss(logits, targets, dice_weight=1.0, manual_pos_weight=None, pos_weight_max=20.0, area_loss_weight=0.0):
    pos_weight = _resolve_pos_weight(targets, manual_pos_weight=manual_pos_weight, pos_weight_max=pos_weight_max)
    bce = F.binary_cross_entropy_with_logits(logits, targets, pos_weight=pos_weight)

    probs = torch.sigmoid(logits)
    inter = (probs * targets).sum(dim=(1, 2, 3))
    union = probs.sum(dim=(1, 2, 3)) + targets.sum(dim=(1, 2, 3))
    dice = (2.0 * inter + 1e-6) / (union + 1e-6)
    dice_loss = 1.0 - dice.mean()

    pred_area = probs.sum(dim=(1, 2, 3))
    target_area = targets.sum(dim=(1, 2, 3))
    area_loss = torch.mean(torch.abs(pred_area - target_area) / target_area.clamp_min(1.0))

    loss = bce + float(dice_weight) * dice_loss + float(area_loss_weight) * area_loss
    return loss, bce.detach(), dice_loss.detach(), area_loss.detach(), float(pos_weight.item())


def configure_dataloader_runtime():
    sharing_strategy = None
    if sys.platform.startswith("linux"):
        try:
            torch.multiprocessing.set_sharing_strategy("file_system")
            sharing_strategy = "file_system"
        except Exception:
            sharing_strategy = None
    return sharing_strategy


def dataloader_kwargs(batch_size, workers, shuffle, device, pin_memory, persistent_workers, prefetch_factor):
    use_cuda = str(device).startswith("cuda") and torch.cuda.is_available()
    num_workers = int(max(0, workers))
    kwargs = {
        "batch_size": int(batch_size),
        "shuffle": bool(shuffle),
        "num_workers": num_workers,
        "pin_memory": bool(pin_memory) and use_cuda,
    }
    if num_workers > 0:
        kwargs["persistent_workers"] = bool(persistent_workers)
        if prefetch_factor is not None:
            kwargs["prefetch_factor"] = int(prefetch_factor)
    return kwargs


def build_loaders(args, device):
    normalize = args.normalize
    if normalize == "auto":
        normalize = "imagenet" if args.pretrained else "none"

    train_ds = Stage4SegmentationDataset(
        data_dir=args.data_dir,
        split="TRAIN",
        image_size=args.image_size,
        max_videos=args.max_videos,
        normalize=normalize,
        augment=args.augment,
    )
    val_ds = Stage4SegmentationDataset(
        data_dir=args.data_dir,
        split="VAL",
        image_size=args.image_size,
        max_videos=args.max_videos,
        normalize=normalize,
        augment=False,
    )
    test_ds = Stage4SegmentationDataset(
        data_dir=args.data_dir,
        split="TEST",
        image_size=args.image_size,
        max_videos=args.max_videos,
        normalize=normalize,
        augment=False,
    )

    train_loader = DataLoader(
        train_ds,
        **dataloader_kwargs(
            args.batch_size,
            args.workers,
            True,
            device,
            pin_memory=args.pin_memory,
            persistent_workers=args.persistent_workers,
            prefetch_factor=args.prefetch_factor,
        ),
    )
    val_loader = DataLoader(
        val_ds,
        **dataloader_kwargs(
            args.batch_size,
            args.workers,
            False,
            device,
            pin_memory=args.pin_memory,
            persistent_workers=args.persistent_workers,
            prefetch_factor=args.prefetch_factor,
        ),
    )
    test_loader = DataLoader(
        test_ds,
        **dataloader_kwargs(
            args.batch_size,
            args.workers,
            False,
            device,
            pin_memory=args.pin_memory,
            persistent_workers=args.persistent_workers,
            prefetch_factor=args.prefetch_factor,
        ),
    )
    return train_loader, val_loader, test_loader, normalize


def summarize_area_rows(rows, csv_path):
    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    df.to_csv(csv_path, index=False)

    video_summary = (
        df.groupby("file_name", as_index=False)
        .agg(
            frames=("frame_id", "count"),
            gt_area_mean=("gt_area", "mean"),
            pred_area_mean=("pred_area", "mean"),
            abs_error_mean=("abs_error", "mean"),
            pct_error_mean=("pct_error", "mean"),
        )
        .sort_values("file_name")
    )
    video_summary.to_csv(csv_path.replace(".csv", "_video_summary.csv"), index=False)


def evaluate(
    model,
    loader,
    device,
    amp_enabled,
    dice_weight,
    eval_threshold,
    manual_pos_weight,
    pos_weight_max,
    area_loss_weight,
    csv_path=None,
):
    model.eval()

    total_loss = 0.0
    total_bce = 0.0
    total_dice_loss = 0.0
    total_area_loss = 0.0
    total_dice_hard = 0.0
    total_dice_soft = 0.0
    total_pos_weight = 0.0
    total_pred_fg_frac = 0.0
    total_gt_fg_frac = 0.0
    n_batches = 0

    area_abs_errors = []
    area_pct_errors = []
    rows = []

    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device)
            masks = batch["mask"].to(device)

            with autocast_context(device, amp_enabled):
                logits = extract_logits(model(images))
                loss, bce, dice_loss, area_loss, pos_weight_value = segmentation_loss(
                    logits,
                    masks,
                    dice_weight=dice_weight,
                    manual_pos_weight=manual_pos_weight,
                    pos_weight_max=pos_weight_max,
                    area_loss_weight=area_loss_weight,
                )

            dice_hard = dice_from_logits(logits, masks, threshold=eval_threshold, soft=False)
            dice_soft = dice_from_logits(logits, masks, threshold=eval_threshold, soft=True)

            probs = torch.sigmoid(logits)
            pred_masks = (probs >= float(eval_threshold)).to(torch.uint8).cpu().numpy()

            total_loss += float(loss.item())
            total_bce += float(bce.item())
            total_dice_loss += float(dice_loss.item())
            total_area_loss += float(area_loss.item())
            total_dice_hard += float(dice_hard.item())
            total_dice_soft += float(dice_soft.item())
            total_pos_weight += float(pos_weight_value)
            total_pred_fg_frac += float(pred_masks.mean())
            total_gt_fg_frac += float(masks.detach().float().mean().item())
            n_batches += 1

            for i in range(pred_masks.shape[0]):
                pred_small = pred_masks[i, 0]
                h = int(batch["frame_height"][i])
                w = int(batch["frame_width"][i])

                pred_orig = cv2.resize(pred_small, (w, h), interpolation=cv2.INTER_NEAREST)
                pred_area = float(pred_orig.sum())
                gt_area = float(batch["gt_area_orig"][i])

                abs_err = abs(pred_area - gt_area)
                pct_err = (abs_err / gt_area) if gt_area > 0 else 0.0

                area_abs_errors.append(abs_err)
                area_pct_errors.append(pct_err)

                rows.append(
                    {
                        "file_name": str(batch["file_name"][i]),
                        "file_name_ext": str(batch["file_name_ext"][i]),
                        "frame_id": int(batch["frame_id"][i]),
                        "gt_area": float(gt_area),
                        "pred_area": float(pred_area),
                        "abs_error": float(abs_err),
                        "pct_error": float(pct_err),
                        "eval_threshold": float(eval_threshold),
                    }
                )

    frame_area_mae = float(np.mean(area_abs_errors)) if area_abs_errors else float("nan")
    frame_area_mape = float(np.mean(area_pct_errors)) if area_pct_errors else float("nan")

    if csv_path is not None and rows:
        summarize_area_rows(rows, csv_path)

    return {
        "loss": total_loss / max(1, n_batches),
        "bce": total_bce / max(1, n_batches),
        "dice_loss": total_dice_loss / max(1, n_batches),
        "area_loss": total_area_loss / max(1, n_batches),
        "dice_hard": total_dice_hard / max(1, n_batches),
        "dice_soft": total_dice_soft / max(1, n_batches),
        "avg_pos_weight": total_pos_weight / max(1, n_batches),
        "pred_fg_frac": total_pred_fg_frac / max(1, n_batches),
        "gt_fg_frac": total_gt_fg_frac / max(1, n_batches),
        "frame_area_mae": frame_area_mae,
        "frame_area_mape": frame_area_mape,
        "samples": len(area_abs_errors),
    }


def train_one_epoch(
    model,
    loader,
    optimizer,
    device,
    amp_enabled,
    dice_weight,
    eval_threshold,
    manual_pos_weight,
    pos_weight_max,
    area_loss_weight,
):
    model.train()

    total_loss = 0.0
    total_bce = 0.0
    total_dice_loss = 0.0
    total_area_loss = 0.0
    total_dice_hard = 0.0
    total_dice_soft = 0.0
    total_pos_weight = 0.0
    total_pred_fg_frac = 0.0
    total_gt_fg_frac = 0.0
    n_batches = 0

    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)

    for batch in loader:
        images = batch["image"].to(device)
        masks = batch["mask"].to(device)

        optimizer.zero_grad(set_to_none=True)

        with autocast_context(device, amp_enabled):
            logits = extract_logits(model(images))
            loss, bce, dice_loss, area_loss, pos_weight_value = segmentation_loss(
                logits,
                masks,
                dice_weight=dice_weight,
                manual_pos_weight=manual_pos_weight,
                pos_weight_max=pos_weight_max,
                area_loss_weight=area_loss_weight,
            )

        if amp_enabled:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        probs = torch.sigmoid(logits)
        pred_masks = (probs >= float(eval_threshold)).float()

        dice_hard = ((2.0 * (pred_masks * masks).sum(dim=(1, 2, 3)) + 1e-6) /
                     (pred_masks.sum(dim=(1, 2, 3)) + masks.sum(dim=(1, 2, 3)) + 1e-6)).mean()
        dice_soft = dice_from_logits(logits, masks, threshold=eval_threshold, soft=True)

        total_loss += float(loss.item())
        total_bce += float(bce.item())
        total_dice_loss += float(dice_loss.item())
        total_area_loss += float(area_loss.item())
        total_dice_hard += float(dice_hard.item())
        total_dice_soft += float(dice_soft.item())
        total_pos_weight += float(pos_weight_value)
        total_pred_fg_frac += float(pred_masks.detach().mean().item())
        total_gt_fg_frac += float(masks.detach().mean().item())
        n_batches += 1

    return {
        "loss": total_loss / max(1, n_batches),
        "bce": total_bce / max(1, n_batches),
        "dice_loss": total_dice_loss / max(1, n_batches),
        "dice_hard": total_dice_hard / max(1, n_batches),
        "dice_soft": total_dice_soft / max(1, n_batches),
        "avg_pos_weight": total_pos_weight / max(1, n_batches),
        "pred_fg_frac": total_pred_fg_frac / max(1, n_batches),
        "gt_fg_frac": total_gt_fg_frac / max(1, n_batches),
    }


def _bytes_to_human(num_bytes):
    if num_bytes is None:
        return "unknown"
    value = float(num_bytes)
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if value < 1024.0 or unit == "TB":
            return f"{value:.2f} {unit}"
        value /= 1024.0
    return f"{value:.2f} TB"


def _disk_free_bytes(path):
    try:
        base = os.path.dirname(os.path.abspath(path)) or "."
        return int(shutil.disk_usage(base).free)
    except Exception:
        return None


def _atomic_torch_save(obj, path, use_new_zip=True):
    abs_path = os.path.abspath(path)
    save_dir = os.path.dirname(abs_path)
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    tmp_path = abs_path + ".tmp"
    try:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
    except OSError:
        pass

    torch.save(obj, tmp_path, _use_new_zipfile_serialization=bool(use_new_zip))
    os.replace(tmp_path, abs_path)


def _save_checkpoint_resilient(checkpoint_path, checkpoint_payload):
    payload_slim = dict(checkpoint_payload)
    payload_slim.pop("optimizer_state_dict", None)

    attempts = [
        ("full_zip", checkpoint_payload, True),
        ("full_legacy", checkpoint_payload, False),
        ("slim_zip", payload_slim, True),
        ("slim_legacy", payload_slim, False),
    ]

    last_error = None
    for save_mode, payload, use_new_zip in attempts:
        try:
            _atomic_torch_save(payload, checkpoint_path, use_new_zip=use_new_zip)
            return save_mode
        except Exception as exc:
            last_error = exc

    free_bytes = _disk_free_bytes(checkpoint_path)
    free_text = _bytes_to_human(free_bytes)
    raise RuntimeError(
        f"Checkpoint save failed after fallbacks at '{checkpoint_path}'. "
        f"Free disk near checkpoint: {free_text}. Last error: {last_error}"
    )

def count_parameters(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def build_optimizer(args, model):
    if args.optimizer == "sgd":
        return torch.optim.SGD(
            model.parameters(),
            lr=args.learning_rate,
            momentum=0.9,
            weight_decay=args.weight_decay,
        )
    return torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)


def parse_args():
    parser = argparse.ArgumentParser(description="Train Stage-4 LV segmentation and validate area against VolumeTracings.")
    parser.add_argument("--data-dir", type=str, default=config.DATA_DIR)
    parser.add_argument("--image-size", type=int, default=112)
    parser.add_argument("--batch-size", type=int, default=20)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--pin-memory", action=argparse.BooleanOptionalAction, default=(not sys.platform.startswith("linux")), help="Enable/disable DataLoader pin_memory. Defaults to off on Linux for Stage4 stability.")
    parser.add_argument("--persistent-workers", action=argparse.BooleanOptionalAction, default=False, help="Enable/disable DataLoader persistent workers.")
    parser.add_argument("--prefetch-factor", type=int, default=2, help="DataLoader prefetch factor when workers > 0")
    parser.add_argument("--max-videos", type=int, default=None)
    parser.add_argument("--dice-weight", type=float, default=1.0)
    parser.add_argument("--eval-threshold", type=float, default=0.5)
    parser.add_argument("--pos-weight", type=float, default=None, help="Manual positive-class BCE weight; if omitted, auto-computed per batch")
    parser.add_argument("--pos-weight-max", type=float, default=8.0)
    parser.add_argument("--area-loss-weight", type=float, default=0.20, help="Relative LV-area loss weight for segmentation training")
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--checkpoint", type=str, default="best_stage4_segmentation.pth")
    parser.add_argument("--output-dir", type=str, default=os.path.join("validation", "outputs", "stage4"))
    parser.add_argument("--device", type=str, default=("cuda" if torch.cuda.is_available() else "cpu"))
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--model-name", type=str, default="deeplabv3_resnet50")
    parser.add_argument("--pretrained", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--base-channels", type=int, default=32, help="Used only when model-name=unet")
    parser.add_argument("--optimizer", type=str, choices=["sgd", "adamw"], default="adamw")
    parser.add_argument("--lr-step-period", type=int, default=15, help="<=0 disables step scheduler")
    parser.add_argument("--lr-gamma", type=float, default=0.1)
    parser.add_argument("--normalize", type=str, choices=["auto", "none", "imagenet"], default="auto")
    parser.add_argument("--augment", action=argparse.BooleanOptionalAction, default=True, help="Enable lightweight geometric/intensity augmentation for train split")
    return parser.parse_args()


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main():
    args = parse_args()
    set_seed(args.seed)

    device = args.device
    amp_enabled = bool(args.amp) and str(device).startswith("cuda") and torch.cuda.is_available()
    sharing_strategy = configure_dataloader_runtime()

    train_loader, val_loader, test_loader, normalize_mode = build_loaders(args, device)

    model = build_stage4_segmentation_model(
        model_name=args.model_name,
        pretrained=args.pretrained,
        in_channels=3,
        base_channels=args.base_channels,
    ).to(device)

    optimizer = build_optimizer(args, model)
    scheduler = None
    if int(args.lr_step_period) > 0:
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=int(args.lr_step_period), gamma=float(args.lr_gamma))

    total_params, trainable_params = count_parameters(model)

    print("=" * 96)
    print("STAGE 4 TRAINING (LV SEGMENTATION)")
    print("=" * 96)
    print(f"Device: {device} | AMP: {amp_enabled}")
    print(f"Data dir: {args.data_dir}")
    print(f"Image size: {args.image_size}")
    print(f"Batch size: {args.batch_size} | Epochs: {args.epochs}")
    print(f"Model: {args.model_name} | Pretrained: {args.pretrained} | Normalize: {normalize_mode}")
    print(f"Optimizer: {args.optimizer} | LR: {args.learning_rate} | WD: {args.weight_decay}")
    print(f"Dice weight: {args.dice_weight} | Area loss weight: {args.area_loss_weight} | Eval threshold: {args.eval_threshold}")
    if args.pos_weight is None:
        print(f"BCE pos_weight: auto (clamped to <= {args.pos_weight_max})")
    else:
        print(f"BCE pos_weight: fixed {args.pos_weight}")
    if scheduler is not None:
        print(f"LR scheduler: StepLR(step={args.lr_step_period}, gamma={args.lr_gamma})")
    else:
        print("LR scheduler: disabled")
    print(
        f"Workers: {args.workers} | pin_memory: {args.pin_memory} | persistent_workers: {args.persistent_workers} | "
        f"prefetch_factor: {args.prefetch_factor if args.workers > 0 else 'n/a'} | sharing_strategy: {sharing_strategy or 'default'} | "
        f"Max videos: {args.max_videos if args.max_videos else 'All'}"
    )
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Output dir: {args.output_dir}")
    print(f"Train samples: {len(train_loader.dataset)} | Val samples: {len(val_loader.dataset)} | Test samples: {len(test_loader.dataset)}")
    print(f"Parameters: total={total_params:,} | trainable={trainable_params:,}")
    print("=" * 96)

    best_val_mae = float("inf")
    epochs_without_improvement = 0

    for epoch in range(1, args.epochs + 1):
        t0 = time.perf_counter()

        train_metrics = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            device=device,
            amp_enabled=amp_enabled,
            dice_weight=args.dice_weight,
            eval_threshold=float(args.eval_threshold),
            manual_pos_weight=args.pos_weight,
            pos_weight_max=float(args.pos_weight_max),
            area_loss_weight=float(args.area_loss_weight),
        )

        val_csv = os.path.join(args.output_dir, f"val_epoch{epoch:03d}_frame_areas.csv")
        val_metrics = evaluate(
            model=model,
            loader=val_loader,
            device=device,
            amp_enabled=amp_enabled,
            dice_weight=args.dice_weight,
            eval_threshold=float(args.eval_threshold),
            manual_pos_weight=args.pos_weight,
            pos_weight_max=float(args.pos_weight_max),
            area_loss_weight=float(args.area_loss_weight),
            csv_path=val_csv,
        )

        if scheduler is not None:
            scheduler.step()

        improved = val_metrics["frame_area_mae"] < best_val_mae
        if improved:
            best_val_mae = val_metrics["frame_area_mae"]
            epochs_without_improvement = 0
            checkpoint_payload = {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "epoch": epoch,
                "val_frame_area_mae": best_val_mae,
                "eval_threshold": float(args.eval_threshold),
                "args": vars(args),
            }
            save_variant = _save_checkpoint_resilient(
                checkpoint_path=args.checkpoint,
                checkpoint_payload=checkpoint_payload,
            )
            save_msg = f"saved ({save_variant})"
        else:
            epochs_without_improvement += 1
            save_msg = f"no-improve {epochs_without_improvement}/{args.patience}"

        dt = time.perf_counter() - t0
        print(
            f"Epoch {epoch:03d}/{args.epochs} | "
            f"Train loss {train_metrics['loss']:.4f} (bce {train_metrics['bce']:.4f}, diceL {train_metrics['dice_loss']:.4f}, areaL {train_metrics['area_loss']:.4f}) "
            f"dice@thr {train_metrics['dice_hard']:.4f} softDice {train_metrics['dice_soft']:.4f} "
            f"fg(pred/gt) {train_metrics['pred_fg_frac']:.4f}/{train_metrics['gt_fg_frac']:.4f} posW {train_metrics['avg_pos_weight']:.2f} | "
            f"Val loss {val_metrics['loss']:.4f} (bce {val_metrics['bce']:.4f}, diceL {val_metrics['dice_loss']:.4f}, areaL {val_metrics['area_loss']:.4f}) "
            f"dice@thr {val_metrics['dice_hard']:.4f} softDice {val_metrics['dice_soft']:.4f} "
            f"fg(pred/gt) {val_metrics['pred_fg_frac']:.4f}/{val_metrics['gt_fg_frac']:.4f} posW {val_metrics['avg_pos_weight']:.2f} | "
            f"Val frame area MAE {val_metrics['frame_area_mae']:.2f} px | "
            f"Val frame area MAPE {val_metrics['frame_area_mape']*100:.2f}% | {save_msg} | {dt:.1f}s"
        )

        if val_metrics["pred_fg_frac"] < 0.005:
            print("Warning: validation predicted foreground fraction is very low (<0.5%). Consider lowering --eval-threshold or increasing --pos-weight.")
        elif val_metrics["pred_fg_frac"] > 0.50:
            print("Warning: validation predicted foreground fraction is very high (>50%). Consider raising --eval-threshold or reducing --pos-weight.")

        if epochs_without_improvement >= args.patience:
            print("Early stopping triggered")
            break

    if not os.path.exists(args.checkpoint):
        raise RuntimeError("No checkpoint saved. Training did not produce a valid model.")

    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])

    val_best_csv = os.path.join(args.output_dir, "val_best_frame_areas.csv")
    val_best_metrics = evaluate(
        model=model,
        loader=val_loader,
        device=device,
        amp_enabled=amp_enabled,
        dice_weight=args.dice_weight,
        eval_threshold=float(args.eval_threshold),
        manual_pos_weight=args.pos_weight,
        pos_weight_max=float(args.pos_weight_max),
        area_loss_weight=float(args.area_loss_weight),
        csv_path=val_best_csv,
    )

    test_csv = os.path.join(args.output_dir, "test_frame_areas.csv")
    test_metrics = evaluate(
        model=model,
        loader=test_loader,
        device=device,
        amp_enabled=amp_enabled,
        dice_weight=args.dice_weight,
        eval_threshold=float(args.eval_threshold),
        manual_pos_weight=args.pos_weight,
        pos_weight_max=float(args.pos_weight_max),
        area_loss_weight=float(args.area_loss_weight),
        csv_path=test_csv,
    )

    print("=" * 96)
    print("STAGE 4 FINAL EVALUATION")
    print("=" * 96)
    print(f"Best checkpoint epoch: {ckpt.get('epoch')}")
    print(f"Best val frame area MAE (during training): {ckpt.get('val_frame_area_mae'):.2f} px")
    print(
        f"Val(best) -> loss: {val_best_metrics['loss']:.4f} | bce: {val_best_metrics['bce']:.4f} | "
        f"dice@thr: {val_best_metrics['dice_hard']:.4f} | softDice: {val_best_metrics['dice_soft']:.4f} | "
        f"area MAE: {val_best_metrics['frame_area_mae']:.2f} px | "
        f"area MAPE: {val_best_metrics['frame_area_mape']*100:.2f}%"
    )
    print(
        f"Test -> loss: {test_metrics['loss']:.4f} | bce: {test_metrics['bce']:.4f} | "
        f"dice@thr: {test_metrics['dice_hard']:.4f} | softDice: {test_metrics['dice_soft']:.4f} | "
        f"area MAE: {test_metrics['frame_area_mae']:.2f} px | "
        f"area MAPE: {test_metrics['frame_area_mape']*100:.2f}%"
    )
    print(f"Test samples: {test_metrics['samples']}")
    print(f"Val frame-level CSV: {os.path.abspath(val_best_csv)}")
    print(f"Val video-level CSV: {os.path.abspath(val_best_csv.replace('.csv', '_video_summary.csv'))}")
    print(f"Test frame-level CSV: {os.path.abspath(test_csv)}")
    print(f"Test video-level CSV: {os.path.abspath(test_csv.replace('.csv', '_video_summary.csv'))}")
    print("=" * 96)


if __name__ == "__main__":
    main()

