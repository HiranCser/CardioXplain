import argparse
import os
import sys

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import config
import model_execution


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Train Stage2 ED/ES frame detector."
    )
    parser.add_argument("--checkpoint", type=str, default=getattr(config, "PHASE_CHECKPOINT_PATH", "best_phase_detector.pth"))
    parser.add_argument("--init-checkpoint", type=str, default=None, help="Warm-start from an existing Stage1 EF checkpoint, for example best_model.pth")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--learning-rate", "--lr", dest="learning_rate", type=float, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-frames", type=int, default=None)
    parser.add_argument("--image-size", type=int, default=None)
    parser.add_argument("--dataset-period", type=int, default=None)
    parser.add_argument("--adaptive-dataset-period", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--adaptive-period-frame-threshold", type=int, default=None)
    parser.add_argument("--adaptive-period-long", type=int, default=None)
    parser.add_argument("--dataset-max-length", type=int, default=None)
    parser.add_argument("--train-sampling-mode", type=str, default=None, choices=["global", "echonet", "phase_window"])
    parser.add_argument("--eval-sampling-mode", type=str, default=None, choices=["global", "echonet", "phase_window"])
    parser.add_argument("--max-videos", type=int, default=None)
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--prefetch-factor", type=int, default=None)
    parser.add_argument("--validate-every", type=int, default=1)
    parser.add_argument("--homogenization-stats", type=str, default=None)
    parser.add_argument("--phase-target-accuracy", type=float, default=0.95)
    parser.add_argument("--stop-on-phase-target", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--tolerance", type=int, default=None)
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--phase-backbone-freeze-epochs", type=int, default=None)
    parser.add_argument("--backbone-lr-mult", type=float, default=None)
    parser.add_argument("--phase-soft-sigma", type=float, default=None)
    parser.add_argument("--phase-soft-radius", type=int, default=None)
    parser.add_argument("--phase-hard-index-weight", type=float, default=None)
    parser.add_argument("--phase-frame-ce-weight", type=float, default=None)
    parser.add_argument("--phase-frame-radius", type=int, default=None)
    parser.add_argument("--phase-cyclic-order", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--phase-max-gap-ratio", type=float, default=None)
    parser.add_argument("--phase-pair-loss-weight", type=float, default=None)
    parser.add_argument("--phase-pair-soft-sigma", type=float, default=None)
    parser.add_argument("--phase-pair-soft-radius", type=int, default=None)
    parser.add_argument("--phase-unfreeze-lr-mult", type=float, default=None)
    parser.add_argument("--weight-decay", type=float, default=None)
    parser.add_argument("--max-grad-norm", type=float, default=None)
    return parser.parse_args(argv)


def _append_optional(argv, flag, value):
    if value is not None:
        argv.extend([flag, str(value)])


def build_model_execution_argv(args):
    argv = [
        "--phase-detector",
        "--checkpoint",
        str(args.checkpoint),
        "--validate-every",
        str(args.validate_every),
        "--phase-target-accuracy",
        str(args.phase_target_accuracy),
    ]

    argv.append("--stop-on-phase-target" if args.stop_on_phase_target else "--no-stop-on-phase-target")

    _append_optional(argv, "--epochs", args.epochs)
    _append_optional(argv, "--init-checkpoint", args.init_checkpoint)
    _append_optional(argv, "--learning-rate", args.learning_rate)
    _append_optional(argv, "--batch-size", args.batch_size)
    _append_optional(argv, "--num-frames", args.num_frames)
    _append_optional(argv, "--image-size", args.image_size)
    _append_optional(argv, "--dataset-period", args.dataset_period)
    if args.adaptive_dataset_period is not None:
        argv.append("--adaptive-dataset-period" if args.adaptive_dataset_period else "--no-adaptive-dataset-period")
    _append_optional(argv, "--adaptive-period-frame-threshold", args.adaptive_period_frame_threshold)
    _append_optional(argv, "--adaptive-period-long", args.adaptive_period_long)
    _append_optional(argv, "--dataset-max-length", args.dataset_max_length)
    _append_optional(argv, "--train-sampling-mode", args.train_sampling_mode)
    _append_optional(argv, "--eval-sampling-mode", args.eval_sampling_mode)
    _append_optional(argv, "--max-videos", args.max_videos)
    _append_optional(argv, "--workers", args.workers)
    _append_optional(argv, "--prefetch-factor", args.prefetch_factor)
    _append_optional(argv, "--homogenization-stats", args.homogenization_stats)
    _append_optional(argv, "--tolerance", args.tolerance)
    _append_optional(argv, "--phase-backbone-freeze-epochs", args.phase_backbone_freeze_epochs)
    _append_optional(argv, "--backbone-lr-mult", args.backbone_lr_mult)
    _append_optional(argv, "--phase-soft-sigma", args.phase_soft_sigma)
    _append_optional(argv, "--phase-soft-radius", args.phase_soft_radius)
    _append_optional(argv, "--phase-hard-index-weight", args.phase_hard_index_weight)
    _append_optional(argv, "--phase-frame-ce-weight", args.phase_frame_ce_weight)
    _append_optional(argv, "--phase-frame-radius", args.phase_frame_radius)
    if args.phase_cyclic_order is not None:
        argv.append("--phase-cyclic-order" if args.phase_cyclic_order else "--no-phase-cyclic-order")
    _append_optional(argv, "--phase-max-gap-ratio", args.phase_max_gap_ratio)
    _append_optional(argv, "--phase-pair-loss-weight", args.phase_pair_loss_weight)
    _append_optional(argv, "--phase-pair-soft-sigma", args.phase_pair_soft_sigma)
    _append_optional(argv, "--phase-pair-soft-radius", args.phase_pair_soft_radius)
    _append_optional(argv, "--phase-unfreeze-lr-mult", args.phase_unfreeze_lr_mult)
    _append_optional(argv, "--weight-decay", args.weight_decay)
    _append_optional(argv, "--max-grad-norm", args.max_grad_norm)

    if args.amp is not None:
        argv.append("--amp" if args.amp else "--no-amp")

    return argv


def main(argv=None):
    args = parse_args(argv)
    return model_execution.main(build_model_execution_argv(args))


if __name__ == "__main__":
    main()
