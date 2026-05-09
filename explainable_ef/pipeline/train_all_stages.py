import argparse
import os
import shlex
import subprocess
import sys
import time

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import config


def _run_step(step_name, cmd, env=None):
    print("=" * 96)
    print(f"RUNNING: {step_name}")
    print("Command:", " ".join(shlex.quote(str(c)) for c in cmd))
    print("=" * 96)
    t0 = time.perf_counter()
    subprocess.run(cmd, cwd=ROOT_DIR, env=env, check=True)
    dt = time.perf_counter() - t0
    print(f"Completed {step_name} in {dt:.1f}s")


def parse_args():
    parser = argparse.ArgumentParser(description="Train the EF/ED-ES/segmentation/similarity pipeline in one orchestrated run.")

    parser.add_argument("--data-dir", type=str, default=None, help="Override data dir passed to stage scripts")

    parser.add_argument("--skip-homogenization", action="store_true")
    parser.add_argument("--use-homogenization", action=argparse.BooleanOptionalAction, default=True, help="Apply Stage0 homogenization stats to downstream stages")
    parser.add_argument("--skip-stage1", "--skip-stage123", dest="skip_stage123", action="store_true", help="Skip Stage1 EF model training")
    parser.add_argument("--skip-stage4", action="store_true")
    parser.add_argument("--skip-stage5", action="store_true")
    parser.add_argument("--skip-stage67", action="store_true")

    parser.add_argument("--stage1-checkpoint", "--stage123-checkpoint", dest="stage123_checkpoint", type=str, default=getattr(config, "CHECKPOINT_PATH", "best_model_stage1_ef_96f.pth"))
    parser.add_argument("--train-stage2-detector", "--train-phase-detector", dest="train_phase_detector", action="store_true", help="Train Stage2 ED/ES frame detector after Stage1 EF training")
    parser.add_argument("--stage2-checkpoint", "--phase-checkpoint", dest="phase_checkpoint", type=str, default=getattr(config, "PHASE_CHECKPOINT_PATH", "best_stage2_ed_es_detector.pth"))
    parser.add_argument("--phase-init-checkpoint", type=str, default=None)
    parser.add_argument("--phase-epochs", type=int, default=None)
    parser.add_argument("--phase-learning-rate", type=float, default=None)
    parser.add_argument("--phase-batch-size", type=int, default=None)
    parser.add_argument("--phase-num-frames", type=int, default=None)
    parser.add_argument("--phase-train-sampling-mode", type=str, default=None, choices=["global", "echonet", "phase_window"])
    parser.add_argument("--phase-eval-sampling-mode", type=str, default=None, choices=["global", "echonet", "phase_window"])
    parser.add_argument("--phase-workers", type=int, default=None)
    parser.add_argument("--phase-max-videos", type=int, default=None)
    parser.add_argument("--phase-target-accuracy", type=float, default=0.95)
    parser.add_argument("--phase-stop-on-target", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--phase-tolerance", type=int, default=None)
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
    parser.add_argument("--phase-unfreeze-lr-mult", type=float, default=None)
    parser.add_argument("--phase-weight-decay", type=float, default=None)
    parser.add_argument("--phase-max-grad-norm", type=float, default=None)
    parser.add_argument("--homogenization-stats", type=str, default=os.path.join("validation", "outputs", "homogenization", "frame_homogenization.json"))
    parser.add_argument("--homogenization-max-videos", type=int, default=0)
    parser.add_argument("--homogenization-sample-every", type=int, default=10)
    parser.add_argument("--preprocess-preset", type=str, default="balanced", choices=["off", "conservative", "balanced", "aggressive"])
    parser.add_argument(
        "--homogenization-method",
        type=str,
        default=None,
        choices=["luma_unsharp", "luma_percentile_unsharp", "luma_mean_std_clahe"],
        help="Deprecated legacy method selector; prefer --preprocess-preset",
    )
    parser.add_argument("--enable-harmonization", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--enable-enhancement", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--homogenization-contrast-lower-percentile", type=float, default=None)
    parser.add_argument("--homogenization-contrast-upper-percentile", type=float, default=None)
    parser.add_argument("--homogenization-contrast-blend", "--harmonization-blend", dest="homogenization_contrast_blend", type=float, default=None)
    parser.add_argument("--denoise-method", type=str, default=None, choices=["none", "bilateral", "nlm"])
    parser.add_argument("--bilateral-d", type=int, default=None)
    parser.add_argument("--bilateral-sigma-color", type=float, default=None)
    parser.add_argument("--bilateral-sigma-space", type=float, default=None)
    parser.add_argument("--nlm-h", type=float, default=None)
    parser.add_argument("--homogenization-clahe-clip-limit", "--clahe-clip-limit", dest="homogenization_clahe_clip_limit", type=float, default=None)
    parser.add_argument("--homogenization-clahe-tile-grid-size", "--clahe-tile-grid-size", dest="homogenization_clahe_tile_grid_size", type=int, default=None)
    parser.add_argument("--homogenization-unsharp-amount", "--unsharp-amount", dest="homogenization_unsharp_amount", type=float, default=None)
    parser.add_argument("--homogenization-unsharp-radius", "--unsharp-radius", dest="homogenization_unsharp_radius", type=float, default=None)
    parser.add_argument("--homogenization-unsharp-threshold", "--unsharp-threshold", dest="homogenization_unsharp_threshold", type=float, default=None)
    parser.add_argument("--stage1-epochs", "--stage123-epochs", dest="stage123_epochs", type=int, default=None)
    parser.add_argument("--stage1-learning-rate", "--stage123-learning-rate", dest="stage123_learning_rate", type=float, default=None)
    parser.add_argument("--stage1-batch-size", "--stage123-batch-size", dest="stage123_batch_size", type=int, default=None)
    parser.add_argument("--stage1-num-frames", "--stage123-num-frames", dest="stage123_num_frames", type=int, default=None)
    parser.add_argument("--stage1-dataset-period", "--stage123-dataset-period", dest="stage123_dataset_period", type=int, default=None)
    parser.add_argument("--adaptive-dataset-period", action=argparse.BooleanOptionalAction, default=False, help="Use period 2 for long videos and period 1 for shorter videos in sequence stages")
    parser.add_argument("--adaptive-period-frame-threshold", type=int, default=192)
    parser.add_argument("--adaptive-period-long", type=int, default=2)
    parser.add_argument("--stage1-dataset-max-length", "--stage123-dataset-max-length", dest="stage123_dataset_max_length", type=int, default=None)
    parser.add_argument("--stage1-eval-clips", "--stage123-eval-clips", dest="stage123_eval_clips", type=int, default=None)
    parser.add_argument("--stage1-train-pad", "--stage123-train-pad", dest="stage123_train_pad", type=int, default=None)
    parser.add_argument("--stage1-train-noise", "--stage123-train-noise", dest="stage123_train_noise", type=float, default=None)
    parser.add_argument("--stage1-preserve-temporal-stride", "--stage123-stage1-preserve-temporal-stride", dest="stage123_stage1_preserve_temporal_stride", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--stage1-echonet-style-profile", "--stage123-echonet-style-profile", dest="stage123_echonet_style_profile", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--stage1-workers", "--stage123-workers", dest="stage123_workers", type=int, default=None)
    parser.add_argument("--stage1-max-videos", "--stage123-max-videos", dest="stage123_max_videos", type=int, default=None)
    parser.add_argument("--stage1-phase-validation-metrics", action=argparse.BooleanOptionalAction, default=False, help="Report ED/ES validation metrics during Stage1 EF training")

    parser.add_argument("--stage4-checkpoint", type=str, default=getattr(config, "STAGE4_CHECKPOINT_PATH", "best_stage4_segmentation_area.pth"))
    parser.add_argument("--stage4-epochs", type=int, default=50)
    parser.add_argument("--stage4-learning-rate", type=float, default=1e-4)
    parser.add_argument("--stage4-batch-size", type=int, default=20)
    parser.add_argument("--stage4-workers", type=int, default=8)
    parser.add_argument("--stage4-image-size", type=int, default=112)
    parser.add_argument("--stage4-max-videos", type=int, default=None)
    parser.add_argument("--stage4-model-name", type=str, default="deeplabv3_resnet50")
    parser.add_argument("--stage4-pretrained", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--stage4-optimizer", type=str, choices=["sgd", "adamw"], default="adamw")

    parser.add_argument("--stage5-max-videos", type=int, default=0, help="0 means all videos")
    parser.add_argument("--stage5-save-overlays", action="store_true")
    parser.add_argument("--stage5-stage4-checkpoint", type=str, default=None, help="Defaults to --stage4-checkpoint")
    parser.add_argument("--stage5-stage4-model-name", type=str, default="deeplabv3_resnet50")
    parser.add_argument("--stage5-stage4-base-channels", type=int, default=32)
    parser.add_argument("--stage5-eval-threshold", type=float, default=0.5)

    parser.add_argument("--stage67-output-dir", type=str, default=os.path.join("validation", "outputs", "stage67"))
    parser.add_argument("--stage67-stage5-metrics-dir", type=str, default=os.path.join("validation", "outputs", "stage45"))
    parser.add_argument("--stage67-max-videos", type=int, default=None)
    parser.add_argument("--stage67-normal-threshold", type=float, default=50.0)
    parser.add_argument("--stage67-severe-threshold", type=float, default=30.0)
    parser.add_argument("--stage67-backend", type=str, choices=["similarity", "mlp"], default="similarity")
    parser.add_argument("--stage67-mlp-hidden-dim", type=int, default=64)
    parser.add_argument("--stage67-mlp-dropout", type=float, default=0.1)
    parser.add_argument("--stage67-mlp-epochs", type=int, default=80)
    parser.add_argument("--stage67-mlp-batch-size", type=int, default=128)
    parser.add_argument("--stage67-mlp-learning-rate", type=float, default=1e-3)
    parser.add_argument("--stage67-mlp-weight-decay", type=float, default=1e-4)
    parser.add_argument("--stage67-mlp-patience", type=int, default=10)

    parser.add_argument("--device", type=str, default=None, help="Optional device override propagated to stage scripts")
    return parser.parse_args()


def main():
    args = parse_args()
    python_bin = sys.executable
    homogenization_stats = args.homogenization_stats if args.use_homogenization else None

    t0 = time.perf_counter()

    if args.use_homogenization and not args.skip_homogenization:
        cmd = [
            python_bin,
            os.path.join(ROOT_DIR, "pipeline", "train_frame_homogenization.py"),
            "--output",
            str(args.homogenization_stats),
            "--sample-every",
            str(args.homogenization_sample_every),
            "--preprocess-preset",
            str(args.preprocess_preset),
        ]
        optional_stage0_args = [
            ("--method", args.homogenization_method),
            ("--enable-harmonization" if args.enable_harmonization else "--no-enable-harmonization", args.enable_harmonization),
            ("--enable-enhancement" if args.enable_enhancement else "--no-enable-enhancement", args.enable_enhancement),
            ("--contrast-lower-percentile", args.homogenization_contrast_lower_percentile),
            ("--contrast-upper-percentile", args.homogenization_contrast_upper_percentile),
            ("--harmonization-blend", args.homogenization_contrast_blend),
            ("--denoise-method", args.denoise_method),
            ("--bilateral-d", args.bilateral_d),
            ("--bilateral-sigma-color", args.bilateral_sigma_color),
            ("--bilateral-sigma-space", args.bilateral_sigma_space),
            ("--nlm-h", args.nlm_h),
            ("--clahe-clip-limit", args.homogenization_clahe_clip_limit),
            ("--clahe-tile-grid-size", args.homogenization_clahe_tile_grid_size),
            ("--unsharp-amount", args.homogenization_unsharp_amount),
            ("--unsharp-radius", args.homogenization_unsharp_radius),
            ("--unsharp-threshold", args.homogenization_unsharp_threshold),
        ]
        for flag, value in optional_stage0_args:
            if value is None:
                continue
            if isinstance(value, bool):
                cmd.append(flag)
            else:
                cmd += [flag, str(value)]
        if args.data_dir is not None:
            cmd += ["--data-dir", str(args.data_dir)]
        if args.homogenization_max_videos and int(args.homogenization_max_videos) > 0:
            cmd += ["--max-videos", str(args.homogenization_max_videos)]
        _run_step("Stage0 frame homogenization fitting", cmd)
    elif not args.use_homogenization:
        print("Homogenization disabled: downstream stages will run without Stage0 stats")
    else:
        print("Skipping Stage0 frame homogenization fitting")

    # Stage 1: EF regression
    if not args.skip_stage123:
        ignored_stage123_args = {
            "--stage1-eval-clips": args.stage123_eval_clips,
            "--stage1-train-pad": args.stage123_train_pad,
            "--stage1-train-noise": args.stage123_train_noise,
            "--stage1-preserve-temporal-stride": args.stage123_stage1_preserve_temporal_stride,
            "--stage1-echonet-style-profile": args.stage123_echonet_style_profile if args.stage123_echonet_style_profile else None,
        }
        ignored_stage123_args = {k: v for k, v in ignored_stage123_args.items() if v is not None}
        if ignored_stage123_args:
            print(
                "Ignoring Stage1 EF options no longer supported by model_execution.py: "
                + ", ".join(f"{k}={v}" for k, v in ignored_stage123_args.items())
            )

        cmd = [
            python_bin,
            os.path.join(ROOT_DIR, "model_execution.py"),
            "--no-phase-only",
            "--phase-loss-weight",
            "0.0",
            "--checkpoint",
            str(args.stage123_checkpoint),
        ]
        cmd += ["--phase-validation-metrics" if args.stage1_phase_validation_metrics else "--no-phase-validation-metrics"]
        if args.stage123_epochs is not None:
            cmd += ["--epochs", str(args.stage123_epochs)]
        if args.stage123_learning_rate is not None:
            cmd += ["--learning-rate", str(args.stage123_learning_rate)]
        if args.stage123_batch_size is not None:
            cmd += ["--batch-size", str(args.stage123_batch_size)]
        if args.stage123_num_frames is not None:
            cmd += ["--num-frames", str(args.stage123_num_frames)]
        if args.stage123_dataset_period is not None:
            cmd += ["--dataset-period", str(args.stage123_dataset_period)]
        if args.stage123_dataset_max_length is not None:
            cmd += ["--dataset-max-length", str(args.stage123_dataset_max_length)]
        if args.adaptive_dataset_period:
            cmd += [
                "--adaptive-dataset-period",
                "--adaptive-period-frame-threshold",
                str(args.adaptive_period_frame_threshold),
                "--adaptive-period-long",
                str(args.adaptive_period_long),
            ]
        if homogenization_stats is not None and os.path.exists(homogenization_stats):
            cmd += ["--homogenization-stats", str(homogenization_stats)]
        if args.stage123_workers is not None:
            cmd += ["--workers", str(args.stage123_workers)]
        if args.stage123_max_videos is not None:
            cmd += ["--max-videos", str(args.stage123_max_videos)]
        if args.device is not None and str(args.device).lower() == "cpu":
            cmd += ["--no-amp"]

        _run_step("Stage1 EF training", cmd)
    else:
        print("Skipping Stage1 EF training")

    # Stage 2: ED/ES frame detection
    if args.train_phase_detector:
        cmd = [
            python_bin,
            os.path.join(ROOT_DIR, "pipeline", "train_phase_detector.py"),
            "--checkpoint",
            str(args.phase_checkpoint),
            "--phase-target-accuracy",
            str(args.phase_target_accuracy),
        ]
        cmd += ["--stop-on-phase-target" if args.phase_stop_on_target else "--no-stop-on-phase-target"]

        phase_epochs = args.phase_epochs if args.phase_epochs is not None else args.stage123_epochs
        phase_lr = args.phase_learning_rate if args.phase_learning_rate is not None else args.stage123_learning_rate
        phase_batch = args.phase_batch_size if args.phase_batch_size is not None else args.stage123_batch_size
        phase_frames = args.phase_num_frames if args.phase_num_frames is not None else args.stage123_num_frames
        phase_workers = args.phase_workers if args.phase_workers is not None else args.stage123_workers
        phase_max_videos = args.phase_max_videos if args.phase_max_videos is not None else args.stage123_max_videos

        if phase_epochs is not None:
            cmd += ["--epochs", str(phase_epochs)]
        phase_init_checkpoint = args.phase_init_checkpoint if args.phase_init_checkpoint is not None else args.stage123_checkpoint
        if phase_init_checkpoint is not None:
            if not os.path.exists(phase_init_checkpoint):
                raise FileNotFoundError(
                    f"Stage2 init checkpoint not found: {phase_init_checkpoint}. "
                    "Pass an existing --phase-init-checkpoint path, or remove it to train from scratch."
                )
            cmd += ["--init-checkpoint", str(phase_init_checkpoint)]
        if phase_lr is not None:
            cmd += ["--learning-rate", str(phase_lr)]
        if phase_batch is not None:
            cmd += ["--batch-size", str(phase_batch)]
        if phase_frames is not None:
            cmd += ["--num-frames", str(phase_frames)]
        if args.stage123_dataset_period is not None:
            cmd += ["--dataset-period", str(args.stage123_dataset_period)]
        if args.stage123_dataset_max_length is not None:
            cmd += ["--dataset-max-length", str(args.stage123_dataset_max_length)]
        if args.adaptive_dataset_period:
            cmd += [
                "--adaptive-dataset-period",
                "--adaptive-period-frame-threshold",
                str(args.adaptive_period_frame_threshold),
                "--adaptive-period-long",
                str(args.adaptive_period_long),
            ]
        if args.phase_train_sampling_mode is not None:
            cmd += ["--train-sampling-mode", str(args.phase_train_sampling_mode)]
        if args.phase_eval_sampling_mode is not None:
            cmd += ["--eval-sampling-mode", str(args.phase_eval_sampling_mode)]
        if phase_workers is not None:
            cmd += ["--workers", str(phase_workers)]
        if phase_max_videos is not None:
            cmd += ["--max-videos", str(phase_max_videos)]
        if args.phase_tolerance is not None:
            cmd += ["--tolerance", str(args.phase_tolerance)]
        if args.phase_backbone_freeze_epochs is not None:
            cmd += ["--phase-backbone-freeze-epochs", str(args.phase_backbone_freeze_epochs)]
        if args.backbone_lr_mult is not None:
            cmd += ["--backbone-lr-mult", str(args.backbone_lr_mult)]
        if args.phase_soft_sigma is not None:
            cmd += ["--phase-soft-sigma", str(args.phase_soft_sigma)]
        if args.phase_soft_radius is not None:
            cmd += ["--phase-soft-radius", str(args.phase_soft_radius)]
        if args.phase_hard_index_weight is not None:
            cmd += ["--phase-hard-index-weight", str(args.phase_hard_index_weight)]
        if args.phase_frame_ce_weight is not None:
            cmd += ["--phase-frame-ce-weight", str(args.phase_frame_ce_weight)]
        if args.phase_frame_radius is not None:
            cmd += ["--phase-frame-radius", str(args.phase_frame_radius)]
        if args.phase_cyclic_order is not None:
            cmd.append("--phase-cyclic-order" if args.phase_cyclic_order else "--no-phase-cyclic-order")
        if args.phase_max_gap_ratio is not None:
            cmd += ["--phase-max-gap-ratio", str(args.phase_max_gap_ratio)]
        if args.phase_pair_loss_weight is not None:
            cmd += ["--phase-pair-loss-weight", str(args.phase_pair_loss_weight)]
        if args.phase_unfreeze_lr_mult is not None:
            cmd += ["--phase-unfreeze-lr-mult", str(args.phase_unfreeze_lr_mult)]
        if args.phase_weight_decay is not None:
            cmd += ["--weight-decay", str(args.phase_weight_decay)]
        if args.phase_max_grad_norm is not None:
            cmd += ["--max-grad-norm", str(args.phase_max_grad_norm)]
        if homogenization_stats is not None and os.path.exists(homogenization_stats):
            cmd += ["--homogenization-stats", str(homogenization_stats)]
        if args.device is not None and str(args.device).lower() == "cpu":
            cmd += ["--no-amp"]

        _run_step("Stage2 ED/ES frame detector training", cmd)
    else:
        print("Skipping Stage2 ED/ES frame detector training")

    # Stage 4
    if not args.skip_stage4:
        cmd = [
            python_bin,
            os.path.join(ROOT_DIR, "pipeline", "train_stage4_segmentation.py"),
            "--checkpoint",
            str(args.stage4_checkpoint),
            "--epochs",
            str(args.stage4_epochs),
            "--learning-rate",
            str(args.stage4_learning_rate),
            "--batch-size",
            str(args.stage4_batch_size),
            "--workers",
            str(args.stage4_workers),
            "--image-size",
            str(args.stage4_image_size),
            "--model-name",
            str(args.stage4_model_name),
            "--optimizer",
            str(args.stage4_optimizer),
        ]

        if args.data_dir is not None:
            cmd += ["--data-dir", str(args.data_dir)]
        if homogenization_stats is not None and os.path.exists(homogenization_stats):
            cmd += ["--homogenization-stats", str(homogenization_stats)]
        if args.stage4_max_videos is not None:
            cmd += ["--max-videos", str(args.stage4_max_videos)]
        if args.device is not None:
            cmd += ["--device", str(args.device)]
        if args.stage4_pretrained:
            cmd += ["--pretrained"]
        else:
            cmd += ["--no-pretrained"]

        _run_step("Stage4 training", cmd)
    else:
        print("Skipping Stage4 training")

    # Stage 5
    if not args.skip_stage5:
        stage5_ckpt = args.stage5_stage4_checkpoint if args.stage5_stage4_checkpoint else args.stage4_checkpoint

        for split in ("TRAIN", "VAL", "TEST"):
            cmd = [
                python_bin,
                os.path.join(ROOT_DIR, "pipeline", "run_stage45_from_tracings.py"),
                "--split",
                split,
                "--output-dir",
                os.path.join("validation", "outputs", "stage45", split.lower()),
            ]

            if args.data_dir is not None:
                cmd += ["--data-dir", str(args.data_dir)]
            if args.stage5_max_videos and int(args.stage5_max_videos) > 0:
                cmd += ["--max-videos", str(args.stage5_max_videos)]
            if args.stage5_save_overlays:
                cmd += ["--save-overlays"]

            cmd += [
                "--stage4-checkpoint",
                str(stage5_ckpt),
                "--stage4-model-name",
                str(args.stage5_stage4_model_name),
                "--stage4-base-channels",
                str(args.stage5_stage4_base_channels),
                "--eval-threshold",
                str(args.stage5_eval_threshold),
            ]
            if homogenization_stats is not None and os.path.exists(homogenization_stats):
                cmd += ["--homogenization-stats", str(homogenization_stats)]
            if args.device is not None:
                cmd += ["--device", str(args.device)]

            _run_step(f"Stage5 evaluation ({split})", cmd)
    else:
        print("Skipping Stage5 evaluation")

    # Stage 6/7
    if not args.skip_stage67:
        cmd = [
            python_bin,
            os.path.join(ROOT_DIR, "pipeline", "train_stage67_similarity.py"),
            "--stage123-checkpoint",
            str(args.stage123_checkpoint),
            "--output-dir",
            str(args.stage67_output_dir),
            "--stage5-metrics-dir",
            str(args.stage67_stage5_metrics_dir),
            "--normal-threshold",
            str(args.stage67_normal_threshold),
            "--severe-threshold",
            str(args.stage67_severe_threshold),
            "--stage6-backend",
            str(args.stage67_backend),
        ]

        if args.data_dir is not None:
            cmd += ["--data-dir", str(args.data_dir)]
        if args.stage123_num_frames is not None:
            cmd += ["--num-frames", str(args.stage123_num_frames)]
        if homogenization_stats is not None and os.path.exists(homogenization_stats):
            cmd += ["--homogenization-stats", str(homogenization_stats)]
        if args.stage123_dataset_period is not None:
            cmd += ["--dataset-period", str(args.stage123_dataset_period)]
        if args.adaptive_dataset_period:
            cmd += [
                "--adaptive-dataset-period",
                "--adaptive-period-frame-threshold",
                str(args.adaptive_period_frame_threshold),
                "--adaptive-period-long",
                str(args.adaptive_period_long),
            ]
        if args.stage123_dataset_max_length is not None:
            cmd += ["--dataset-max-length", str(args.stage123_dataset_max_length)]
        if args.stage67_max_videos is not None:
            cmd += ["--max-videos", str(args.stage67_max_videos)]
        if args.device is not None:
            cmd += ["--device", str(args.device)]

        if str(args.stage67_backend) == "mlp":
            cmd += [
                "--stage6-mlp-hidden-dim",
                str(args.stage67_mlp_hidden_dim),
                "--stage6-mlp-dropout",
                str(args.stage67_mlp_dropout),
                "--stage6-mlp-epochs",
                str(args.stage67_mlp_epochs),
                "--stage6-mlp-batch-size",
                str(args.stage67_mlp_batch_size),
                "--stage6-mlp-learning-rate",
                str(args.stage67_mlp_learning_rate),
                "--stage6-mlp-weight-decay",
                str(args.stage67_mlp_weight_decay),
                "--stage6-mlp-patience",
                str(args.stage67_mlp_patience),
            ]

        _run_step("Stage6-7 training", cmd)
    else:
        print("Skipping Stage6-7 training")

    dt = time.perf_counter() - t0
    print("=" * 96)
    print("ALL-STAGE ORCHESTRATION COMPLETED")
    print(f"Total duration: {dt:.1f}s")
    print("=" * 96)


if __name__ == "__main__":
    main()
