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
    parser = argparse.ArgumentParser(description="Train Stage1-7 pipeline in one orchestrated run.")

    parser.add_argument("--data-dir", type=str, default=None, help="Override data dir passed to stage scripts")
    parser.add_argument("--clip-period", type=int, default=getattr(config, "CLIP_PERIOD", 1), help="EchoNet-style clip sampling stride")
    parser.add_argument("--clip-eval-mode", type=str, choices=["center", "all"], default=getattr(config, "CLIP_EVAL_MODE", "center"))

    parser.add_argument("--skip-stage123", action="store_true")
    parser.add_argument("--skip-stage4", action="store_true")
    parser.add_argument("--skip-stage5", action="store_true")
    parser.add_argument("--skip-stage67", action="store_true")

    parser.add_argument("--stage123-checkpoint", type=str, default=getattr(config, "CHECKPOINT_PATH", "best_model_stage123_96f.pth"))
    parser.add_argument("--stage123-epochs", type=int, default=None)
    parser.add_argument("--stage123-learning-rate", type=float, default=None)
    parser.add_argument("--stage123-batch-size", type=int, default=None)
    parser.add_argument("--stage123-num-frames", type=int, default=None)
    parser.add_argument("--stage123-workers", type=int, default=None)
    parser.add_argument("--stage123-max-videos", type=int, default=None)
    parser.add_argument("--stage123-train-clips-per-video", type=int, default=None)
    parser.add_argument("--stage123-clip-start-mode", type=str, choices=["random", "center", "prior"], default=None)
    parser.add_argument("--stage123-clip-prior-path", type=str, default=None)
    parser.add_argument("--stage123-clip-prior-jitter-std", type=float, default=None)
    parser.add_argument("--stage123-phase-loss-weight", type=float, default=None)
    parser.add_argument("--stage123-phase-frame-class-weights", type=str, default=None)
    parser.add_argument("--stage123-phase-event-heatmap-weight", type=float, default=None)
    parser.add_argument("--stage123-phase-only-warmup-epochs", type=int, default=None)
    parser.add_argument("--stage123-ef-loss", type=str, choices=["smooth_l1", "l1", "mse"], default=None)
    parser.add_argument("--stage123-ef-smooth-l1-beta", type=float, default=None)
    parser.add_argument("--stage123-monitor", type=str, choices=["joint_score", "ef_mae", "ef_mae_with_phase_gate"], default=None)
    parser.add_argument("--stage123-phase-gate", type=float, default=None)

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
    parser.add_argument(
        "--stage5-mode",
        type=str,
        choices=["tracing", "predicted_masks"],
        default="predicted_masks",
        help="tracing baseline or learned Stage4 mask-based EF evaluation",
    )
    parser.add_argument("--stage5-stage4-checkpoint", type=str, default=None, help="Defaults to --stage4-checkpoint")
    parser.add_argument("--stage5-stage4-model-name", type=str, default="deeplabv3_resnet50")
    parser.add_argument("--stage5-stage4-base-channels", type=int, default=32)
    parser.add_argument("--stage5-eval-threshold", type=float, default=0.5)

    parser.add_argument("--stage67-output-dir", type=str, default=os.path.join("validation", "outputs", "stage67"))
    parser.add_argument("--stage67-max-videos", type=int, default=None)
    parser.add_argument("--stage67-normal-threshold", type=float, default=50.0)
    parser.add_argument("--stage67-severe-threshold", type=float, default=30.0)
    parser.add_argument("--stage67-clip-eval-mode", type=str, choices=["center", "all"], default="all")
    parser.add_argument("--stage67-clip-batch-size", type=int, default=8)
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

    t0 = time.perf_counter()

    # Stage 1-3
    if not args.skip_stage123:
        cmd = [
            python_bin,
            os.path.join(ROOT_DIR, "model_execution.py"),
            "--train-stage123",
            "--no-phase-only",
            "--checkpoint",
            str(args.stage123_checkpoint),
            "--clip-period",
            str(args.clip_period),
            "--clip-eval-mode",
            str(args.clip_eval_mode),
        ]
        if args.stage123_epochs is not None:
            cmd += ["--epochs", str(args.stage123_epochs)]
        if args.data_dir is not None:
            cmd += ["--data-dir", str(args.data_dir)]
        if args.stage123_learning_rate is not None:
            cmd += ["--learning-rate", str(args.stage123_learning_rate)]
        if args.stage123_batch_size is not None:
            cmd += ["--batch-size", str(args.stage123_batch_size)]
        if args.stage123_num_frames is not None:
            cmd += ["--num-frames", str(args.stage123_num_frames)]
        if args.stage123_workers is not None:
            cmd += ["--workers", str(args.stage123_workers)]
        if args.stage123_max_videos is not None:
            cmd += ["--max-videos", str(args.stage123_max_videos)]
        if args.stage123_train_clips_per_video is not None:
            cmd += ["--train-clips-per-video", str(args.stage123_train_clips_per_video)]
        if args.stage123_clip_start_mode is not None:
            cmd += ["--clip-start-mode", str(args.stage123_clip_start_mode)]
        if args.stage123_clip_prior_path is not None:
            cmd += ["--clip-prior-path", str(args.stage123_clip_prior_path)]
        if args.stage123_clip_prior_jitter_std is not None:
            cmd += ["--clip-prior-jitter-std", str(args.stage123_clip_prior_jitter_std)]
        if args.stage123_phase_loss_weight is not None:
            cmd += ["--phase-loss-weight", str(args.stage123_phase_loss_weight)]
        if args.stage123_phase_frame_class_weights is not None:
            cmd += ["--phase-frame-class-weights", str(args.stage123_phase_frame_class_weights)]
        if args.stage123_phase_event_heatmap_weight is not None:
            cmd += ["--phase-event-heatmap-weight", str(args.stage123_phase_event_heatmap_weight)]
        if args.stage123_phase_only_warmup_epochs is not None:
            cmd += ["--phase-only-warmup-epochs", str(args.stage123_phase_only_warmup_epochs)]
        if args.stage123_ef_loss is not None:
            cmd += ["--ef-loss", str(args.stage123_ef_loss)]
        if args.stage123_ef_smooth_l1_beta is not None:
            cmd += ["--ef-smooth-l1-beta", str(args.stage123_ef_smooth_l1_beta)]
        if args.stage123_monitor is not None:
            cmd += ["--stage123-monitor", str(args.stage123_monitor)]
        if args.stage123_phase_gate is not None:
            cmd += ["--stage123-phase-gate", str(args.stage123_phase_gate)]
        if args.device is not None and str(args.device).lower() == "cpu":
            cmd += ["--no-amp"]

        _run_step("Stage1-3 training", cmd)
    else:
        print("Skipping Stage1-3 training")

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

        for split in ("VAL", "TEST"):
            cmd = [
                python_bin,
                os.path.join(ROOT_DIR, "pipeline", "run_stage45_from_tracings.py"),
                "--split",
                split,
                "--mode",
                str(args.stage5_mode),
                "--output-dir",
                os.path.join("validation", "outputs", "stage45", split.lower()),
            ]

            if args.data_dir is not None:
                cmd += ["--data-dir", str(args.data_dir)]
            if args.stage5_max_videos and int(args.stage5_max_videos) > 0:
                cmd += ["--max-videos", str(args.stage5_max_videos)]
            if args.stage5_save_overlays:
                cmd += ["--save-overlays"]

            if str(args.stage5_mode) == "predicted_masks":
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
            "--normal-threshold",
            str(args.stage67_normal_threshold),
            "--severe-threshold",
            str(args.stage67_severe_threshold),
            "--stage6-backend",
            str(args.stage67_backend),
            "--clip-period",
            str(args.clip_period),
            "--clip-eval-mode",
            str(args.stage67_clip_eval_mode),
            "--clip-batch-size",
            str(args.stage67_clip_batch_size),
        ]

        if args.data_dir is not None:
            cmd += ["--data-dir", str(args.data_dir)]
        if args.stage123_num_frames is not None:
            cmd += ["--num-frames", str(args.stage123_num_frames)]
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
