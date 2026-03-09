import argparse
import os
import shlex
import subprocess
import sys
import time

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


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

    parser.add_argument("--skip-stage123", action="store_true")
    parser.add_argument("--skip-stage4", action="store_true")
    parser.add_argument("--skip-stage5", action="store_true")
    parser.add_argument("--skip-stage67", action="store_true")

    parser.add_argument("--stage123-checkpoint", type=str, default="best_model.pth")
    parser.add_argument("--stage123-epochs", type=int, default=None)
    parser.add_argument("--stage123-learning-rate", type=float, default=None)
    parser.add_argument("--stage123-batch-size", type=int, default=None)
    parser.add_argument("--stage123-num-frames", type=int, default=None)
    parser.add_argument("--stage123-workers", type=int, default=None)
    parser.add_argument("--stage123-max-videos", type=int, default=None)

    parser.add_argument("--stage4-checkpoint", type=str, default="best_stage4_segmentation.pth")
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

    parser.add_argument("--stage67-output-dir", type=str, default=os.path.join("validation", "outputs", "stage67"))
    parser.add_argument("--stage67-max-videos", type=int, default=None)
    parser.add_argument("--stage67-normal-threshold", type=float, default=50.0)
    parser.add_argument("--stage67-severe-threshold", type=float, default=30.0)

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
        ]
        if args.stage123_epochs is not None:
            cmd += ["--epochs", str(args.stage123_epochs)]
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

    # Stage 5 (deterministic EF computation from tracings)
    if not args.skip_stage5:
        for split in ("VAL", "TEST"):
            cmd = [
                python_bin,
                os.path.join(ROOT_DIR, "pipeline", "run_stage45_from_tracings.py"),
                "--split",
                split,
                "--output-dir",
                os.path.join("validation", "outputs", "stage45", split.lower()),
            ]
            if args.stage5_max_videos and int(args.stage5_max_videos) > 0:
                cmd += ["--max-videos", str(args.stage5_max_videos)]
            if args.stage5_save_overlays:
                cmd += ["--save-overlays"]

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
        ]

        if args.data_dir is not None:
            cmd += ["--data-dir", str(args.data_dir)]
        if args.stage123_num_frames is not None:
            cmd += ["--num-frames", str(args.stage123_num_frames)]
        if args.stage67_max_videos is not None:
            cmd += ["--max-videos", str(args.stage67_max_videos)]
        if args.device is not None:
            cmd += ["--device", str(args.device)]

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
