import argparse
import os
import sys

import cv2
import numpy as np
import pandas as pd

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import config
from data.frame_homogenization import PREPROCESS_PRESETS, save_homogenization_stats


def parse_args():
    parser = argparse.ArgumentParser(description="Fit frame homogenization statistics from training videos.")
    parser.add_argument("--data-dir", type=str, default=config.DATA_DIR)
    parser.add_argument("--output", type=str, default=os.path.join("validation", "outputs", "homogenization", "frame_homogenization.json"))
    parser.add_argument("--split", type=str, default="TRAIN", choices=["TRAIN", "VAL", "TEST"])
    parser.add_argument("--max-videos", type=int, default=0, help="0 means all videos")
    parser.add_argument("--sample-every", type=int, default=10, help="Read every Nth frame for fitting")
    parser.add_argument("--image-size", type=int, default=112)
    parser.add_argument(
        "--preprocess-preset",
        type=str,
        default="balanced",
        choices=["off", "conservative", "balanced", "aggressive"],
        help="Preset controlling harmonization and enhancement strength",
    )
    parser.add_argument("--enable-harmonization", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--enable-enhancement", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument(
        "--method",
        type=str,
        default=None,
        choices=["luma_unsharp", "luma_percentile_unsharp", "luma_mean_std_clahe"],
        help="Deprecated legacy method selector; prefer --preprocess-preset",
    )
    parser.add_argument("--contrast-lower-percentile", type=float, default=None)
    parser.add_argument("--contrast-upper-percentile", type=float, default=None)
    parser.add_argument("--harmonization-blend", "--contrast-blend", dest="harmonization_blend", type=float, default=None)
    parser.add_argument("--denoise-method", type=str, default=None, choices=["none", "bilateral", "nlm"])
    parser.add_argument("--bilateral-d", type=int, default=None)
    parser.add_argument("--bilateral-sigma-color", type=float, default=None)
    parser.add_argument("--bilateral-sigma-space", type=float, default=None)
    parser.add_argument("--nlm-h", type=float, default=None)
    parser.add_argument("--clahe-clip-limit", type=float, default=None)
    parser.add_argument("--clahe-tile-grid-size", type=int, default=None)
    parser.add_argument("--unsharp-amount", type=float, default=None)
    parser.add_argument("--unsharp-radius", type=float, default=None)
    parser.add_argument("--unsharp-threshold", type=float, default=None)
    return parser.parse_args()


def _resolve_stats_template(args):
    stats = {k: v for k, v in PREPROCESS_PRESETS[str(args.preprocess_preset).lower()].items()}
    import copy

    stats = copy.deepcopy(stats)
    if args.enable_harmonization is not None:
        stats["harmonization"]["enabled"] = bool(args.enable_harmonization)
    if args.enable_enhancement is not None:
        stats["enhancement"]["enabled"] = bool(args.enable_enhancement)

    if args.method == "luma_unsharp":
        stats["harmonization"]["enabled"] = False
        stats["enhancement"]["enabled"] = True
        stats["enhancement"]["denoise_method"] = "none"
        stats["enhancement"]["clahe_clip_limit"] = 0.0
    elif args.method == "luma_percentile_unsharp":
        stats["harmonization"]["enabled"] = True
        stats["enhancement"]["enabled"] = True
        stats["enhancement"]["denoise_method"] = "none"
        stats["enhancement"]["clahe_clip_limit"] = 0.0
    elif args.method == "luma_mean_std_clahe":
        stats["legacy_method"] = args.method

    harmonization = stats["harmonization"]
    enhancement = stats["enhancement"]
    overrides = [
        (harmonization, "lower_percentile", args.contrast_lower_percentile),
        (harmonization, "upper_percentile", args.contrast_upper_percentile),
        (harmonization, "blend", args.harmonization_blend),
        (enhancement, "denoise_method", args.denoise_method),
        (enhancement, "bilateral_d", args.bilateral_d),
        (enhancement, "bilateral_sigma_color", args.bilateral_sigma_color),
        (enhancement, "bilateral_sigma_space", args.bilateral_sigma_space),
        (enhancement, "nlm_h", args.nlm_h),
        (enhancement, "clahe_clip_limit", args.clahe_clip_limit),
        (enhancement, "clahe_tile_grid_size", args.clahe_tile_grid_size),
        (enhancement, "unsharp_amount", args.unsharp_amount),
        (enhancement, "unsharp_radius", args.unsharp_radius),
        (enhancement, "unsharp_threshold", args.unsharp_threshold),
    ]
    for section, key, value in overrides:
        if value is not None:
            section[key] = value
    return stats


def _iter_sampled_luma(video_path, image_size, sample_every):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")

    idx = 0
    sample_every = max(1, int(sample_every))
    while True:
        ok, frame_bgr = cap.read()
        if not ok or frame_bgr is None:
            break
        if idx % sample_every == 0:
            frame_bgr = cv2.resize(frame_bgr, (int(image_size), int(image_size)), interpolation=cv2.INTER_LINEAR)
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            y = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2YCrCb)[:, :, 0].astype(np.float32) / 255.0
            yield y
        idx += 1

    cap.release()


def main():
    args = parse_args()
    stats = _resolve_stats_template(args)
    harmonization = stats["harmonization"]
    filelist_path = os.path.join(args.data_dir, "FileList.csv")
    filelist = pd.read_csv(filelist_path)
    split_df = filelist[filelist["Split"].astype(str).str.upper() == str(args.split).upper()].copy()
    if int(args.max_videos) > 0:
        split_df = split_df.head(int(args.max_videos))

    means = []
    stds = []
    lows = []
    highs = []
    frame_count = 0

    for _, row in split_df.iterrows():
        fname_ext = str(row["FileName"]).strip() + ".avi"
        video_path = os.path.join(args.data_dir, "Videos", fname_ext)
        if not os.path.exists(video_path):
            print(f"Warning: missing video {video_path}")
            continue
        for y in _iter_sampled_luma(video_path, args.image_size, args.sample_every):
            means.append(float(np.mean(y)))
            stds.append(float(np.std(y)))
            lows.append(float(np.percentile(y, harmonization.get("lower_percentile", 1.0))))
            highs.append(float(np.percentile(y, harmonization.get("upper_percentile", 99.0))))
            frame_count += 1

    if frame_count <= 0:
        raise RuntimeError("No frames sampled for homogenization fitting")

    stats["fit_split"] = str(args.split).upper()
    stats["fit_videos"] = int(len(split_df))
    stats["fit_frames"] = int(frame_count)
    stats["sample_every"] = int(max(1, args.sample_every))
    stats["image_size"] = int(args.image_size)
    stats["target_mean"] = float(np.median(means))
    stats["target_std"] = float(np.median(stds))
    stats["harmonization"]["target_low"] = float(np.median(lows))
    stats["harmonization"]["target_high"] = float(np.median(highs))
    save_homogenization_stats(args.output, stats)

    print("=" * 88)
    print("FRAME HOMOGENIZATION FIT COMPLETE")
    print("=" * 88)
    print(f"Output:       {os.path.abspath(args.output)}")
    print(f"Videos:       {len(split_df)}")
    print(f"Frames:       {frame_count}")
    print(f"Preset:       {stats['preset']}")
    print(f"Pipeline:     {stats['pipeline']}")
    print(f"Target mean:  {stats['target_mean']:.4f}")
    print(f"Target std:   {stats['target_std']:.4f}")
    print(f"Target range: {stats['harmonization']['target_low']:.4f} - {stats['harmonization']['target_high']:.4f}")
    print("=" * 88)


if __name__ == "__main__":
    main()
