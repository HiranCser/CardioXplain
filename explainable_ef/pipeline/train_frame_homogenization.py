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
from data.frame_homogenization import save_homogenization_stats


def parse_args():
    parser = argparse.ArgumentParser(description="Fit frame homogenization statistics from training videos.")
    parser.add_argument("--data-dir", type=str, default=config.DATA_DIR)
    parser.add_argument("--output", type=str, default=os.path.join("validation", "outputs", "homogenization", "frame_homogenization.json"))
    parser.add_argument("--split", type=str, default="TRAIN", choices=["TRAIN", "VAL", "TEST"])
    parser.add_argument("--max-videos", type=int, default=0, help="0 means all videos")
    parser.add_argument("--sample-every", type=int, default=10, help="Read every Nth frame for fitting")
    parser.add_argument("--image-size", type=int, default=112)
    parser.add_argument("--clahe-clip-limit", type=float, default=2.0)
    parser.add_argument("--clahe-tile-grid-size", type=int, default=8)
    return parser.parse_args()


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
    filelist_path = os.path.join(args.data_dir, "FileList.csv")
    filelist = pd.read_csv(filelist_path)
    split_df = filelist[filelist["Split"].astype(str).str.upper() == str(args.split).upper()].copy()
    if int(args.max_videos) > 0:
        split_df = split_df.head(int(args.max_videos))

    means = []
    stds = []
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
            frame_count += 1

    if frame_count <= 0:
        raise RuntimeError("No frames sampled for homogenization fitting")

    stats = {
        "enabled": True,
        "method": "luma_mean_std_clahe",
        "target_mean": float(np.median(means)),
        "target_std": float(np.median(stds)),
        "clahe_clip_limit": float(args.clahe_clip_limit),
        "clahe_tile_grid_size": int(args.clahe_tile_grid_size),
        "fit_split": str(args.split).upper(),
        "fit_videos": int(len(split_df)),
        "fit_frames": int(frame_count),
        "sample_every": int(max(1, args.sample_every)),
        "image_size": int(args.image_size),
    }
    save_homogenization_stats(args.output, stats)

    print("=" * 88)
    print("FRAME HOMOGENIZATION FIT COMPLETE")
    print("=" * 88)
    print(f"Output:       {os.path.abspath(args.output)}")
    print(f"Videos:       {len(split_df)}")
    print(f"Frames:       {frame_count}")
    print(f"Target mean:  {stats['target_mean']:.4f}")
    print(f"Target std:   {stats['target_std']:.4f}")
    print("=" * 88)


if __name__ == "__main__":
    main()
