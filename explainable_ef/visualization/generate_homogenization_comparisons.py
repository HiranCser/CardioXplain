"""Generate frame comparison images for harmonization and enhancement."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import config
from data.frame_homogenization import apply_frame_preprocessing_steps, load_homogenization_stats


VIDEO_EXTENSIONS = {".avi", ".mp4", ".mov", ".mkv"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Save Original | Harmonized | Enhanced frame comparison images."
    )
    parser.add_argument("--data-dir", type=str, default=config.DATA_DIR)
    parser.add_argument(
        "--homogenization-stats",
        type=str,
        default=os.path.join("validation", "outputs", "homogenization", "frame_homogenization.json"),
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=os.path.join("validation", "outputs", "homogenization", "frame_comparisons"),
    )
    parser.add_argument("--max-videos", type=int, default=0, help="0 means all videos")
    parser.add_argument("--max-frames-per-video", type=int, default=0, help="0 means all frames")
    parser.add_argument("--sample-every", type=int, default=1, help="Save every Nth frame")
    parser.add_argument("--frame-width", type=int, default=512, help="Width of each comparison panel")
    parser.add_argument("--separator-width", type=int, default=36)
    parser.add_argument("--label-height", type=int, default=48)
    return parser.parse_args()


def find_videos(data_dir: Path) -> list[Path]:
    videos_dir = data_dir / "Videos"
    search_root = videos_dir if videos_dir.exists() else data_dir
    return sorted(path for path in search_root.rglob("*") if path.suffix.lower() in VIDEO_EXTENSIONS)


def resize_to_width(image: np.ndarray, width: int) -> np.ndarray:
    height, current_width = image.shape[:2]
    if current_width == width:
        return image
    scale = width / float(current_width)
    new_size = (width, max(1, int(round(height * scale))))
    return cv2.resize(image, new_size, interpolation=cv2.INTER_AREA)


def add_label(image: np.ndarray, text: str, label_height: int) -> np.ndarray:
    label = np.full((label_height, image.shape[1], 3), 28, dtype=np.uint8)
    cv2.putText(
        label,
        text,
        (18, min(label_height - 14, 32)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (245, 245, 245),
        2,
        cv2.LINE_AA,
    )
    return np.vstack([label, image])


def make_separator(height: int, width: int) -> np.ndarray:
    separator = np.full((height, width, 3), 245, dtype=np.uint8)
    center_x = width // 2
    cv2.line(separator, (center_x, 0), (center_x, height), (80, 80, 80), 2)
    return separator


def make_comparison(original_bgr: np.ndarray, harmonized_bgr: np.ndarray, enhanced_bgr: np.ndarray, args: argparse.Namespace) -> np.ndarray:
    panels = [
        add_label(resize_to_width(original_bgr, args.frame_width), "Original", args.label_height),
        add_label(resize_to_width(harmonized_bgr, args.frame_width), "Harmonized", args.label_height),
        add_label(resize_to_width(enhanced_bgr, args.frame_width), "Enhanced final", args.label_height),
    ]

    panel_height = max(panel.shape[0] for panel in panels)
    panels = [pad_to_height(panel, panel_height) for panel in panels]
    separator = make_separator(panel_height, args.separator_width)
    return np.hstack([panels[0], separator, panels[1], separator.copy(), panels[2]])


def pad_to_height(image: np.ndarray, target_height: int) -> np.ndarray:
    pad_height = target_height - image.shape[0]
    if pad_height <= 0:
        return image
    padding = np.full((pad_height, image.shape[1], 3), 0, dtype=np.uint8)
    return np.vstack([image, padding])


def safe_name(path: Path) -> str:
    return "".join(char if char.isalnum() or char in "._-" else "_" for char in path.stem)


def process_video(video_path: Path, stats: dict, output_dir: Path, args: argparse.Namespace) -> int:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"SKIP could not open: {video_path}")
        return 0

    video_output_dir = output_dir / safe_name(video_path)
    video_output_dir.mkdir(parents=True, exist_ok=True)

    saved = 0
    frame_index = 0
    while True:
        ok, before_bgr = cap.read()
        if not ok:
            break

        should_save = frame_index % max(1, args.sample_every) == 0
        if should_save:
            original_rgb = cv2.cvtColor(before_bgr, cv2.COLOR_BGR2RGB)
            steps = apply_frame_preprocessing_steps(original_rgb, stats)
            harmonized_bgr = cv2.cvtColor(steps["harmonized"], cv2.COLOR_RGB2BGR)
            enhanced_bgr = cv2.cvtColor(steps["enhanced"], cv2.COLOR_RGB2BGR)
            comparison = make_comparison(before_bgr, harmonized_bgr, enhanced_bgr, args)

            output_path = video_output_dir / f"frame_{frame_index:05d}.png"
            cv2.imwrite(str(output_path), comparison)
            saved += 1

            if args.max_frames_per_video > 0 and saved >= args.max_frames_per_video:
                break

        frame_index += 1

    cap.release()
    print(f"{video_path.name}: saved {saved} comparison image(s)")
    return saved


def main() -> None:
    args = parse_args()
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    stats = load_homogenization_stats(args.homogenization_stats)

    videos = find_videos(data_dir)
    if args.max_videos > 0:
        videos = videos[: args.max_videos]
    if not videos:
        raise RuntimeError(f"No videos found under {data_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)
    total_saved = 0
    for index, video_path in enumerate(videos, start=1):
        print(f"[{index}/{len(videos)}] {video_path}")
        total_saved += process_video(video_path, stats, output_dir, args)

    print("=" * 88)
    print(f"Saved {total_saved} comparison image(s)")
    print(f"Output: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
