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
from pipeline.stage45_pipeline import Stage45Pipeline


def read_video_frame(video_path, frame_idx):
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_idx))
    ok, frame = cap.read()
    cap.release()
    if not ok or frame is None:
        raise ValueError(f"Could not read frame {frame_idx} from {video_path}")
    return frame


def main():
    parser = argparse.ArgumentParser(description="Run Stage 4/5 (LV segmentation + EF) from tracings.")
    parser.add_argument("--split", type=str, default="VAL", help="Dataset split: TRAIN/VAL/TEST")
    parser.add_argument("--max-videos", type=int, default=25, help="Number of videos to process")
    parser.add_argument("--save-overlays", action="store_true", help="Save ED/ES mask overlays")
    parser.add_argument(
        "--output-dir",
        type=str,
        default=os.path.join("visualization", "outputs", "stage45"),
        help="Directory for outputs"
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    filelist_path = os.path.join(config.DATA_DIR, "FileList.csv")
    tracing_path = os.path.join(config.DATA_DIR, "VolumeTracings.csv")

    filelist = pd.read_csv(filelist_path)
    tracings = pd.read_csv(tracing_path)

    split_df = filelist[filelist["Split"] == args.split].copy()
    if args.max_videos > 0:
        split_df = split_df.head(args.max_videos)

    errors = []
    processed = 0

    for _, row in split_df.iterrows():
        fname = row["FileName"]
        fname_ext = f"{fname}.avi"
        height = int(row["FrameHeight"])
        width = int(row["FrameWidth"])

        video_rows = tracings[tracings["FileName"] == fname_ext]
        frame_ids = sorted(video_rows["Frame"].unique().tolist())
        if len(frame_ids) == 0:
            continue

        frame_areas = []
        frame_masks = {}
        for frame_id in frame_ids:
            fr = video_rows[video_rows["Frame"] == frame_id]
            mask = Stage45Pipeline.tracing_to_mask(fr, height=height, width=width)
            area = Stage45Pipeline.mask_area(mask)
            frame_masks[int(frame_id)] = mask
            frame_areas.append((int(frame_id), area))

        # ED: max cavity area, ES: min cavity area
        ed_frame, ed_area = max(frame_areas, key=lambda x: x[1])
        es_frame, es_area = min(frame_areas, key=lambda x: x[1])

        ef_proxy = Stage45Pipeline.compute_ef_from_areas(ed_area, es_area)
        ef_gt = float(row["EF"]) / 100.0
        abs_err = abs(ef_proxy - ef_gt)
        errors.append(abs_err)
        processed += 1

        if args.save_overlays:
            video_path = os.path.join(config.DATA_DIR, "Videos", fname_ext)
            try:
                ed_img = read_video_frame(video_path, ed_frame)
                es_img = read_video_frame(video_path, es_frame)

                # Keep visual output size consistent with video dimensions in filelist
                ed_img = cv2.resize(ed_img, (width, height))
                es_img = cv2.resize(es_img, (width, height))

                ed_overlay = Stage45Pipeline.overlay_mask(ed_img, frame_masks[ed_frame], color=(0, 255, 0))
                es_overlay = Stage45Pipeline.overlay_mask(es_img, frame_masks[es_frame], color=(0, 0, 255))

                cv2.putText(
                    ed_overlay,
                    f"ED frame={ed_frame} area={ed_area:.1f}",
                    (5, 16),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    (255, 255, 255),
                    1,
                    cv2.LINE_AA,
                )
                cv2.putText(
                    es_overlay,
                    f"ES frame={es_frame} area={es_area:.1f}",
                    (5, 16),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    (255, 255, 255),
                    1,
                    cv2.LINE_AA,
                )

                cv2.imwrite(os.path.join(args.output_dir, f"{fname}_ed_overlay.png"), ed_overlay)
                cv2.imwrite(os.path.join(args.output_dir, f"{fname}_es_overlay.png"), es_overlay)
            except Exception as exc:
                print(f"Warning: overlay generation failed for {fname_ext}: {exc}")

    if processed == 0:
        print("No videos were processed.")
        return

    mae = float(np.mean(errors))

    print("=" * 72)
    print("STAGE 4/5 SUMMARY (TRACING-BASED)")
    print("=" * 72)
    print(f"Split:                 {args.split}")
    print(f"Videos processed:      {processed}")
    print(f"EF proxy MAE (0-1):    {mae:.4f}")
    print(f"EF proxy MAE (%):      {mae * 100:.2f}")
    print(f"Output dir:            {os.path.abspath(args.output_dir)}")
    print("=" * 72)


if __name__ == "__main__":
    main()
