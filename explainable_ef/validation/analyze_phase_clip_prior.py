import argparse
import json
import os
import sys

import cv2
import numpy as np
import pandas as pd

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from data.phase_ground_truth import compute_ed_es_from_video_rows


def count_video_frames(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return 0
    count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if count > 0:
        cap.release()
        return count

    count = 0
    while True:
        ok, _frame = cap.read()
        if not ok:
            break
        count += 1
    cap.release()
    return count


def percentile_dict(values):
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return {}
    return {
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "p05": float(np.percentile(arr, 5)),
        "p25": float(np.percentile(arr, 25)),
        "p50": float(np.percentile(arr, 50)),
        "p75": float(np.percentile(arr, 75)),
        "p95": float(np.percentile(arr, 95)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
    }


def centered_window_contains(frame, total_frames, clip_span):
    start = max(0, (int(total_frames) - int(clip_span)) // 2)
    end = min(int(total_frames) - 1, start + int(clip_span) - 1)
    return start <= int(frame) <= end


def main():
    parser = argparse.ArgumentParser(description="Analyze TRAIN ED/ES position priors and cardiac-cycle frame coverage.")
    parser.add_argument("--data-dir", type=str, default=None)
    parser.add_argument("--output-json", type=str, default=os.path.join("validation", "outputs", "train_phase_clip_prior.json"))
    parser.add_argument("--output-csv", type=str, default=os.path.join("validation", "outputs", "train_phase_clip_prior_videos.csv"))
    parser.add_argument("--clip-frames", type=str, default="64,80,96,112,128")
    parser.add_argument("--clip-period", type=int, default=1)
    parser.add_argument("--max-videos", type=int, default=None)
    args = parser.parse_args()

    if args.data_dir is None:
        import config
        data_dir = config.DATA_DIR
    else:
        data_dir = args.data_dir

    filelist = pd.read_csv(os.path.join(data_dir, "FileList.csv"))
    tracings = pd.read_csv(os.path.join(data_dir, "VolumeTracings.csv"))
    train_rows = filelist[filelist["Split"].astype(str).str.upper() == "TRAIN"].copy()
    if args.max_videos is not None and args.max_videos > 0:
        train_rows = train_rows.iloc[: args.max_videos]

    rows = []
    for _, row in train_rows.iterrows():
        stem = str(row["FileName"])
        filename = stem + ".avi"
        video_path = os.path.join(data_dir, "Videos", filename)
        total_frames = count_video_frames(video_path)
        video_rows = tracings[tracings["FileName"] == filename]
        phase = compute_ed_es_from_video_rows(video_rows)
        ed = int(phase["ed_frame"])
        es = int(phase["es_frame"])
        if total_frames <= 0 or ed < 0 or es < 0:
            continue

        half_cycle = abs(es - ed)
        cycle_frames = 2 * half_cycle if half_cycle > 0 else np.nan
        rows.append(
            {
                "file_name": stem,
                "total_frames": int(total_frames),
                "ed_frame": ed,
                "es_frame": es,
                "ed_rel": float(ed / max(1, total_frames - 1)),
                "es_rel": float(es / max(1, total_frames - 1)),
                "pair_mid_rel": float(((ed + es) * 0.5) / max(1, total_frames - 1)),
                "half_cycle_frames": float(half_cycle),
                "cycle_frames_est": float(cycle_frames),
                "two_cycle_frames_est": float(2 * cycle_frames) if np.isfinite(cycle_frames) else np.nan,
            }
        )

    if not rows:
        raise RuntimeError("No usable TRAIN videos found for phase prior analysis")

    df = pd.DataFrame(rows)
    clip_frames = [int(x.strip()) for x in str(args.clip_frames).split(",") if x.strip()]
    clip_summaries = {}
    for frames in clip_frames:
        span = (frames - 1) * max(1, int(args.clip_period)) + 1
        center_ed = []
        center_es = []
        center_both = []
        one_cycle = []
        two_cycles = []
        for item in rows:
            center_ed.append(centered_window_contains(item["ed_frame"], item["total_frames"], span))
            center_es.append(centered_window_contains(item["es_frame"], item["total_frames"], span))
            center_both.append(center_ed[-1] and center_es[-1])
            one_cycle.append(span >= item["cycle_frames_est"])
            two_cycles.append(span >= item["two_cycle_frames_est"])
        clip_summaries[str(frames)] = {
            "clip_span_original_frames": int(span),
            "center_clip_contains_ed_pct": float(np.mean(center_ed) * 100.0),
            "center_clip_contains_es_pct": float(np.mean(center_es) * 100.0),
            "center_clip_contains_both_pct": float(np.mean(center_both) * 100.0),
            "clip_covers_one_est_cycle_pct": float(np.mean(one_cycle) * 100.0),
            "clip_covers_two_est_cycles_pct": float(np.mean(two_cycles) * 100.0),
        }

    prior = {
        "source_split": "TRAIN",
        "num_videos": int(len(df)),
        "ed_rel": percentile_dict(df["ed_rel"]),
        "es_rel": percentile_dict(df["es_rel"]),
        "pair_mid_rel": percentile_dict(df["pair_mid_rel"]),
        "half_cycle_frames": percentile_dict(df["half_cycle_frames"]),
        "cycle_frames_est": percentile_dict(df["cycle_frames_est"].dropna()),
        "two_cycle_frames_est": percentile_dict(df["two_cycle_frames_est"].dropna()),
        "clip_summaries": clip_summaries,
        "clip_center_priors": {
            "ed_rel_mean": float(df["ed_rel"].mean()),
            "es_rel_mean": float(df["es_rel"].mean()),
            "pair_mid_rel_mean": float(df["pair_mid_rel"].mean()),
            "ed_rel_std": float(df["ed_rel"].std(ddof=0)),
            "es_rel_std": float(df["es_rel"].std(ddof=0)),
            "pair_mid_rel_std": float(df["pair_mid_rel"].std(ddof=0)),
        },
    }

    os.makedirs(os.path.dirname(args.output_json), exist_ok=True)
    os.makedirs(os.path.dirname(args.output_csv), exist_ok=True)
    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(prior, f, indent=2)
    df.to_csv(args.output_csv, index=False)

    print(f"Wrote prior JSON: {args.output_json}")
    print(f"Wrote per-video CSV: {args.output_csv}")
    print(json.dumps(prior, indent=2))


if __name__ == "__main__":
    main()
