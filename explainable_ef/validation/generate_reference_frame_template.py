import argparse
import os
import sys

import pandas as pd

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import config
from data.phase_ground_truth import compute_ed_es_from_video_rows


def _format_frame_value(value):
    if pd.isna(value):
        return ""
    return int(value)


def _build_tracing_prefill_table(data_dir, method="global_extrema", smooth_window=5, enforce_es_after_ed=True):
    tracings_path = os.path.join(data_dir, "VolumeTracings.csv")
    if not os.path.exists(tracings_path):
        raise FileNotFoundError(f"VolumeTracings.csv not found: {tracings_path}")

    tracings = pd.read_csv(tracings_path)

    rows = []
    for file_name, group in tracings.groupby("FileName"):
        phase = compute_ed_es_from_video_rows(
            group,
            method=method,
            smooth_window=smooth_window,
            enforce_es_after_ed=enforce_es_after_ed,
        )
        rows.append(
            {
                "FileNameWithExt": str(file_name),
                "ED_prefill": phase["ed_frame"],
                "ES_prefill": phase["es_frame"],
            }
        )

    return pd.DataFrame(rows)


def build_template(
    data_dir,
    split=None,
    max_videos=None,
    include_meta=True,
    prefill_from_tracings=False,
    prefill_method="global_extrema",
    prefill_smooth_window=5,
    enforce_es_after_ed=True,
):
    filelist_path = os.path.join(data_dir, "FileList.csv")
    if not os.path.exists(filelist_path):
        raise FileNotFoundError(f"FileList.csv not found: {filelist_path}")

    filelist = pd.read_csv(filelist_path)

    if split and str(split).upper() != "ALL":
        split_norm = str(split).upper()
        filelist = filelist[filelist["Split"].str.upper() == split_norm]

    if max_videos is not None and max_videos > 0:
        filelist = filelist.head(max_videos)

    out = pd.DataFrame()
    out["FileName"] = filelist["FileName"].astype(str)
    out["FileNameWithExt"] = out["FileName"] + ".avi"
    out["Split"] = filelist["Split"].astype(str)

    if include_meta:
        for col in ["NumberOfFrames", "FPS", "FrameHeight", "FrameWidth", "EF", "EDV", "ESV"]:
            if col in filelist.columns:
                out[col] = filelist[col]

    out["ED"] = ""
    out["ES"] = ""

    if prefill_from_tracings:
        prefill_df = _build_tracing_prefill_table(
            data_dir=data_dir,
            method=prefill_method,
            smooth_window=prefill_smooth_window,
            enforce_es_after_ed=enforce_es_after_ed,
        )
        out = out.merge(prefill_df, on="FileNameWithExt", how="left")
        out["ED"] = out["ED_prefill"].apply(_format_frame_value)
        out["ES"] = out["ES_prefill"].apply(_format_frame_value)
        out.drop(columns=["ED_prefill", "ES_prefill"], inplace=True)

    out["Reviewer"] = ""
    out["Notes"] = ""

    return out


def main():
    parser = argparse.ArgumentParser(
        description="Generate ED/ES annotation template from FileList.csv, optionally prefilled from VolumeTracings.csv"
    )
    parser.add_argument("--data-dir", type=str, default=config.DATA_DIR, help="Path containing FileList.csv")
    parser.add_argument("--split", type=str, default="ALL", help="ALL | TRAIN | VAL | TEST")
    parser.add_argument("--max-videos", type=int, default=None, help="Optional cap for quick templates")
    parser.add_argument(
        "--include-meta",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Include useful metadata columns (FPS/frames/EF/EDV/ESV)",
    )
    parser.add_argument(
        "--prefill-from-tracings",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Prefill ED/ES using VolumeTracings.csv",
    )
    parser.add_argument(
        "--prefill-method",
        type=str,
        default="global_extrema",
        choices=["global_extrema", "curve"],
        help="Method used when prefilling ED/ES from tracings",
    )
    parser.add_argument(
        "--prefill-smooth-window",
        type=int,
        default=5,
        help="Smoothing window used in curve prefill mode",
    )
    parser.add_argument(
        "--enforce-es-after-ed",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enforce ES after ED when using curve prefill mode",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=os.path.join("validation", "outputs", "reference_frame_template.csv"),
        help="Output CSV path",
    )
    args = parser.parse_args()

    out_dir = os.path.dirname(args.output)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    template_df = build_template(
        data_dir=args.data_dir,
        split=args.split,
        max_videos=args.max_videos,
        include_meta=bool(args.include_meta),
        prefill_from_tracings=bool(args.prefill_from_tracings),
        prefill_method=args.prefill_method,
        prefill_smooth_window=int(args.prefill_smooth_window),
        enforce_es_after_ed=bool(args.enforce_es_after_ed),
    )
    template_df.to_csv(args.output, index=False)

    print("=" * 72)
    print("REFERENCE TEMPLATE GENERATED")
    print("=" * 72)
    print(f"Rows:                 {len(template_df)}")
    print(f"Split:                {args.split}")
    print(f"Prefill from tracings:{bool(args.prefill_from_tracings)}")
    if args.prefill_from_tracings:
        print(f"Prefill method:       {args.prefill_method}")
    print(f"Output:               {os.path.abspath(args.output)}")
    if args.prefill_from_tracings:
        print("ED/ES columns are prefilled from VolumeTracings.csv and can be edited if needed.")
    else:
        print("Fill ED and ES columns with expert frame numbers, then use it in validation.")
    print("=" * 72)


if __name__ == "__main__":
    main()
