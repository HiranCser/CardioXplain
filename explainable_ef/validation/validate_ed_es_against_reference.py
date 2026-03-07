import argparse
import os
import sys

import pandas as pd

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import config
from data.phase_ground_truth import compute_ed_es_from_video_rows, normalize_filename


def infer_column(df, user_value, candidates, required=True):
    if user_value:
        if user_value not in df.columns:
            raise ValueError(f"Column '{user_value}' not found in reference file")
        return user_value

    lower_map = {c.lower(): c for c in df.columns}
    for candidate in candidates:
        if candidate.lower() in lower_map:
            return lower_map[candidate.lower()]

    if required:
        raise ValueError(
            f"Could not infer column. Available columns: {list(df.columns)} | "
            f"Expected one of: {candidates}"
        )
    return None


def load_reference_table(path, sheet_name=None):
    ext = os.path.splitext(path)[1].lower()

    if ext == ".csv":
        return pd.read_csv(path)

    if ext in {".xlsx", ".xls"}:
        try:
            return pd.read_excel(path, sheet_name=sheet_name if sheet_name else 0)
        except ImportError as exc:
            raise ImportError(
                "Reading Excel requires openpyxl (for .xlsx) or xlrd (for .xls). "
                "Install with: pip install openpyxl"
            ) from exc

    raise ValueError(f"Unsupported reference file extension: {ext}")


def build_computed_table(data_dir, split=None, max_videos=None):
    filelist_path = os.path.join(data_dir, "FileList.csv")
    tracings_path = os.path.join(data_dir, "VolumeTracings.csv")

    filelist = pd.read_csv(filelist_path)
    if split:
        filelist = filelist[filelist["Split"] == split]

    if max_videos is not None and max_videos > 0:
        filelist = filelist.head(max_videos)

    tracings = pd.read_csv(tracings_path)

    rows = []
    for _, row in filelist.iterrows():
        fname = row["FileName"]
        fname_ext = f"{fname}.avi"

        vr = tracings[tracings["FileName"] == fname_ext]
        phase_info = compute_ed_es_from_video_rows(vr)

        rows.append(
            {
                "file_name": fname,
                "file_name_with_ext": fname_ext,
                "split": row["Split"],
                "ed_calc": phase_info["ed_frame"],
                "es_calc": phase_info["es_frame"],
                "ed_area": phase_info["ed_area"],
                "es_area": phase_info["es_area"],
                "num_traced_frames": phase_info["num_traced_frames"],
            }
        )

    return pd.DataFrame(rows)


def build_reference_from_volume_tracings(reference_df):
    file_col = infer_column(reference_df, None, ["FileName", "file_name", "filename"])
    frame_col = infer_column(reference_df, None, ["Frame", "frame", "frame_id"])
    x1_col = infer_column(reference_df, None, ["X1", "x1"])
    y1_col = infer_column(reference_df, None, ["Y1", "y1"])
    x2_col = infer_column(reference_df, None, ["X2", "x2"])
    y2_col = infer_column(reference_df, None, ["Y2", "y2"])

    df = reference_df[[file_col, frame_col, x1_col, y1_col, x2_col, y2_col]].copy()
    df.columns = ["FileName", "Frame", "X1", "Y1", "X2", "Y2"]

    rows = []
    for file_name, group in df.groupby("FileName"):
        phase_info = compute_ed_es_from_video_rows(group)
        rows.append(
            {
                "file_name_ref": normalize_filename(file_name),
                "ed_ref": phase_info["ed_frame"],
                "es_ref": phase_info["es_frame"],
            }
        )

    return pd.DataFrame(rows)


def try_build_reference_frame_table(reference_df, args):
    file_col = infer_column(reference_df, args.file_col, ["FileName", "file_name", "filename", "video", "video_name"])
    ed_col = infer_column(reference_df, args.ed_col, ["ED", "ed", "ed_frame", "ED_Frame", "EDFrame", "ed frame", "ed_index"])
    es_col = infer_column(reference_df, args.es_col, ["ES", "es", "es_frame", "ES_Frame", "ESFrame", "es frame", "es_index"])

    ref = reference_df[[file_col, ed_col, es_col]].copy()
    ref.columns = ["file_name_ref", "ed_ref", "es_ref"]
    ref["file_name_ref"] = ref["file_name_ref"].apply(normalize_filename)
    return ref


def prepare_reference_table(reference_df, args):
    mode = args.reference_mode.lower()
    if mode not in {"auto", "frame_table", "volume_tracings"}:
        raise ValueError("--reference-mode must be one of: auto, frame_table, volume_tracings")

    if mode == "frame_table":
        return try_build_reference_frame_table(reference_df, args), "frame_table"

    if mode == "volume_tracings":
        return build_reference_from_volume_tracings(reference_df), "volume_tracings"

    # auto
    try:
        return try_build_reference_frame_table(reference_df, args), "frame_table"
    except Exception:
        pass

    try:
        return build_reference_from_volume_tracings(reference_df), "volume_tracings"
    except Exception as exc:
        raise ValueError(
            "Could not auto-detect reference format. "
            "Use --reference-mode frame_table with --file-col/--ed-col/--es-col, "
            "or --reference-mode volume_tracings for tracing-style files. "
            f"Columns found: {list(reference_df.columns)}"
        ) from exc


def validate_against_reference(computed_df, reference_prepared_df, args):
    comp = computed_df.copy()
    comp["file_name_norm"] = comp["file_name"].apply(normalize_filename)

    ref = reference_prepared_df.copy()
    ref["file_name_norm"] = ref["file_name_ref"].apply(normalize_filename)

    matched = comp.merge(ref[["file_name_norm", "ed_ref", "es_ref"]], on="file_name_norm", how="inner")

    if len(matched) == 0:
        raise ValueError("No rows matched between computed file names and reference file names.")

    matched["ed_ref"] = matched["ed_ref"].astype(int)
    matched["es_ref"] = matched["es_ref"].astype(int)

    matched["ed_abs_error"] = (matched["ed_calc"] - matched["ed_ref"]).abs()
    matched["es_abs_error"] = (matched["es_calc"] - matched["es_ref"]).abs()

    matched["ed_exact"] = matched["ed_abs_error"] == 0
    matched["es_exact"] = matched["es_abs_error"] == 0
    matched["joint_exact"] = matched["ed_exact"] & matched["es_exact"]

    tol = max(0, int(args.tolerance))
    matched["ed_within_tol"] = matched["ed_abs_error"] <= tol
    matched["es_within_tol"] = matched["es_abs_error"] <= tol
    matched["joint_within_tol"] = matched["ed_within_tol"] & matched["es_within_tol"]

    summary = {
        "rows_computed": int(len(computed_df)),
        "rows_matched": int(len(matched)),
        "ed_mae": float(matched["ed_abs_error"].mean()),
        "es_mae": float(matched["es_abs_error"].mean()),
        "ed_exact_rate": float(matched["ed_exact"].mean()),
        "es_exact_rate": float(matched["es_exact"].mean()),
        "joint_exact_rate": float(matched["joint_exact"].mean()),
        "ed_within_tol_rate": float(matched["ed_within_tol"].mean()),
        "es_within_tol_rate": float(matched["es_within_tol"].mean()),
        "joint_within_tol_rate": float(matched["joint_within_tol"].mean()),
        "tolerance": tol,
    }

    return matched, summary


def main():
    parser = argparse.ArgumentParser(description="Validate computed ED/ES frame GT against reference Excel/CSV frame numbers.")
    parser.add_argument("--data-dir", type=str, default=config.DATA_DIR, help="Path containing FileList.csv and VolumeTracings.csv")
    parser.add_argument("--split", type=str, default=None, help="Optional split filter: TRAIN/VAL/TEST")
    parser.add_argument("--max-videos", type=int, default=None, help="Optional cap for quick checks")
    parser.add_argument("--reference", type=str, default=None, help="Path to reference CSV/XLSX/XLS with frame numbers")
    parser.add_argument("--reference-mode", type=str, default="auto", help="auto | frame_table | volume_tracings")
    parser.add_argument("--sheet", type=str, default=None, help="Excel sheet name (optional)")
    parser.add_argument("--file-col", type=str, default=None, help="Reference filename column name")
    parser.add_argument("--ed-col", type=str, default=None, help="Reference ED frame column name")
    parser.add_argument("--es-col", type=str, default=None, help="Reference ES frame column name")
    parser.add_argument("--tolerance", type=int, default=0, help="Tolerance in frames for within-tolerance rates")
    parser.add_argument(
        "--output",
        type=str,
        default=os.path.join("validation", "outputs", "ed_es_validation.csv"),
        help="CSV output path",
    )
    parser.add_argument(
        "--mismatch-output",
        type=str,
        default=os.path.join("validation", "outputs", "ed_es_mismatches.csv"),
        help="CSV output path for mismatches only",
    )
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    os.makedirs(os.path.dirname(args.mismatch_output), exist_ok=True)

    computed_df = build_computed_table(
        data_dir=args.data_dir,
        split=args.split,
        max_videos=args.max_videos,
    )

    if not args.reference:
        computed_df.to_csv(args.output, index=False)
        print("=" * 72)
        print("ED/ES COMPUTED TABLE EXPORTED")
        print("=" * 72)
        print(f"Rows:          {len(computed_df)}")
        print(f"Output:        {os.path.abspath(args.output)}")
        print("Tip: pass --reference <excel_or_csv> to compare against your annotated frame numbers.")
        print("=" * 72)
        return

    reference_df = load_reference_table(args.reference, sheet_name=args.sheet)
    reference_prepared_df, detected_mode = prepare_reference_table(reference_df, args)
    matched_df, summary = validate_against_reference(computed_df, reference_prepared_df, args)

    matched_df.to_csv(args.output, index=False)
    mismatches = matched_df[~matched_df["joint_within_tol"]].copy()
    mismatches.to_csv(args.mismatch_output, index=False)

    print("=" * 72)
    print("ED/ES REFERENCE VALIDATION SUMMARY")
    print("=" * 72)
    print(f"Reference mode:           {detected_mode}")
    print(f"Computed rows:            {summary['rows_computed']}")
    print(f"Matched rows:             {summary['rows_matched']}")
    print(f"Tolerance:                +/-{summary['tolerance']} frames")
    print(f"ED MAE (frames):          {summary['ed_mae']:.3f}")
    print(f"ES MAE (frames):          {summary['es_mae']:.3f}")
    print(f"ED exact match:           {summary['ed_exact_rate'] * 100:.2f}%")
    print(f"ES exact match:           {summary['es_exact_rate'] * 100:.2f}%")
    print(f"Joint exact match:        {summary['joint_exact_rate'] * 100:.2f}%")
    print(f"ED within tolerance:      {summary['ed_within_tol_rate'] * 100:.2f}%")
    print(f"ES within tolerance:      {summary['es_within_tol_rate'] * 100:.2f}%")
    print(f"Joint within tolerance:   {summary['joint_within_tol_rate'] * 100:.2f}%")
    print(f"Detailed output:          {os.path.abspath(args.output)}")
    print(f"Mismatch output:          {os.path.abspath(args.mismatch_output)}")
    print("=" * 72)


if __name__ == "__main__":
    main()
