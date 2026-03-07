import cv2
import numpy as np


def normalize_filename(name):
    """Normalize video identifiers by removing .avi suffix and trimming spaces."""
    name = str(name).strip()
    if name.lower().endswith(".avi"):
        name = name[:-4]
    return name


def compute_lv_area(frame_rows):
    """Compute LV cavity area from tracing rows using ordered contour points."""
    if frame_rows.empty:
        return 0.0

    left_wall = frame_rows[["X1", "Y1"]].to_numpy(dtype=np.float32)
    right_wall = frame_rows[["X2", "Y2"]].to_numpy(dtype=np.float32)[::-1]
    contour = np.concatenate([left_wall, right_wall], axis=0)

    if contour.shape[0] < 3:
        return 0.0
    return float(cv2.contourArea(contour))


def compute_ed_es_from_video_rows(video_rows):
    """
    Compute ED/ES frames from all tracing rows of a single video.

    Returns:
        {
            "ed_frame": int,
            "es_frame": int,
            "ed_area": float,
            "es_area": float,
            "num_traced_frames": int,
        }
    """
    frame_ids = video_rows["Frame"].unique()
    if len(frame_ids) == 0:
        return {
            "ed_frame": -1,
            "es_frame": -1,
            "ed_area": 0.0,
            "es_area": 0.0,
            "num_traced_frames": 0,
        }

    frame_ids_int = []
    areas = []

    for frame_id in frame_ids:
        frame_rows = video_rows[video_rows["Frame"] == frame_id]
        area = compute_lv_area(frame_rows)
        frame_ids_int.append(int(frame_id))
        areas.append(area)

    frame_ids_np = np.array(frame_ids_int, dtype=np.int32)
    areas_np = np.array(areas, dtype=np.float64)

    ed_idx = int(np.argmax(areas_np))
    es_idx = int(np.argmin(areas_np))

    return {
        "ed_frame": int(frame_ids_np[ed_idx]),
        "es_frame": int(frame_ids_np[es_idx]),
        "ed_area": float(areas_np[ed_idx]),
        "es_area": float(areas_np[es_idx]),
        "num_traced_frames": int(len(frame_ids_np)),
    }
