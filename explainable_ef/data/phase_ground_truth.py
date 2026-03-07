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


def extract_frame_area_series(video_rows):
    """Return sorted frame ids and LV areas for each traced frame."""
    frame_ids = sorted(int(frame_id) for frame_id in video_rows["Frame"].unique())
    if len(frame_ids) == 0:
        return np.array([], dtype=np.int32), np.array([], dtype=np.float64)

    areas = []
    for frame_id in frame_ids:
        frame_rows = video_rows[video_rows["Frame"] == frame_id]
        areas.append(compute_lv_area(frame_rows))

    return np.array(frame_ids, dtype=np.int32), np.array(areas, dtype=np.float64)


def smooth_area_curve(values, window=5):
    """Simple edge-padded moving average smoother."""
    values = np.asarray(values, dtype=np.float64)
    if values.size <= 2:
        return values.copy()

    window = int(max(1, window))
    if window % 2 == 0:
        window += 1

    if window > values.size:
        window = values.size if values.size % 2 == 1 else max(1, values.size - 1)

    if window <= 1:
        return values.copy()

    pad = window // 2
    padded = np.pad(values, (pad, pad), mode="edge")
    kernel = np.ones(window, dtype=np.float64) / float(window)
    return np.convolve(padded, kernel, mode="valid")


def _find_local_maxima(values):
    if values.size < 3:
        return np.array([], dtype=np.int32)
    mask = (values[1:-1] >= values[:-2]) & (values[1:-1] >= values[2:])
    return np.where(mask)[0].astype(np.int32) + 1


def _find_local_minima(values):
    if values.size < 3:
        return np.array([], dtype=np.int32)
    mask = (values[1:-1] <= values[:-2]) & (values[1:-1] <= values[2:])
    return np.where(mask)[0].astype(np.int32) + 1


def detect_ed_es_from_area_curve(frame_ids, areas, smooth_window=5, enforce_es_after_ed=True):
    """
    Detect ED/ES from a temporal LV-area curve.

    ED is the largest cavity frame and ES is the smallest cavity frame, with optional
    cycle-order constraint that ES should occur after ED in the sampled sequence.
    """
    frame_ids = np.asarray(frame_ids, dtype=np.int32)
    areas = np.asarray(areas, dtype=np.float64)

    if frame_ids.size == 0:
        return {
            "ed_frame": -1,
            "es_frame": -1,
            "ed_area": 0.0,
            "es_area": 0.0,
            "ed_index": -1,
            "es_index": -1,
            "smoothed_areas": np.array([], dtype=np.float64),
        }

    smoothed = smooth_area_curve(areas, window=smooth_window)

    ed_candidates = _find_local_maxima(smoothed)
    es_candidates = _find_local_minima(smoothed)

    if ed_candidates.size == 0:
        ed_candidates = np.array([int(np.argmax(smoothed))], dtype=np.int32)
    if es_candidates.size == 0:
        es_candidates = np.array([int(np.argmin(smoothed))], dtype=np.int32)

    best_pair = None
    best_drop = -np.inf

    if enforce_es_after_ed:
        for ed_idx in ed_candidates:
            es_after = es_candidates[es_candidates > ed_idx]
            if es_after.size == 0:
                continue

            es_idx = int(es_after[np.argmin(smoothed[es_after])])
            drop = float(smoothed[ed_idx] - smoothed[es_idx])
            if drop > best_drop:
                best_drop = drop
                best_pair = (int(ed_idx), int(es_idx))

    if best_pair is None:
        # Fallback to global extrema if local extrema pairing is not possible.
        ed_idx = int(np.argmax(smoothed))
        es_idx = int(np.argmin(smoothed))

        if enforce_es_after_ed and es_idx <= ed_idx:
            later_indices = np.arange(ed_idx + 1, smoothed.size, dtype=np.int32)
            if later_indices.size > 0:
                es_idx = int(later_indices[np.argmin(smoothed[later_indices])])

        best_pair = (ed_idx, es_idx)

    ed_idx, es_idx = best_pair

    return {
        "ed_frame": int(frame_ids[ed_idx]),
        "es_frame": int(frame_ids[es_idx]),
        "ed_area": float(areas[ed_idx]),
        "es_area": float(areas[es_idx]),
        "ed_index": int(ed_idx),
        "es_index": int(es_idx),
        "smoothed_areas": smoothed,
    }


def compute_ed_es_from_video_rows(
    video_rows,
    method="global_extrema",
    smooth_window=5,
    enforce_es_after_ed=True,
):
    """
    Compute ED/ES frames from tracing rows of a single video.

    Args:
        method: "global_extrema" or "curve"
        smooth_window: moving-average window used only in "curve" mode
        enforce_es_after_ed: in "curve" mode, prefer ES after ED in sequence

    Returns:
        {
            "ed_frame": int,
            "es_frame": int,
            "ed_area": float,
            "es_area": float,
            "num_traced_frames": int,
        }
    """
    frame_ids, areas = extract_frame_area_series(video_rows)
    if frame_ids.size == 0:
        return {
            "ed_frame": -1,
            "es_frame": -1,
            "ed_area": 0.0,
            "es_area": 0.0,
            "num_traced_frames": 0,
        }

    method = str(method).strip().lower()

    if method == "curve":
        detected = detect_ed_es_from_area_curve(
            frame_ids=frame_ids,
            areas=areas,
            smooth_window=smooth_window,
            enforce_es_after_ed=enforce_es_after_ed,
        )
        ed_frame = detected["ed_frame"]
        es_frame = detected["es_frame"]
        ed_area = detected["ed_area"]
        es_area = detected["es_area"]
    elif method == "global_extrema":
        ed_idx = int(np.argmax(areas))
        es_idx = int(np.argmin(areas))
        ed_frame = int(frame_ids[ed_idx])
        es_frame = int(frame_ids[es_idx])
        ed_area = float(areas[ed_idx])
        es_area = float(areas[es_idx])
    else:
        raise ValueError("method must be one of: global_extrema, curve")

    return {
        "ed_frame": int(ed_frame),
        "es_frame": int(es_frame),
        "ed_area": float(ed_area),
        "es_area": float(es_area),
        "num_traced_frames": int(frame_ids.size),
    }
