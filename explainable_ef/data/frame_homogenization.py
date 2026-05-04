import json
import os

import cv2
import numpy as np


DEFAULT_HOMOGENIZATION = {
    "enabled": False,
    "target_mean": 0.45,
    "target_std": 0.20,
    "clahe_clip_limit": 2.0,
    "clahe_tile_grid_size": 8,
}


def load_homogenization_stats(path):
    if path is None or str(path).strip() == "":
        return None
    if not os.path.exists(path):
        raise FileNotFoundError(f"Homogenization stats not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        stats = json.load(f)
    merged = dict(DEFAULT_HOMOGENIZATION)
    merged.update(stats)
    merged["enabled"] = bool(merged.get("enabled", True))
    return merged


def save_homogenization_stats(path, stats):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, sort_keys=True)


def apply_frame_homogenization(frame_rgb, stats):
    if not stats or not bool(stats.get("enabled", False)):
        return frame_rgb

    frame_rgb = np.asarray(frame_rgb, dtype=np.uint8)
    ycrcb = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2YCrCb)
    y = ycrcb[:, :, 0].astype(np.float32) / 255.0

    current_mean = float(np.mean(y))
    current_std = float(np.std(y))
    target_mean = float(stats.get("target_mean", DEFAULT_HOMOGENIZATION["target_mean"]))
    target_std = max(1e-6, float(stats.get("target_std", DEFAULT_HOMOGENIZATION["target_std"])))

    y = (y - current_mean) * (target_std / max(1e-6, current_std)) + target_mean
    y_u8 = np.clip(y * 255.0, 0.0, 255.0).astype(np.uint8)

    clip_limit = float(stats.get("clahe_clip_limit", DEFAULT_HOMOGENIZATION["clahe_clip_limit"]))
    tile_size = int(stats.get("clahe_tile_grid_size", DEFAULT_HOMOGENIZATION["clahe_tile_grid_size"]))
    if clip_limit > 0.0 and tile_size > 1:
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_size, tile_size))
        y_u8 = clahe.apply(y_u8)

    ycrcb[:, :, 0] = y_u8
    return cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2RGB)


def apply_video_homogenization(frames_rgb, stats):
    if not stats or not bool(stats.get("enabled", False)):
        return frames_rgb
    return np.stack([apply_frame_homogenization(frame, stats) for frame in frames_rgb], axis=0)
