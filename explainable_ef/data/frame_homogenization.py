import copy
import json
import os

import cv2
import numpy as np


PREPROCESS_PRESETS = {
    "off": {
        "enabled": False,
        "pipeline": "off",
        "preset": "off",
        "harmonization": {"enabled": False},
        "enhancement": {"enabled": False},
    },
    "conservative": {
        "enabled": True,
        "pipeline": "harmonize_then_enhance",
        "preset": "conservative",
        "harmonization": {
            "enabled": True,
            "method": "percentile",
            "lower_percentile": 1.0,
            "upper_percentile": 99.0,
            "blend": 0.25,
            "target_low": 0.0,
            "target_high": 0.70,
        },
        "enhancement": {
            "enabled": True,
            "denoise_method": "bilateral",
            "bilateral_d": 3,
            "bilateral_sigma_color": 12.0,
            "bilateral_sigma_space": 3.0,
            "nlm_h": 3.0,
            "clahe_clip_limit": 0.4,
            "clahe_tile_grid_size": 8,
            "unsharp_amount": 0.12,
            "unsharp_radius": 1.0,
            "unsharp_threshold": 0.025,
        },
    },
    "balanced": {
        "enabled": True,
        "pipeline": "harmonize_then_enhance",
        "preset": "balanced",
        "harmonization": {
            "enabled": True,
            "method": "percentile",
            "lower_percentile": 1.0,
            "upper_percentile": 99.0,
            "blend": 0.40,
            "target_low": 0.0,
            "target_high": 0.70,
        },
        "enhancement": {
            "enabled": True,
            "denoise_method": "bilateral",
            "bilateral_d": 5,
            "bilateral_sigma_color": 18.0,
            "bilateral_sigma_space": 5.0,
            "nlm_h": 4.0,
            "clahe_clip_limit": 0.7,
            "clahe_tile_grid_size": 8,
            "unsharp_amount": 0.18,
            "unsharp_radius": 1.0,
            "unsharp_threshold": 0.02,
        },
    },
    "aggressive": {
        "enabled": True,
        "pipeline": "harmonize_then_enhance",
        "preset": "aggressive",
        "harmonization": {
            "enabled": True,
            "method": "percentile",
            "lower_percentile": 1.0,
            "upper_percentile": 99.0,
            "blend": 0.60,
            "target_low": 0.0,
            "target_high": 0.72,
        },
        "enhancement": {
            "enabled": True,
            "denoise_method": "bilateral",
            "bilateral_d": 7,
            "bilateral_sigma_color": 25.0,
            "bilateral_sigma_space": 7.0,
            "nlm_h": 6.0,
            "clahe_clip_limit": 1.0,
            "clahe_tile_grid_size": 8,
            "unsharp_amount": 0.25,
            "unsharp_radius": 1.0,
            "unsharp_threshold": 0.015,
        },
    },
}

DEFAULT_HOMOGENIZATION = copy.deepcopy(PREPROCESS_PRESETS["balanced"])


def _deep_update(base, updates):
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _deep_update(base[key], value)
        else:
            base[key] = value
    return base


def _legacy_to_structured(stats):
    method = str(stats.get("method", "")).strip().lower()
    if stats.get("pipeline") or method not in {"luma_unsharp", "luma_percentile_unsharp", "luma_mean_std_clahe"}:
        return stats

    converted = copy.deepcopy(PREPROCESS_PRESETS["balanced"])
    converted["enabled"] = bool(stats.get("enabled", True))
    converted["pipeline"] = "harmonize_then_enhance"
    converted["preset"] = "legacy"
    converted["legacy_method"] = method

    converted["harmonization"]["enabled"] = method == "luma_percentile_unsharp"
    converted["harmonization"]["target_low"] = float(stats.get("target_low", 0.0))
    converted["harmonization"]["target_high"] = float(stats.get("target_high", 0.70))
    converted["harmonization"]["blend"] = float(stats.get("contrast_blend", 0.40))
    converted["harmonization"]["lower_percentile"] = float(stats.get("contrast_lower_percentile", 1.0))
    converted["harmonization"]["upper_percentile"] = float(stats.get("contrast_upper_percentile", 99.0))

    converted["enhancement"]["enabled"] = method != "luma_mean_std_clahe"
    converted["enhancement"]["denoise_method"] = "none"
    converted["enhancement"]["clahe_clip_limit"] = float(stats.get("clahe_clip_limit", 0.0))
    converted["enhancement"]["clahe_tile_grid_size"] = int(stats.get("clahe_tile_grid_size", 8))
    converted["enhancement"]["unsharp_amount"] = float(stats.get("unsharp_amount", 0.0))
    converted["enhancement"]["unsharp_radius"] = float(stats.get("unsharp_radius", 1.0))
    converted["enhancement"]["unsharp_threshold"] = float(stats.get("unsharp_threshold", 0.02))
    return converted


def load_homogenization_stats(path):
    if path is None or str(path).strip() == "":
        return None
    if not os.path.exists(path):
        raise FileNotFoundError(f"Homogenization stats not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        stats = json.load(f)

    stats = _legacy_to_structured(stats)
    preset_name = str(stats.get("preset", "balanced")).strip().lower()
    preset = PREPROCESS_PRESETS.get(preset_name, PREPROCESS_PRESETS["balanced"])
    merged = copy.deepcopy(preset)
    _deep_update(merged, stats)
    merged["enabled"] = bool(merged.get("enabled", True))
    return merged


def save_homogenization_stats(path, stats):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, sort_keys=True)


def _clip01(value):
    return min(1.0, max(0.0, float(value)))


def _apply_harmonization_luma(y, settings):
    if not settings or not bool(settings.get("enabled", False)):
        return y

    method = str(settings.get("method", "percentile")).strip().lower()
    if method != "percentile":
        return y

    lower_percentile = float(settings.get("lower_percentile", 1.0))
    upper_percentile = float(settings.get("upper_percentile", 99.0))
    current_low = float(np.percentile(y, lower_percentile))
    current_high = float(np.percentile(y, upper_percentile))
    target_low = _clip01(settings.get("target_low", 0.0))
    target_high = _clip01(settings.get("target_high", 0.70))
    blend = _clip01(settings.get("blend", 0.40))

    if current_high - current_low <= 1e-6 or target_high <= target_low:
        return y

    normalized = (y - current_low) / (current_high - current_low)
    normalized = np.clip(normalized, 0.0, 1.0)
    harmonized = target_low + normalized * (target_high - target_low)
    return np.clip((1.0 - blend) * y + blend * harmonized, 0.0, 1.0)


def _apply_denoise(y_u8, settings):
    method = str(settings.get("denoise_method", "bilateral")).strip().lower()
    if method in {"", "none", "off"}:
        return y_u8
    if method == "nlm":
        h = max(0.0, float(settings.get("nlm_h", 4.0)))
        return cv2.fastNlMeansDenoising(y_u8, None, h=h, templateWindowSize=7, searchWindowSize=21)

    d = max(1, int(settings.get("bilateral_d", 5)))
    sigma_color = max(0.0, float(settings.get("bilateral_sigma_color", 18.0)))
    sigma_space = max(0.0, float(settings.get("bilateral_sigma_space", 5.0)))
    return cv2.bilateralFilter(y_u8, d=d, sigmaColor=sigma_color, sigmaSpace=sigma_space)


def _apply_clahe(y_u8, settings):
    clip_limit = float(settings.get("clahe_clip_limit", 0.7))
    tile_size = int(settings.get("clahe_tile_grid_size", 8))
    if clip_limit > 0.0 and tile_size > 1:
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_size, tile_size))
        return clahe.apply(y_u8)
    return y_u8


def _apply_unsharp_luma(y, settings):
    amount = max(0.0, float(settings.get("unsharp_amount", 0.18)))
    radius = max(0.0, float(settings.get("unsharp_radius", 1.0)))
    threshold = max(0.0, float(settings.get("unsharp_threshold", 0.02)))
    if amount <= 0.0 or radius <= 0.0:
        return y

    blurred = cv2.GaussianBlur(y, ksize=(0, 0), sigmaX=radius, sigmaY=radius)
    detail = y - blurred
    if threshold > 0.0:
        detail = np.where(np.abs(detail) >= threshold, detail, 0.0)
    return np.clip(y + amount * detail, 0.0, 1.0)


def _apply_enhancement_luma(y, settings):
    if not settings or not bool(settings.get("enabled", False)):
        return y

    y_u8 = np.clip(y * 255.0, 0.0, 255.0).astype(np.uint8)
    y_u8 = _apply_denoise(y_u8, settings)
    y_u8 = _apply_clahe(y_u8, settings)
    y = y_u8.astype(np.float32) / 255.0
    return _apply_unsharp_luma(y, settings)


def _replace_luma(frame_rgb, y):
    ycrcb = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2YCrCb)
    ycrcb[:, :, 0] = np.clip(y * 255.0, 0.0, 255.0).astype(np.uint8)
    return cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2RGB)


def apply_frame_preprocessing_steps(frame_rgb, stats):
    frame_rgb = np.asarray(frame_rgb, dtype=np.uint8)
    if not stats or not bool(stats.get("enabled", False)):
        return {
            "original": frame_rgb,
            "harmonized": frame_rgb,
            "enhanced": frame_rgb,
        }

    ycrcb = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2YCrCb)
    y = ycrcb[:, :, 0].astype(np.float32) / 255.0

    harmonized_y = _apply_harmonization_luma(y, stats.get("harmonization", {}))
    enhanced_y = _apply_enhancement_luma(harmonized_y, stats.get("enhancement", {}))

    return {
        "original": frame_rgb,
        "harmonized": _replace_luma(frame_rgb, harmonized_y),
        "enhanced": _replace_luma(frame_rgb, enhanced_y),
    }


def apply_frame_homogenization(frame_rgb, stats):
    return apply_frame_preprocessing_steps(frame_rgb, stats)["enhanced"]


def apply_video_homogenization(frames_rgb, stats):
    if not stats or not bool(stats.get("enabled", False)):
        return frames_rgb
    return np.stack([apply_frame_homogenization(frame, stats) for frame in frames_rgb], axis=0)
