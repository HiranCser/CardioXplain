import math

import numpy as np
import torch


AVAILABLE_PERTURBATIONS = (
    "frame_drop",
    "random_mask",
    "attention_guided_mask",
    "contiguous_mask",
    "temporal_shift",
    "local_shuffle",
    "reverse_window",
)


def summarize_temporal_attention(attention):
    """Collapse attention outputs into a 1D temporal importance curve."""
    attn = np.asarray(attention, dtype=np.float64)
    if attn.ndim == 2 and attn.shape[1] > 0:
        attn = attn.mean(axis=1)
    else:
        attn = attn.reshape(-1)

    if attn.size == 0:
        return np.zeros(0, dtype=np.float64)

    attn = np.clip(attn, 0.0, None)
    total = float(attn.sum())
    if not math.isfinite(total) or total <= 0.0:
        return np.full(attn.shape, 1.0 / float(attn.size), dtype=np.float64)
    return attn / total


def _clone_clip(video):
    if not torch.is_tensor(video):
        raise TypeError("video must be a torch.Tensor")
    if video.ndim != 4:
        raise ValueError("video must have shape (C, T, H, W)")
    return video.detach().clone()


def _num_frames(video):
    return int(video.shape[1])


def _severity_to_count(num_frames, severity):
    severity = float(severity)
    if not math.isfinite(severity):
        raise ValueError("severity must be finite")
    severity = min(1.0, max(0.0, severity))
    return max(1, min(num_frames, int(round(severity * num_frames))))


def _mask_indices(video, indices):
    out = _clone_clip(video)
    out[:, indices, :, :] = 0
    return out


def _window_start(num_frames, window, rng):
    if window >= num_frames:
        return 0
    return int(rng.integers(0, num_frames - window + 1))


def perturb_random_mask(video, severity, rng, frame_scores=None):
    del frame_scores
    t = _num_frames(video)
    count = _severity_to_count(t, severity)
    indices = np.sort(rng.choice(t, size=count, replace=False).astype(np.int32))
    out = _mask_indices(video, indices.tolist())
    return out, {"masked_indices": indices.tolist(), "count": int(count)}


def perturb_attention_guided_mask(video, severity, rng, frame_scores=None):
    del rng
    if frame_scores is None:
        raise ValueError("attention_guided_mask requires frame_scores")
    t = _num_frames(video)
    count = _severity_to_count(t, severity)
    scores = np.asarray(frame_scores, dtype=np.float64).reshape(-1)
    if scores.size != t:
        raise ValueError("frame_scores must have the same temporal length as the clip")
    order = np.argsort(-scores, kind="stable")
    indices = np.sort(order[:count].astype(np.int32))
    out = _mask_indices(video, indices.tolist())
    return out, {"masked_indices": indices.tolist(), "count": int(count)}


def perturb_contiguous_mask(video, severity, rng, frame_scores=None):
    del frame_scores
    t = _num_frames(video)
    window = _severity_to_count(t, severity)
    start = _window_start(t, window, rng)
    indices = np.arange(start, start + window, dtype=np.int32)
    out = _mask_indices(video, indices.tolist())
    return out, {"start": int(start), "window": int(window), "masked_indices": indices.tolist()}


def perturb_temporal_shift(video, severity, rng, frame_scores=None):
    del frame_scores
    t = _num_frames(video)
    max_shift = _severity_to_count(t, severity)
    max_shift = min(t - 1, max_shift)
    if max_shift <= 0:
        return _clone_clip(video), {"shift": 0}

    shift = int(rng.integers(-max_shift, max_shift + 1))
    if shift == 0:
        shift = max_shift if max_shift > 0 else 0

    out = _clone_clip(video)
    if shift > 0:
        out[:, shift:, :, :] = video[:, : t - shift, :, :]
        out[:, :shift, :, :] = video[:, :1, :, :].expand(-1, shift, -1, -1)
    else:
        shift_abs = abs(shift)
        out[:, : t - shift_abs, :, :] = video[:, shift_abs:, :, :]
        out[:, t - shift_abs :, :, :] = video[:, -1:, :, :].expand(-1, shift_abs, -1, -1)
    return out, {"shift": int(shift)}


def perturb_local_shuffle(video, severity, rng, frame_scores=None):
    del frame_scores
    t = _num_frames(video)
    window = _severity_to_count(t, severity)
    start = _window_start(t, window, rng)
    indices = np.arange(start, start + window, dtype=np.int32)
    shuffled = indices.copy()
    rng.shuffle(shuffled)

    out = _clone_clip(video)
    out[:, indices.tolist(), :, :] = video[:, shuffled.tolist(), :, :]
    return out, {
        "start": int(start),
        "window": int(window),
        "original_indices": indices.tolist(),
        "shuffled_indices": shuffled.tolist(),
    }


def perturb_reverse_window(video, severity, rng, frame_scores=None):
    del frame_scores
    t = _num_frames(video)
    window = _severity_to_count(t, severity)
    start = _window_start(t, window, rng)
    indices = np.arange(start, start + window, dtype=np.int32)
    reversed_indices = indices[::-1]

    out = _clone_clip(video)
    out[:, indices.tolist(), :, :] = video[:, reversed_indices.tolist(), :, :]
    return out, {
        "start": int(start),
        "window": int(window),
        "original_indices": indices.tolist(),
        "reversed_indices": reversed_indices.tolist(),
    }


def perturb_frame_drop(video, severity, rng, frame_scores=None):
    del frame_scores
    t = _num_frames(video)
    drop_count = _severity_to_count(t, severity)
    if drop_count >= t:
        drop_count = t - 1
    if drop_count <= 0:
        return _clone_clip(video), {"dropped_indices": []}

    dropped = np.sort(rng.choice(t, size=drop_count, replace=False).astype(np.int32))
    keep_mask = np.ones(t, dtype=bool)
    keep_mask[dropped] = False
    kept = np.arange(t, dtype=np.int32)[keep_mask]
    resample_positions = np.linspace(0, len(kept) - 1, t).round().astype(np.int32)
    source_indices = kept[resample_positions]

    out = _clone_clip(video)
    out[:, :, :, :] = video[:, source_indices.tolist(), :, :]
    return out, {
        "dropped_indices": dropped.tolist(),
        "kept_indices": kept.tolist(),
        "source_indices": source_indices.tolist(),
    }


def apply_temporal_perturbation(video, perturbation, severity, rng, frame_scores=None):
    perturbation = str(perturbation).strip().lower()
    if perturbation == "random_mask":
        return perturb_random_mask(video, severity, rng, frame_scores=frame_scores)
    if perturbation == "attention_guided_mask":
        return perturb_attention_guided_mask(video, severity, rng, frame_scores=frame_scores)
    if perturbation == "contiguous_mask":
        return perturb_contiguous_mask(video, severity, rng, frame_scores=frame_scores)
    if perturbation == "temporal_shift":
        return perturb_temporal_shift(video, severity, rng, frame_scores=frame_scores)
    if perturbation == "local_shuffle":
        return perturb_local_shuffle(video, severity, rng, frame_scores=frame_scores)
    if perturbation == "reverse_window":
        return perturb_reverse_window(video, severity, rng, frame_scores=frame_scores)
    if perturbation == "frame_drop":
        return perturb_frame_drop(video, severity, rng, frame_scores=frame_scores)
    raise ValueError(f"Unknown perturbation: {perturbation}")
