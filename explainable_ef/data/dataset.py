import os
import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from data.phase_ground_truth import compute_ed_es_from_video_rows


KINETICS_MEAN = (0.43216, 0.394666, 0.37645)
KINETICS_STD = (0.22803, 0.22145, 0.216989)


class EchoDataset(Dataset):
    def __init__(
        self,
        data_dir,
        num_frames=32,
        length=None,
        frame_size=(112, 112),
        split="TRAIN",
        max_videos=None,
        transform=None,
        normalize_input=True,
        period=1,
        max_length=None,
        sampling_mode="global",
        clips=1,
        pad=None,
        noise=None,
        temporal_window_mode="full",
        temporal_window_margin_mult=1.5,
        temporal_window_jitter_mult=0.0,
    ):
        self.data_dir = data_dir
        resolved_length = num_frames if length is None else length
        self.num_frames = int(resolved_length) if resolved_length is not None else None
        self.length = int(resolved_length) if resolved_length is not None else None
        self.frame_size = frame_size
        self.max_videos = max_videos
        self.transform = transform
        self.normalize_input = bool(normalize_input)
        self.period = max(1, int(period))
        self.max_length = None if max_length is None else max(1, int(max_length))
        sampling_mode = str(sampling_mode).strip().lower()
        if sampling_mode in {"full", "full_video"}:
            sampling_mode = "global"
        if sampling_mode not in {"global", "echonet"}:
            raise ValueError("sampling_mode must be 'global' or 'echonet'")
        self.sampling_mode = sampling_mode
        self.clips = clips if clips == "all" else max(1, int(clips))
        self.pad = None if pad is None else max(0, int(pad))
        self.noise = None if noise is None else float(min(1.0, max(0.0, noise)))
        self.split = str(split).strip().upper()
        # Retained for caller compatibility; clip sampling no longer uses ED/ES
        # labels to crop the video before sampling.
        self.temporal_window_mode = str(temporal_window_mode).strip().lower()
        self.temporal_window_margin_mult = float(max(0.0, temporal_window_margin_mult))
        self.temporal_window_jitter_mult = float(max(0.0, temporal_window_jitter_mult))

        if self.length is not None and self.length <= 0:
            raise ValueError("length/num_frames must be positive")

        self._mean = torch.tensor(KINETICS_MEAN, dtype=torch.float32).view(3, 1, 1, 1)
        self._std = torch.tensor(KINETICS_STD, dtype=torch.float32).view(3, 1, 1, 1)

        if not os.path.exists(data_dir):
            raise FileNotFoundError(f"Data directory not found: {data_dir}")

        filelist_path = os.path.join(data_dir, "FileList.csv")
        if not os.path.exists(filelist_path):
            raise FileNotFoundError(f"FileList.csv not found in {data_dir}")

        volume_filelist_path = os.path.join(data_dir, "VolumeTracings.csv")
        if not os.path.exists(volume_filelist_path):
            raise FileNotFoundError(f"VolumeTracings.csv not found in {data_dir}")

        self.filelist = pd.read_csv(filelist_path)
        self.filelist = self.filelist[self.filelist["Split"] == split]

        self.volume_tracing = pd.read_csv(volume_filelist_path)

        if max_videos is not None and max_videos > 0:
            self.filelist = self.filelist.iloc[:max_videos]

        if len(self.filelist) == 0:
            raise ValueError("No training samples found in FileList.csv")

        self.phase_dict = {}

        for video_name in self.filelist["FileName"].unique():
            file_name_with_extension = video_name + ".avi"
            video_rows = self.volume_tracing[self.volume_tracing["FileName"] == file_name_with_extension]

            phase_info = compute_ed_es_from_video_rows(video_rows)
            self.phase_dict[file_name_with_extension] = {
                "ed": phase_info["ed_frame"],
                "es": phase_info["es_frame"],
            }

    def __len__(self):
        return len(self.filelist)

    def _resolve_clip_length(self, total_video_frames):
        if self.length is None:
            clip_length = max(1, int(total_video_frames) // int(self.period))
            if self.max_length is not None:
                clip_length = min(clip_length, int(self.max_length))
        else:
            clip_length = int(self.length)

        return max(1, int(clip_length))

    def _sample_start_positions(self, total_video_frames, clip_length):
        """
        EchoNet-style clip starts over the padded video timeline.

        For training, starts are random. For validation/test, starts are
        deterministic so repeated evaluation is stable.
        """
        total_video_frames = int(total_video_frames)
        clip_length = int(max(1, clip_length))
        padded_frames = max(total_video_frames, clip_length * self.period)
        num_start_positions = max(1, padded_frames - (clip_length - 1) * self.period)

        if self.clips == "all":
            return np.arange(num_start_positions, dtype=np.int32), padded_frames

        num_clips = int(self.clips)
        if num_clips <= 1:
            if self.split == "TRAIN":
                start = int(np.random.randint(0, num_start_positions))
            else:
                start = num_start_positions // 2
            return np.array([start], dtype=np.int32), padded_frames

        if self.split == "TRAIN":
            replace = num_clips > num_start_positions
            starts = np.random.choice(num_start_positions, size=num_clips, replace=replace)
            starts = np.sort(starts.astype(np.int32, copy=False))
        else:
            starts = np.linspace(0, num_start_positions - 1, num_clips).round().astype(np.int32)

        return starts, padded_frames

    def _sample_global_indices(self, total_video_frames, clip_length):
        total_video_frames = int(total_video_frames)
        clip_length = int(max(1, clip_length))

        if total_video_frames >= clip_length:
            indices = np.linspace(0, total_video_frames - 1, clip_length).round().astype(np.int32)
        else:
            observed = np.arange(total_video_frames, dtype=np.int32)
            pad_count = clip_length - total_video_frames
            if total_video_frames > 0 and pad_count > 0:
                tail = np.full((pad_count,), total_video_frames - 1, dtype=np.int32)
                indices = np.concatenate([observed, tail], axis=0)
            else:
                indices = observed

        if self.clips == "all":
            return np.expand_dims(indices, axis=0)

        num_clips = 1 if self.clips == "all" else int(self.clips)
        if num_clips <= 1:
            return indices

        return np.repeat(indices[None, :], num_clips, axis=0)

    def _apply_noise(self, frames_array):
        if self.noise is None or self.noise <= 0.0:
            return frames_array

        noisy = frames_array.copy()
        total_pixels = noisy.shape[0] * noisy.shape[1] * noisy.shape[2]
        num_mask = int(round(self.noise * total_pixels))
        if num_mask <= 0:
            return noisy

        ind = np.random.choice(total_pixels, num_mask, replace=False)
        frame_idx = ind // (noisy.shape[1] * noisy.shape[2])
        pixel_idx = ind % (noisy.shape[1] * noisy.shape[2])
        row_idx = pixel_idx // noisy.shape[2]
        col_idx = pixel_idx % noisy.shape[2]
        noisy[frame_idx, row_idx, col_idx, :] = 0
        return noisy

    def _apply_pad_crop(self, frames_tensor):
        if self.pad is None or self.pad <= 0:
            return frames_tensor

        if frames_tensor.ndim == 4:
            c, t, h, w = frames_tensor.shape
            padded = torch.zeros((c, t, h + 2 * self.pad, w + 2 * self.pad), dtype=frames_tensor.dtype)
            padded[:, :, self.pad : self.pad + h, self.pad : self.pad + w] = frames_tensor
            i, j = np.random.randint(0, 2 * self.pad + 1, size=2)
            return padded[:, :, i : i + h, j : j + w]

        clips = []
        for clip in frames_tensor:
            clips.append(self._apply_pad_crop(clip))
        return torch.stack(clips, dim=0)

    def load_video(self, path, ed_original=-1, es_original=-1):
        _ = ed_original, es_original
        cap = cv2.VideoCapture(path)
        frames = []

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.resize(frame, self.frame_size)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)

        cap.release()

        if len(frames) == 0:
            raise ValueError(f"No frames loaded from video: {path}")

        frames_array = np.array(frames, dtype=np.uint8)
        frames_array = self._apply_noise(frames_array)
        total_video_frames = len(frames_array)
        clip_length = self._resolve_clip_length(total_video_frames)
        if self.sampling_mode == "global":
            sampled_indices = self._sample_global_indices(total_video_frames, clip_length)
        else:
            start_positions, padded_frames = self._sample_start_positions(total_video_frames, clip_length)

            if padded_frames > total_video_frames:
                frames_array = np.pad(
                    frames_array,
                    ((0, padded_frames - total_video_frames), (0, 0), (0, 0), (0, 0)),
                    mode="constant",
                    constant_values=0,
                )

            offsets = self.period * np.arange(clip_length, dtype=np.int32)
            sampled_indices = start_positions[:, None] + offsets[None, :]
        sampled_frames = frames_array[sampled_indices]

        if sampled_frames.shape[0] == 1:
            sampled_frames = sampled_frames[0]
            sampled_indices = sampled_indices[0]
            frames_tensor = torch.from_numpy(sampled_frames).permute(3, 0, 1, 2).float() / 255.0
            if self.normalize_input:
                frames_tensor = (frames_tensor - self._mean) / self._std
        else:
            frames_tensor = torch.from_numpy(sampled_frames).permute(0, 4, 1, 2, 3).float() / 255.0
            if self.normalize_input:
                frames_tensor = (frames_tensor - self._mean.unsqueeze(0)) / self._std.unsqueeze(0)

        frames_tensor = self._apply_pad_crop(frames_tensor)

        if self.transform is not None:
            frames_tensor = self.transform(frames_tensor)

        return frames_tensor, sampled_indices

    def __getitem__(self, idx):
        row = self.filelist.iloc[idx]
        file_name_with_extension = row["FileName"] + ".avi"
        video_path = os.path.join(self.data_dir, "Videos", file_name_with_extension)
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")

        ed_original = self.phase_dict[file_name_with_extension]["ed"]
        es_original = self.phase_dict[file_name_with_extension]["es"]

        ef = torch.tensor(row["EF"]).float() / 100.0

        video, sampled_indices = self.load_video(
            video_path,
            ed_original=ed_original,
            es_original=es_original,
        )

        sampled_indices = np.asarray(sampled_indices, dtype=np.int32)

        if ed_original >= 0:
            ed_idx = np.argmin(np.abs(sampled_indices - int(ed_original)), axis=-1).astype(np.int64)
        else:
            ed_idx = np.zeros(sampled_indices.shape[:-1], dtype=np.int64)

        if es_original >= 0:
            es_idx = np.argmin(np.abs(sampled_indices - int(es_original)), axis=-1).astype(np.int64)
        else:
            es_idx = np.zeros(sampled_indices.shape[:-1], dtype=np.int64)

        if np.ndim(ed_idx) == 0:
            ed_idx = int(ed_idx)
        if np.ndim(es_idx) == 0:
            es_idx = int(es_idx)

        return video, ef, ed_idx, es_idx
