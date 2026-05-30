import os
import json
import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from data.phase_ground_truth import compute_ed_es_from_video_rows, compute_continuous_phase_labels


KINETICS_MEAN = (0.43216, 0.394666, 0.37645)
KINETICS_STD = (0.22803, 0.22145, 0.216989)


class EchoDataset(Dataset):
    def __init__(
        self,
        data_dir,
        num_frames=32,
        frame_size=(112, 112),
        split="TRAIN",
        max_videos=None,
        transform=None,
        normalize_input=True,
        clip_period=1,
        clip_eval_mode="all",
        train_clips_per_video=1,
        clip_start_mode="random",
        clip_prior_path=None,
        clip_prior_jitter_std=0.5,
    ):
        self.data_dir = data_dir
        self.num_frames = int(num_frames)
        self.frame_size = frame_size
        self.max_videos = max_videos
        self.transform = transform
        self.normalize_input = bool(normalize_input)
        self.split = str(split).strip().upper()
        self.clip_period = max(1, int(clip_period))
        self.clip_eval_mode = str(clip_eval_mode).strip().lower()
        self.train_clips_per_video = max(1, int(train_clips_per_video)) if self.split == "TRAIN" else 1
        self.clip_start_mode = str(clip_start_mode).strip().lower()
        self.clip_prior_jitter_std = max(0.0, float(clip_prior_jitter_std))
        self.clip_prior = self._load_clip_prior(clip_prior_path)
        if self.clip_eval_mode not in {"center", "all"}:
            raise ValueError(f"Unsupported clip_eval_mode: {clip_eval_mode}")
        if self.clip_start_mode not in {"random", "center", "prior"}:
            raise ValueError(f"Unsupported clip_start_mode: {clip_start_mode}")
        if self.clip_start_mode == "prior" and self.clip_prior is None:
            raise ValueError("clip_start_mode='prior' requires clip_prior_path")

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

    @staticmethod
    def _load_clip_prior(path):
        if path is None or str(path).strip() == "":
            return None
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    def __len__(self):
        return len(self.filelist) * self.train_clips_per_video

    def _prior_center_frame(self, total_video_frames):
        priors = self.clip_prior.get("clip_center_priors", {}) if self.clip_prior else {}
        if not priors:
            return None

        if self.split == "TRAIN":
            keys = ["pair_mid_rel_mean", "ed_rel_mean", "es_rel_mean"]
            key = str(np.random.choice(keys, p=np.array([0.5, 0.25, 0.25])))
        else:
            key = "pair_mid_rel_mean"

        rel = float(priors.get(key, priors.get("pair_mid_rel_mean", 0.5)))
        std_key = key.replace("_mean", "_std")
        rel_std = float(priors.get(std_key, 0.0))
        if self.split == "TRAIN" and rel_std > 0.0 and self.clip_prior_jitter_std > 0.0:
            rel += float(np.random.normal(0.0, rel_std * self.clip_prior_jitter_std))
        rel = float(np.clip(rel, 0.0, 1.0))
        return int(round(rel * max(0, int(total_video_frames) - 1)))

    def _clip_start_indices(self, total_video_frames, mode=None, ed_original=-1, es_original=-1, contain_events=False):
        required_frames = (self.num_frames - 1) * self.clip_period + 1
        padded_frames = max(int(total_video_frames), int(required_frames))
        max_start = max(0, padded_frames - required_frames)

        start_low = 0
        start_high = max_start
        if contain_events and ed_original >= 0 and es_original >= 0:
            left = min(int(ed_original), int(es_original))
            right = max(int(ed_original), int(es_original))
            start_low = max(start_low, right - required_frames + 1)
            start_high = min(start_high, left)

            if start_low > start_high:
                start_low = 0
                start_high = max_start

        if self.split == "TRAIN":
            if self.clip_start_mode == "center":
                return np.array([max_start // 2], dtype=np.int32)
            if self.clip_start_mode == "prior":
                center = self._prior_center_frame(total_video_frames)
                if center is not None:
                    start = int(np.clip(center - required_frames // 2, 0, max_start))
                    return np.array([start], dtype=np.int32)
            return np.array([np.random.randint(start_low, start_high + 1)], dtype=np.int32)

        eval_mode = self.clip_eval_mode if mode is None else str(mode).strip().lower()
        if eval_mode == "all":
            return np.arange(max_start + 1, dtype=np.int32)
        if eval_mode != "center":
            raise ValueError(f"Unsupported clip eval mode: {eval_mode}")
        if self.clip_start_mode == "prior":
            center = self._prior_center_frame(total_video_frames)
            if center is not None:
                return np.array([int(np.clip(center - required_frames // 2, 0, max_start))], dtype=np.int32)
        return np.array([(start_low + start_high) // 2], dtype=np.int32)

    def _clip_indices_from_start(self, start, total_video_frames):
        raw_indices = int(start) + self.clip_period * np.arange(self.num_frames, dtype=np.int32)
        if total_video_frames <= 0:
            return raw_indices
        return np.minimum(raw_indices, int(total_video_frames) - 1).astype(np.int32)

    def _frames_to_tensor(self, sampled_frames):
        frames_tensor = torch.from_numpy(sampled_frames).permute(3, 0, 1, 2).float() / 255.0

        if self.normalize_input:
            frames_tensor = (frames_tensor - self._mean) / self._std

        if self.transform is not None:
            frames_tensor = self.transform(frames_tensor)

        return frames_tensor

    def _read_video_frames(self, path):
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

        return np.array(frames, dtype=np.uint8)

    def _sample_clip_from_frames(self, frames_array, start):
        total_video_frames = len(frames_array)
        raw_indices = int(start) + self.clip_period * np.arange(self.num_frames, dtype=np.int32)
        valid_mask = raw_indices < total_video_frames

        sampled_frames = np.zeros(
            (self.num_frames, frames_array.shape[1], frames_array.shape[2], frames_array.shape[3]),
            dtype=frames_array.dtype,
        )
        if np.any(valid_mask):
            sampled_frames[valid_mask] = frames_array[raw_indices[valid_mask]]

        sampled_indices = self._clip_indices_from_start(start, total_video_frames)
        return self._frames_to_tensor(sampled_frames), sampled_indices

    def load_video(self, path, ed_original=-1, es_original=-1, contain_events=None):
        frames_array = self._read_video_frames(path)
        if contain_events is None:
            contain_events = int(ed_original) >= 0 and int(es_original) >= 0
        start = self._clip_start_indices(
            len(frames_array),
            mode="center" if self.split != "TRAIN" else None,
            ed_original=ed_original,
            es_original=es_original,
            contain_events=bool(contain_events),
        )[0]
        return self._sample_clip_from_frames(frames_array, start)

    @staticmethod
    def _original_to_clip_index(original_frame, sampled_indices):
        """Map an original video frame id to the nearest sampled clip index."""
        sampled_indices = np.asarray(sampled_indices, dtype=np.int32)
        if sampled_indices.size == 0:
            return 0
        return int(np.argmin(np.abs(sampled_indices - int(original_frame))))

    @staticmethod
    def _frame_is_visible(original_frame, sampled_indices):
        sampled_indices = np.asarray(sampled_indices, dtype=np.int32)
        if sampled_indices.size == 0:
            return False
        return bool(np.any(sampled_indices == int(original_frame)))

    def load_video_clips(self, path, mode=None):
        frames_array = self._read_video_frames(path)
        starts = self._clip_start_indices(len(frames_array), mode=mode)
        clips = []
        sampled_indices = []
        for start in starts:
            clip, indices = self._sample_clip_from_frames(frames_array, start)
            clips.append(clip)
            sampled_indices.append(indices)

        return torch.stack(clips, dim=0), np.stack(sampled_indices, axis=0)

    def __getitem__(self, idx):
        video_idx = int(idx) // self.train_clips_per_video if self.split == "TRAIN" else int(idx)
        row = self.filelist.iloc[video_idx]
        file_name_with_extension = row["FileName"] + ".avi"
        video_path = os.path.join(self.data_dir, "Videos", file_name_with_extension)
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")

        ed_original = self.phase_dict[file_name_with_extension]["ed"]
        es_original = self.phase_dict[file_name_with_extension]["es"]

        ef = torch.tensor(row["EF"]).float() / 100.0

        video, sampled_indices = self.load_video(video_path)

        ed_idx = torch.tensor(self._original_to_clip_index(ed_original, sampled_indices), dtype=torch.long)
        es_idx = torch.tensor(self._original_to_clip_index(es_original, sampled_indices), dtype=torch.long)
        ed_visible = torch.tensor(self._frame_is_visible(ed_original, sampled_indices), dtype=torch.bool)
        es_visible = torch.tensor(self._frame_is_visible(es_original, sampled_indices), dtype=torch.bool)

        return video, ef, ed_idx, es_idx, ed_visible, es_visible
