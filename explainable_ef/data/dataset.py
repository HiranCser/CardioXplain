import os
import torch
import pandas as pd
import cv2
import numpy as np
from torch.utils.data import Dataset

from data.phase_ground_truth import compute_ed_es_from_video_rows


class EchoDataset(Dataset):
    def __init__(self, data_dir, num_frames=32, frame_size=(112, 112), split="TRAIN", max_videos=None, transform=None):
        self.data_dir = data_dir
        self.num_frames = num_frames
        self.frame_size = frame_size
        self.max_videos = max_videos
        self.transform = transform

        # Check if data directory exists
        if not os.path.exists(data_dir):
            raise FileNotFoundError(f"Data directory not found: {data_dir}")

        # Load the file list
        filelist_path = os.path.join(data_dir, "FileList.csv")
        if not os.path.exists(filelist_path):
            raise FileNotFoundError(f"FileList.csv not found in {data_dir}")

        volume_filelist_path = os.path.join(data_dir, "VolumeTracings.csv")
        if not os.path.exists(volume_filelist_path):
            raise FileNotFoundError(f"VolumeTracings.csv not found in {data_dir}")

        self.filelist = pd.read_csv(filelist_path)
        self.filelist = self.filelist[self.filelist["Split"] == split]

        self.volume_tracing = pd.read_csv(volume_filelist_path)

        # Limit to max_videos if specified
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

    def load_video(self, path):
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

        # Convert list of numpy arrays to single numpy array (num_total_frames, H, W, C)
        frames_array = np.array(frames, dtype=np.uint8)
        total_video_frames = len(frames_array)

        # Sample frames uniformly across the full video so phase labels and inputs align.
        if total_video_frames >= self.num_frames:
            sampled_indices = np.linspace(0, total_video_frames - 1, self.num_frames).round().astype(np.int32)
            sampled_frames = frames_array[sampled_indices]
        else:
            sampled_indices = np.arange(total_video_frames, dtype=np.int32)
            padding = self.num_frames - total_video_frames
            sampled_frames = np.pad(
                frames_array,
                ((0, padding), (0, 0), (0, 0), (0, 0)),
                mode="constant",
                constant_values=0
            )
            if padding > 0:
                pad_indices = np.full((padding,), total_video_frames - 1, dtype=np.int32)
                sampled_indices = np.concatenate([sampled_indices, pad_indices], axis=0)

        # Convert (num_frames, height, width, channels) to (channels, num_frames, height, width)
        frames_tensor = torch.from_numpy(sampled_frames).permute(3, 0, 1, 2).float() / 255.0
        return frames_tensor, sampled_indices

    def __getitem__(self, idx):
        row = self.filelist.iloc[idx]
        file_name_with_extension = row["FileName"] + ".avi"
        video_path = os.path.join(self.data_dir, "Videos", file_name_with_extension)
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")

        ed_original = self.phase_dict[file_name_with_extension]["ed"]
        es_original = self.phase_dict[file_name_with_extension]["es"]

        ef = torch.tensor(row["EF"]).float() / 100.0  # Normalize EF to [0, 1]

        video, sampled_indices = self.load_video(video_path)

        # Map original traced frame ids to nearest sampled frame index.
        if ed_original >= 0:
            ed_idx = int(np.argmin(np.abs(sampled_indices - int(ed_original))))
        else:
            ed_idx = 0

        if es_original >= 0:
            es_idx = int(np.argmin(np.abs(sampled_indices - int(es_original))))
        else:
            es_idx = 0

        return video, ef, ed_idx, es_idx
