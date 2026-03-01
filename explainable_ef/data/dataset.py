import os
import torch
import pandas as pd
import cv2
import numpy as np
from torch.utils.data import Dataset

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

            frame_ids = video_rows["Frame"].unique()

            areas = []

            for f in frame_ids:
                frame_rows = video_rows[video_rows["Frame"] == f]

                points = []
                for _, row in frame_rows.iterrows():
                    points.append([row["X1"], row["Y1"]])
                    points.append([row["X2"], row["Y2"]])

                points = np.unique(np.array(points), axis=0)
                area = cv2.contourArea(points.astype(np.float32))
                areas.append(area)

            ed_frame = -1
            es_frame = -1

            if len(frame_ids) > 0:
                ed_frame = frame_ids[np.argmax(areas)]
                es_frame = frame_ids[np.argmin(areas)]

            self.phase_dict[file_name_with_extension] = {
                "ed": ed_frame,
                "es": es_frame
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

        # Convert list of numpy arrays to single numpy array (much faster than torch.tensor on list)
        frames_array = np.array(frames, dtype=np.uint8)  # Shape: (num_frames, H, W, C)
        
        # Handle frame count mismatch
        if len(frames_array) < self.num_frames:
            # Pad with zeros if video has fewer frames than required
            padding = self.num_frames - len(frames_array)
            frames_array = np.pad(
                frames_array, 
                ((0, padding), (0, 0), (0, 0), (0, 0)), 
                mode='constant', 
                constant_values=0
            )
        else:
            # Truncate if video has more frames than required
            frames_array = frames_array[:self.num_frames]
        
        # Convert (num_frames, height, width, channels) to (channels, num_frames, height, width)
        frames_tensor = torch.from_numpy(frames_array).permute(3, 0, 1, 2).float() / 255.0
        return frames_tensor

    def __getitem__(self, idx):
        row = self.filelist.iloc[idx]
        file_name_with_extension = row["FileName"] + ".avi"
        video_path = os.path.join(self.data_dir, "Videos", file_name_with_extension)
        total_frames = self.filelist.iloc[idx]["NumberOfFrames"]
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")
        
        ed_original = self.phase_dict[file_name_with_extension]["ed"]
        es_original = self.phase_dict[file_name_with_extension]["es"]

        ed_idx = int(ed_original / total_frames * self.num_frames)
        es_idx = int(es_original / total_frames * self.num_frames)

        ed_idx = min(ed_idx, self.num_frames - 1)
        es_idx = min(es_idx, self.num_frames - 1)

        ef = torch.tensor(row["EF"]).float() / 100.0  # Normalize EF to [0, 1]

        video = self.load_video(video_path)

        return video, ef, ed_idx, es_idx