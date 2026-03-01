import os
import torch
import pandas as pd
import cv2
import numpy as np
from torch.utils.data import Dataset

class EchoDataset(Dataset):
    def __init__(self, data_dir, num_frames=32, frame_size=(112, 112), max_videos=None, transform=None):
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
        
        self.filelist = pd.read_csv(filelist_path)
        self.filelist = self.filelist[self.filelist["Split"] == "TRAIN"]
        
        # Limit to max_videos if specified
        if max_videos is not None and max_videos > 0:
            self.filelist = self.filelist.iloc[:max_videos]
        
        if len(self.filelist) == 0:
            raise ValueError("No training samples found in FileList.csv")

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
        video_path = os.path.join(self.data_dir, "Videos", row["FileName"] + ".avi")
        
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")
        
        ef = torch.tensor(row["EF"]).float() / 100.0  # Normalize EF to [0, 1]

        video = self.load_video(video_path)

        return video, ef