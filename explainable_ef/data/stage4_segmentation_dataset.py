import os

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from pipeline.stage45_pipeline import Stage45Pipeline


class Stage4SegmentationDataset(Dataset):
    """Frame-level Stage-4 dataset built from VolumeTracings.csv."""

    def __init__(self, data_dir, split="TRAIN", image_size=112, max_videos=None, normalize="none"):
        self.data_dir = data_dir
        self.split = str(split).upper()
        self.image_size = int(image_size)
        self.normalize = str(normalize).lower()

        filelist_path = os.path.join(data_dir, "FileList.csv")
        tracings_path = os.path.join(data_dir, "VolumeTracings.csv")

        if not os.path.exists(filelist_path):
            raise FileNotFoundError(f"FileList.csv not found in {data_dir}")
        if not os.path.exists(tracings_path):
            raise FileNotFoundError(f"VolumeTracings.csv not found in {data_dir}")

        filelist = pd.read_csv(filelist_path)
        if "Split" not in filelist.columns:
            raise ValueError("FileList.csv must contain a 'Split' column")

        filelist["Split"] = filelist["Split"].astype(str).str.upper()
        filelist = filelist[filelist["Split"] == self.split].copy()
        if max_videos is not None and int(max_videos) > 0:
            filelist = filelist.head(int(max_videos))

        self.filelist = filelist.reset_index(drop=True)
        self.tracings = pd.read_csv(tracings_path)

        expected_cols = {"FileName", "X1", "Y1", "X2", "Y2", "Frame"}
        if not expected_cols.issubset(set(self.tracings.columns)):
            raise ValueError(f"VolumeTracings.csv must contain columns: {sorted(expected_cols)}")

        self.group_indices = self.tracings.groupby(["FileName", "Frame"]).indices

        self.samples = []
        for _, row in self.filelist.iterrows():
            file_name = str(row["FileName"]).strip()
            file_name_ext = file_name + ".avi"
            frame_height = int(row["FrameHeight"])
            frame_width = int(row["FrameWidth"])

            video_rows = self.tracings[self.tracings["FileName"] == file_name_ext]
            if video_rows.empty:
                continue

            frame_ids = sorted(int(v) for v in video_rows["Frame"].unique().tolist())
            for frame_id in frame_ids:
                key = (file_name_ext, frame_id)
                if key not in self.group_indices:
                    continue

                idxs = self.group_indices[key]
                frame_rows = self.tracings.iloc[idxs].sort_index()
                gt_mask_orig = Stage45Pipeline.tracing_to_mask(frame_rows, height=frame_height, width=frame_width)
                gt_area_orig = float(gt_mask_orig.sum())

                self.samples.append(
                    {
                        "file_name": file_name,
                        "file_name_ext": file_name_ext,
                        "frame_id": int(frame_id),
                        "frame_height": frame_height,
                        "frame_width": frame_width,
                        "trace_indices": idxs,
                        "gt_area_orig": gt_area_orig,
                    }
                )

        if not self.samples:
            raise ValueError(f"No Stage-4 samples found for split={self.split}")

    def __len__(self):
        return len(self.samples)

    def _read_video_frame(self, video_path, frame_id):
        cap = cv2.VideoCapture(video_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_id))
        ok, frame_bgr = cap.read()
        cap.release()
        if not ok or frame_bgr is None:
            raise ValueError(f"Could not read frame {frame_id} from {video_path}")
        return frame_bgr

    def _normalize_image(self, image_tensor):
        if self.normalize == "imagenet":
            mean = torch.tensor([0.485, 0.456, 0.406], dtype=image_tensor.dtype).view(3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225], dtype=image_tensor.dtype).view(3, 1, 1)
            image_tensor = (image_tensor - mean) / std
        return image_tensor

    def __getitem__(self, idx):
        sample = self.samples[idx]
        file_name_ext = sample["file_name_ext"]
        frame_id = sample["frame_id"]

        video_path = os.path.join(self.data_dir, "Videos", file_name_ext)
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")

        frame_bgr = self._read_video_frame(video_path, frame_id)
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        frame_resized = cv2.resize(frame_rgb, (self.image_size, self.image_size), interpolation=cv2.INTER_LINEAR)

        frame_rows = self.tracings.iloc[sample["trace_indices"]].sort_index()
        gt_mask_orig = Stage45Pipeline.tracing_to_mask(
            frame_rows,
            height=sample["frame_height"],
            width=sample["frame_width"],
        )
        gt_mask_resized = cv2.resize(
            gt_mask_orig.astype(np.uint8),
            (self.image_size, self.image_size),
            interpolation=cv2.INTER_NEAREST,
        )

        image_tensor = torch.from_numpy(frame_resized).permute(2, 0, 1).float() / 255.0
        image_tensor = self._normalize_image(image_tensor)
        mask_tensor = torch.from_numpy(gt_mask_resized).unsqueeze(0).float()

        return {
            "image": image_tensor,
            "mask": mask_tensor,
            "file_name": sample["file_name"],
            "file_name_ext": file_name_ext,
            "frame_id": int(frame_id),
            "frame_height": int(sample["frame_height"]),
            "frame_width": int(sample["frame_width"]),
            "gt_area_orig": float(sample["gt_area_orig"]),
        }
