"""
Video Preprocessing Script
Pre-decodes all videos to individual frames for ~10x faster training.
Run this ONCE before training: python preprocess_videos.py
"""

import os
import cv2
import pandas as pd
import numpy as np
from tqdm import tqdm
import config
from pathlib import Path

def preprocess_videos():
    """Pre-decode all videos into individual frame files."""
    
    data_dir = config.DATA_DIR
    output_dir = os.path.join(data_dir, "PreprocessedFrames")
    
    print(f"Loading dataset from: {data_dir}")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Read file list
    filelist_path = os.path.join(data_dir, "FileList.csv")
    filelist = pd.read_csv(filelist_path)
    
    print(f"Total videos: {len(filelist)}")
    
    # Process each video
    processed = 0
    skipped = 0
    
    for idx, row in tqdm(filelist.iterrows(), total=len(filelist), desc="Preprocessing videos"):
        try:
            video_path = os.path.join(data_dir, "Videos", row["FileName"] + ".avi")
            
            if not os.path.exists(video_path):
                print(f"  WARNING: Video not found: {video_path}")
                skipped += 1
                continue
            
            # Create output folder for this video
            video_name = row["FileName"]
            video_output_dir = os.path.join(output_dir, video_name)
            os.makedirs(video_output_dir, exist_ok=True)
            
            # Skip if already processed
            if os.path.exists(os.path.join(video_output_dir, "frames_info.txt")):
                processed += 1
                continue
            
            # Decode video
            cap = cv2.VideoCapture(video_path)
            frame_count = 0
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Resize frame
                frame = cv2.resize(frame, (config.IMAGE_SIZE, config.IMAGE_SIZE))
                
                # Save as numpy array (faster than PNG)
                frame_path = os.path.join(video_output_dir, f"frame_{frame_count:04d}.npy")
                np.save(frame_path, frame)
                
                frame_count += 1
                
                # Limit to NUM_FRAMES + 10 buffer
                if frame_count > config.NUM_FRAMES + 10:
                    break
            
            cap.release()
            
            # Save metadata
            with open(os.path.join(video_output_dir, "frames_info.txt"), "w") as f:
                f.write(f"num_frames:{frame_count}\n")
                f.write(f"ef:{row['EF']}\n")
            
            processed += 1
            
        except Exception as e:
            print(f"  ERROR processing {row['FileName']}: {e}")
            skipped += 1
            continue
    
    print(f"\n{'='*80}")
    print(f"Preprocessing Complete!")
    print(f"  Processed: {processed}")
    print(f"  Skipped: {skipped}")
    print(f"  Output: {output_dir}")
    print(f"{'='*80}")
    print(f"\nNext steps:")
    print(f"1. Update dataset.py to use preprocess_videos parameter")
    print(f"2. Set PREPROCESSED_FRAMES = True in config.py")
    print(f"3. Training will now be ~10x faster!")

if __name__ == "__main__":
    print("="*80)
    print("VIDEO PREPROCESSING SCRIPT")
    print("="*80)
    print(f"\nThis script will pre-decode all videos to individual frames.")
    print(f"Expected output size: {len(pd.read_csv(os.path.join(config.DATA_DIR, 'FileList.csv')))} videos")
    print(f"\nWarning: This will take ~30-60 minutes depending on your CPU.")
    print(f"Output directory: {os.path.join(config.DATA_DIR, 'PreprocessedFrames')}\n")
    
    response = input("Continue? (yes/no): ")
    if response.lower() == "yes":
        preprocess_videos()
    else:
        print("Aborted.")
