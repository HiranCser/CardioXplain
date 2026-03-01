# Performance Optimization Guide

## Current Bottlenecks
Based on your 94.46s/batch timing, the main bottleneck is **video I/O and frame loading** from disk.

## ✓ Applied Optimizations (Already Implemented)
1. **Parallel Data Loading**: `num_workers=4` in DataLoader
2. **Prefetching**: `prefetch_factor=2` for better memory utilization
3. **Persistent Workers**: Reduces worker restart overhead
4. **Numpy-based conversion**: Faster than torch.tensor on lists

## 🚀 Additional Optimizations to Apply

### Option 1: Reduce Frame Sampling (Fastest)
**Expected speedup: 2-3x**
```python
# In config.py
NUM_FRAMES = 16  # Instead of 32 (sample every other frame)
```
- Reduce from 32 frames to 16 frames (sample every 2nd frame)
- Loss in temporal information is minimal for EF prediction

### Option 2: Reduce Image Resolution
**Expected speedup: 1.5-2x**
```python
# In config.py
IMAGE_SIZE = 56  # Instead of 112
```
- Change from 112x112 to 56x56 pixels
- Significantly reduces memory bandwidth

### Option 3: Pre-process Videos (Fastest - ~10x speedup)
**One-time cost, huge training speedup**
Create preprocessed video frames:
```bash
# In PowerShell
python preprocess_videos.py
```
Then update dataset.py to load from preprocessed files instead of video files.

### Option 4: Use Model Quantization
**Expected speedup: 1.2-1.5x**
Use int8 quantization on CPU:
```python
# In train.py (after loading model)
model = torch.quantization.quantize_dynamic(
    model, {torch.nn.Linear}, dtype=torch.qint8
)
```

### Option 5: Enable Torch Optimization (Python 3.11+)
**Expected speedup: 1.1-1.3x**
```python
# In train.py (after model creation)
model = torch.compile(model)  # Requires Python 3.11+
```

### Option 6: Reduce Batch Size for Faster Iteration
**For development/debugging only**
```python
# In config.py
BATCH_SIZE = 8  # Instead of 20
```

## 📊 Recommended Combined Approach
For fastest training on CPU:
1. **First**: Apply Option 3 (preprocess videos) - One-time cost
2. **Then**: Reduce frames to 16 (Option 1)
3. **Then**: Consider reducing image to 84x84 (Option 2)

**Expected total speedup: 15-30x faster training**

## Performance Comparison Estimates

| Optimization | Speedup | Quality Loss |
|---|---|---|
| Original | 1x | Baseline |
| + Num_workers=4 | 2-4x | None |
| + Reduce frames 32→16 | 4-6x | Minimal |
| + Preprocess Videos | 20-30x | None |
| + Image resize 112→64 | 30-50x | Very Small |

## Monitoring Performance
- Check batch processing time in progress bar
- If still slow: verify `num_workers` is working (use `nvidia-smi` for GPU if available)
- Monitor CPU/Memory usage during training

## Next Steps
Would you like me to:
1. Create a video preprocessing script for Option 3?
2. Adjust frames/image size with Options 1-2?
3. Add model quantization code?
