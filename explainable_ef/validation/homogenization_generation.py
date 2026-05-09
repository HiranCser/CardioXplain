from pathlib import Path
import cv2
import numpy as np
from data.frame_homogenization import load_homogenization_stats, apply_frame_homogenization

data_dir = Path(r"D:\datascience\MTech\Sem4\Project\CardioXplain\cx\dynamic\a4c-video-dir")
stats_path = r"../frame_homogenization.json"
out_path = Path(r"outputs\homogenization\before_after_homogenization.png")

stats = load_homogenization_stats(stats_path)
videos = list((data_dir / "Videos").glob("*.avi")) or list(data_dir.rglob("*.avi"))
if not videos:
    raise RuntimeError(f"No .avi videos found under {data_dir}")

cap = cv2.VideoCapture(str(videos[0]))
frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, frame_count // 2))
ok, before_bgr = cap.read()
cap.release()
if not ok:
    raise RuntimeError(f"Could not read frame from {videos[0]}")

before_rgb = cv2.cvtColor(before_bgr, cv2.COLOR_BGR2RGB)
after_rgb = apply_frame_homogenization(before_rgb, stats)
after_bgr = cv2.cvtColor(after_rgb, cv2.COLOR_RGB2BGR)

canvas = np.hstack([before_bgr, after_bgr])
cv2.putText(canvas, "Before", (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
cv2.putText(canvas, "After homogenization", (before_bgr.shape[1] + 20, 35), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)

out_path.parent.mkdir(parents=True, exist_ok=True)
cv2.imwrite(str(out_path), canvas)
print(out_path.resolve())
