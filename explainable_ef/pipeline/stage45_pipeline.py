import cv2
import numpy as np

from data.phase_ground_truth import detect_ed_es_from_area_curve


class Stage45Pipeline:
    """Stage 4/5 utilities: LV mask creation, area extraction, and EF computation."""

    @staticmethod
    def tracing_to_contour(frame_rows):
        """
        Build an EchoNet-compatible closed LV contour from tracing pairs.
        This mirrors the logic from dynamic/echonet/datasets/echo.py:
        x = concat(x1[1:], flip(x2[1:])), y = concat(y1[1:], flip(y2[1:])).
        """
        if frame_rows.empty:
            return np.zeros((0, 2), dtype=np.float32)

        t = frame_rows.sort_index()[["X1", "Y1", "X2", "Y2"]].to_numpy(dtype=np.float32)
        if t.shape[0] < 2:
            return np.zeros((0, 2), dtype=np.float32)

        x1, y1, x2, y2 = t[:, 0], t[:, 1], t[:, 2], t[:, 3]
        x = np.concatenate((x1[1:], np.flip(x2[1:])))
        y = np.concatenate((y1[1:], np.flip(y2[1:])))
        contour = np.stack([x, y], axis=1)
        return contour.astype(np.float32)

    @staticmethod
    def tracing_to_mask(frame_rows, height, width):
        """Rasterize tracing contour to a binary LV mask."""
        mask = np.zeros((height, width), dtype=np.uint8)
        if frame_rows.empty:
            return mask

        contour = Stage45Pipeline.tracing_to_contour(frame_rows)
        if contour.shape[0] < 3:
            return mask

        contour_i = np.round(contour).astype(np.int32)
        contour_i[:, 0] = np.clip(contour_i[:, 0], 0, width - 1)
        contour_i[:, 1] = np.clip(contour_i[:, 1], 0, height - 1)

        cv2.fillPoly(mask, [contour_i], color=1)
        return mask

    @staticmethod
    def mask_area(mask):
        """Pixel area of LV mask."""
        return float(mask.sum())

    @staticmethod
    def _odd_kernel(kernel_size):
        k = int(max(0, kernel_size))
        if k <= 1:
            return 0
        if k % 2 == 0:
            k += 1
        return k

    @staticmethod
    def postprocess_mask(mask, keep_largest=True, fill_holes=True, closing_kernel=5, opening_kernel=0):
        """Clean binary LV masks with conservative morphology and largest-component filtering."""
        m = (np.asarray(mask) > 0).astype(np.uint8)
        if m.size == 0:
            return m

        close_k = Stage45Pipeline._odd_kernel(closing_kernel)
        if close_k > 1:
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_k, close_k))
            m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, kernel)

        open_k = Stage45Pipeline._odd_kernel(opening_kernel)
        if open_k > 1:
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (open_k, open_k))
            m = cv2.morphologyEx(m, cv2.MORPH_OPEN, kernel)

        if keep_largest and int(m.sum()) > 0:
            n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
            if n_labels > 1:
                largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
                m = (labels == largest).astype(np.uint8)

        if fill_holes and int(m.sum()) > 0:
            flood = (m * 255).astype(np.uint8)
            h, w = flood.shape[:2]
            flood_mask = np.zeros((h + 2, w + 2), dtype=np.uint8)
            cv2.floodFill(flood, flood_mask, (0, 0), 255)
            holes = cv2.bitwise_not(flood)
            m = (((m * 255) | holes) > 0).astype(np.uint8)

        return (m > 0).astype(np.uint8)

    @staticmethod
    def canonicalize_ed_es_pair(ed_frame, ed_area, es_frame, es_area):
        """Return a physiologically ordered ED/ES pair where ED area is >= ES area."""
        ed_frame = int(ed_frame)
        es_frame = int(es_frame)
        ed_area = float(ed_area)
        es_area = float(es_area)

        if not np.isfinite(ed_area) or not np.isfinite(es_area):
            return {
                "ed_frame": ed_frame,
                "ed_area": ed_area,
                "es_frame": es_frame,
                "es_area": es_area,
                "swapped": False,
            }

        swapped = es_area > ed_area
        if swapped:
            ed_frame, es_frame = es_frame, ed_frame
            ed_area, es_area = es_area, ed_area

        return {
            "ed_frame": ed_frame,
            "ed_area": ed_area,
            "es_frame": es_frame,
            "es_area": es_area,
            "swapped": swapped,
        }

    @staticmethod
    def compute_ef_from_areas(ed_area, es_area):
        """
        Compute EF proxy from area.
        EF = (ED - ES) / ED

        The computation is made physiologically safe by treating the larger area
        as ED and the smaller area as ES, then clamping to [0, 1].
        """
        ed_area = float(ed_area)
        es_area = float(es_area)
        if not np.isfinite(ed_area) or not np.isfinite(es_area):
            return float("nan")

        ed_area, es_area = max(ed_area, es_area), min(ed_area, es_area)
        if ed_area <= 0:
            return 0.0
        ef = (ed_area - es_area) / ed_area
        return float(np.clip(ef, 0.0, 1.0))

    @staticmethod
    def detect_ed_es_from_size_curve(frame_ids, areas, smooth_window=11, enforce_es_after_ed=True):
        """Detect ED/ES on a full-video LV size curve using the same largest-to-smallest drop logic as EchoNet-style pipelines."""
        detected = detect_ed_es_from_area_curve(
            frame_ids=np.asarray(frame_ids, dtype=np.int32),
            areas=np.asarray(areas, dtype=np.float64),
            smooth_window=int(smooth_window),
            enforce_es_after_ed=bool(enforce_es_after_ed),
        )
        return detected

    @staticmethod
    def overlay_mask(frame_bgr, mask, color=(0, 255, 0), alpha=0.35):
        """Overlay binary mask on image for visualization."""
        overlay = frame_bgr.copy()
        overlay[mask > 0] = color
        blended = cv2.addWeighted(overlay, alpha, frame_bgr, 1 - alpha, 0)
        return blended
