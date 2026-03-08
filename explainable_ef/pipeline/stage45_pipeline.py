import cv2
import numpy as np


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
    def compute_ef_from_areas(ed_area, es_area):
        """
        Compute EF proxy from area.
        EF = (ED - ES) / ED
        """
        if ed_area <= 0:
            return 0.0
        return float((ed_area - es_area) / ed_area)

    @staticmethod
    def overlay_mask(frame_bgr, mask, color=(0, 255, 0), alpha=0.35):
        """Overlay binary mask on image for visualization."""
        overlay = frame_bgr.copy()
        overlay[mask > 0] = color
        blended = cv2.addWeighted(overlay, alpha, frame_bgr, 1 - alpha, 0)
        return blended
