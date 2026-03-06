import cv2
import numpy as np


class Stage45Pipeline:
    """Stage 4/5 utilities: LV mask creation, area extraction, and EF computation."""

    @staticmethod
    def tracing_to_contour(frame_rows):
        """
        Build a closed LV contour from tracing pairs.
        X1/Y1 are ordered along one wall, X2/Y2 along the opposite wall.
        """
        left_wall = frame_rows[["X1", "Y1"]].to_numpy(dtype=np.float32)
        right_wall = frame_rows[["X2", "Y2"]].to_numpy(dtype=np.float32)[::-1]
        contour = np.concatenate([left_wall, right_wall], axis=0)
        return contour

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
