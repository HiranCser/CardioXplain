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
    def mask_quality(mask, min_area_fraction=0.001, max_area_fraction=0.60):
        """Return lightweight anatomical plausibility checks for a binary LV mask."""
        m = (np.asarray(mask) > 0).astype(np.uint8)
        if m.size == 0:
            return {
                "valid": False,
                "area": 0.0,
                "area_fraction": 0.0,
                "component_count": 0,
                "largest_component_fraction": 0.0,
                "issues": ["empty_mask"],
            }

        area = float(m.sum())
        area_fraction = float(area / max(1, m.size))
        issues = []
        if area <= 0:
            issues.append("empty_mask")
        if area_fraction < float(min_area_fraction):
            issues.append("mask_area_too_small")
        if area_fraction > float(max_area_fraction):
            issues.append("mask_area_too_large")

        component_count = 0
        largest_component_fraction = 0.0
        if area > 0:
            n_labels, _, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
            component_count = max(0, int(n_labels) - 1)
            if component_count > 0:
                largest_area = float(np.max(stats[1:, cv2.CC_STAT_AREA]))
                largest_component_fraction = float(largest_area / max(1.0, area))
                if largest_component_fraction < 0.80:
                    issues.append("fragmented_mask")

        return {
            "valid": len(issues) == 0,
            "area": area,
            "area_fraction": area_fraction,
            "component_count": component_count,
            "largest_component_fraction": largest_component_fraction,
            "issues": issues,
        }

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
    def validate_ed_es_quality(
        frame_ids,
        ed_frame,
        ed_area,
        es_frame,
        es_area,
        ed_mask=None,
        es_mask=None,
        min_gap=1,
        max_gap_ratio=0.85,
        min_area_delta_fraction=0.03,
        min_ef=0.03,
        max_ef=0.95,
        min_mask_area_fraction=0.001,
        max_mask_area_fraction=0.60,
    ):
        """Validate the ED/ES handoff before Stage 5 EF computation."""
        frame_ids = np.asarray(frame_ids, dtype=np.int32)
        num_frames = int(frame_ids.size)
        issues = []

        try:
            ed_frame = int(ed_frame)
            es_frame = int(es_frame)
            ed_area = float(ed_area)
            es_area = float(es_area)
        except (TypeError, ValueError):
            return {"valid": False, "issues": ["invalid_ed_es_values"]}

        if num_frames <= 0:
            issues.append("empty_frame_curve")
        else:
            frame_set = set(frame_ids.tolist())
            if ed_frame not in frame_set:
                issues.append("ed_frame_missing_from_curve")
            if es_frame not in frame_set:
                issues.append("es_frame_missing_from_curve")

        if not np.isfinite(ed_area) or not np.isfinite(es_area):
            issues.append("nonfinite_area")
        if ed_area <= 0:
            issues.append("ed_area_nonpositive")
        if es_area < 0:
            issues.append("es_area_negative")
        if ed_frame == es_frame:
            issues.append("ed_es_same_frame")
        if np.isfinite(ed_area) and np.isfinite(es_area) and es_area > ed_area:
            issues.append("es_area_exceeds_ed_area")

        if num_frames > 1:
            gap = abs(ed_frame - es_frame)
            min_gap = int(max(1, min(int(min_gap), num_frames - 1)))
            max_gap = int(round(float(max_gap_ratio) * num_frames))
            max_gap = int(max(min_gap, min(num_frames - 1, max_gap)))
            if gap < min_gap:
                issues.append("ed_es_gap_too_small")
            if gap > max_gap:
                issues.append("ed_es_gap_too_large")

        if np.isfinite(ed_area) and np.isfinite(es_area) and ed_area > 0:
            area_delta_fraction = float((ed_area - es_area) / ed_area)
            if area_delta_fraction < float(min_area_delta_fraction):
                issues.append("area_delta_too_small")
            ef = Stage45Pipeline.compute_ef_from_areas(ed_area, es_area)
            if not np.isfinite(ef):
                issues.append("ef_nonfinite")
            elif ef < float(min_ef) or ef > float(max_ef):
                issues.append("ef_out_of_physiologic_range")
        else:
            area_delta_fraction = float("nan")
            ef = float("nan")

        mask_reports = {}
        if ed_mask is not None:
            mask_reports["ed_mask"] = Stage45Pipeline.mask_quality(
                ed_mask,
                min_area_fraction=min_mask_area_fraction,
                max_area_fraction=max_mask_area_fraction,
            )
            issues.extend([f"ed_{issue}" for issue in mask_reports["ed_mask"]["issues"]])
        if es_mask is not None:
            mask_reports["es_mask"] = Stage45Pipeline.mask_quality(
                es_mask,
                min_area_fraction=min_mask_area_fraction,
                max_area_fraction=max_mask_area_fraction,
            )
            issues.extend([f"es_{issue}" for issue in mask_reports["es_mask"]["issues"]])

        return {
            "valid": len(issues) == 0,
            "issues": issues,
            "ef": float(ef) if np.isfinite(ef) else float("nan"),
            "area_delta_fraction": float(area_delta_fraction) if np.isfinite(area_delta_fraction) else float("nan"),
            "num_frames": num_frames,
            "mask_reports": mask_reports,
        }

    @staticmethod
    def _area_for_frame(frame_ids, areas, frame_id):
        frame_ids = np.asarray(frame_ids, dtype=np.int32)
        areas = np.asarray(areas, dtype=np.float64)
        matches = np.where(frame_ids == int(frame_id))[0]
        if matches.size > 0:
            return float(areas[int(matches[0])])
        if areas.size == 0:
            return float("nan")
        nearest = int(np.argmin(np.abs(frame_ids - int(frame_id))))
        return float(areas[nearest])

    @staticmethod
    def select_robust_ed_es_pair(
        frame_ids,
        areas,
        candidate_ed_frame=None,
        candidate_es_frame=None,
        smooth_window=11,
        enforce_es_after_ed=True,
        **quality_kwargs,
    ):
        """
        Select ED/ES with validation and fallbacks.

        Order of attempts:
        1. caller-provided candidate pair, if present
        2. smoothed full-curve ED/ES detector
        3. global max/min extrema
        """
        frame_ids = np.asarray(frame_ids, dtype=np.int32)
        areas = np.asarray(areas, dtype=np.float64)
        if frame_ids.size == 0 or areas.size == 0:
            return {
                "ed_frame": -1,
                "es_frame": -1,
                "ed_area": 0.0,
                "es_area": 0.0,
                "ef": 0.0,
                "quality": {"valid": False, "issues": ["empty_frame_curve"]},
                "fallback_used": "empty",
                "attempts": [],
            }

        attempts = []

        def evaluate(name, ed_frame, es_frame):
            ed_area = Stage45Pipeline._area_for_frame(frame_ids, areas, ed_frame)
            es_area = Stage45Pipeline._area_for_frame(frame_ids, areas, es_frame)
            pair = Stage45Pipeline.canonicalize_ed_es_pair(ed_frame, ed_area, es_frame, es_area)
            quality = Stage45Pipeline.validate_ed_es_quality(
                frame_ids=frame_ids,
                ed_frame=pair["ed_frame"],
                ed_area=pair["ed_area"],
                es_frame=pair["es_frame"],
                es_area=pair["es_area"],
                **quality_kwargs,
            )
            result = {
                "name": name,
                "ed_frame": int(pair["ed_frame"]),
                "es_frame": int(pair["es_frame"]),
                "ed_area": float(pair["ed_area"]),
                "es_area": float(pair["es_area"]),
                "swapped": bool(pair["swapped"]),
                "quality": quality,
            }
            attempts.append(result)
            return result

        if candidate_ed_frame is not None and candidate_es_frame is not None:
            candidate = evaluate("candidate", candidate_ed_frame, candidate_es_frame)
            if candidate["quality"]["valid"]:
                candidate["fallback_used"] = "none"
                candidate["attempts"] = attempts
                candidate["ef"] = Stage45Pipeline.compute_ef_from_areas(candidate["ed_area"], candidate["es_area"])
                return candidate

        try:
            detected = Stage45Pipeline.detect_ed_es_from_size_curve(
                frame_ids=frame_ids,
                areas=areas,
                smooth_window=int(smooth_window),
                enforce_es_after_ed=bool(enforce_es_after_ed),
            )
            curve = evaluate("smoothed_curve", detected["ed_frame"], detected["es_frame"])
            if curve["quality"]["valid"]:
                curve["fallback_used"] = "smoothed_curve"
                curve["attempts"] = attempts
                curve["ef"] = Stage45Pipeline.compute_ef_from_areas(curve["ed_area"], curve["es_area"])
                return curve
        except Exception as exc:
            attempts.append({"name": "smoothed_curve", "error": str(exc), "quality": {"valid": False, "issues": ["curve_detection_failed"]}})

        finite = np.isfinite(areas)
        if not np.any(finite):
            extrema = {
                "ed_frame": int(frame_ids[0]),
                "es_frame": int(frame_ids[0]),
                "ed_area": 0.0,
                "es_area": 0.0,
                "quality": {"valid": False, "issues": ["nonfinite_area_curve"]},
                "swapped": False,
            }
        else:
            valid_ids = frame_ids[finite]
            valid_areas = areas[finite]
            extrema = evaluate("global_extrema", valid_ids[int(np.argmax(valid_areas))], valid_ids[int(np.argmin(valid_areas))])

        extrema["fallback_used"] = "global_extrema"
        extrema["attempts"] = attempts
        extrema["ef"] = Stage45Pipeline.compute_ef_from_areas(extrema["ed_area"], extrema["es_area"])
        return extrema

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
