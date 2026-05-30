import torch.nn as nn

from pipeline.stage1_feature_extractor import Stage1FeatureExtractor
from pipeline.stage2_temporal_model import Stage2TemporalModel
from pipeline.stage3_phase_detector import Stage3PhaseDetector
from pipeline.stage45_pipeline import Stage45Pipeline


class EchoPipeline(nn.Module):
    """Orchestrates Stage 1-5 and EF regression head."""

    def __init__(self, num_frames=32, feature_dim=512, ef_head_arch="mlp"):
        super().__init__()
        self.stage1 = Stage1FeatureExtractor()
        self.stage2 = Stage2TemporalModel(num_frames=num_frames, feature_dim=feature_dim)
        self.stage3 = Stage3PhaseDetector(feature_dim=feature_dim)
        self.stage45 = Stage45Pipeline()
        self.ef_head_arch = str(ef_head_arch).strip().lower()
        self.legacy_ef_dim = feature_dim * 2
        if self.ef_head_arch == "legacy_linear":
            self.ef_regressor = nn.Linear(self.legacy_ef_dim, 1)
        elif self.ef_head_arch == "mlp":
            self.ef_regressor = nn.Sequential(
                nn.LayerNorm(self.stage2.output_dim),
                nn.Linear(self.stage2.output_dim, 512),
                nn.GELU(),
                nn.Dropout(0.2),
                nn.Linear(512, 128),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(128, 1),
            )
        else:
            raise ValueError(f"Unsupported EF head architecture: {ef_head_arch}")

    def _ef_features(self, pooled_features):
        if self.ef_head_arch == "legacy_linear":
            return pooled_features[:, : self.legacy_ef_dim]
        return pooled_features

    def forward(self, x, stage45_context=None, return_stage_outputs=False):
        """
        Standard training/inference path returns (ef, attention, phase_pred).
        If return_stage_outputs=True, also returns a dict with intermediate stage outputs.
        """
        stage1_features = self.stage1(x)
        temporal_features, pooled_features, attention = self.stage2(stage1_features)
        phase_pred = self.stage3(temporal_features)
        pred_ed_idx, pred_es_idx = self.stage3.predict_indices(phase_pred)

        ef = self.ef_regressor(self._ef_features(pooled_features)).squeeze(1)

        if not return_stage_outputs:
            return ef, attention, phase_pred

        stage_outputs = {
            "stage1_features": stage1_features,
            "stage2_temporal_features": temporal_features,
            "stage2_attention": attention,
            "stage3_phase_pred": phase_pred,
            "stage3_pred_ed_idx": pred_ed_idx,
            "stage3_pred_es_idx": pred_es_idx,
        }

        if stage45_context is not None:
            stage_outputs["stage45"] = self.run_stage45_from_tracings(
                video_tracings=stage45_context["video_tracings"],
                frame_height=stage45_context["frame_height"],
                frame_width=stage45_context["frame_width"],
                ed_frame=stage45_context.get("ed_frame"),
                es_frame=stage45_context.get("es_frame"),
            )

        return ef, attention, phase_pred, stage_outputs

    def run_stage45_from_tracings(self, video_tracings, frame_height, frame_width, ed_frame=None, es_frame=None):
        """
        Stage 4/5 execution from tracing rows.
        If ED/ES frames are not provided, they are inferred from max/min cavity area.
        """
        frame_ids = sorted(video_tracings["Frame"].unique().tolist())
        if len(frame_ids) == 0:
            return {
                "ed_frame": -1,
                "es_frame": -1,
                "ed_area": 0.0,
                "es_area": 0.0,
                "ef_from_masks": 0.0,
                "ed_mask": None,
                "es_mask": None,
                "quality": {"valid": False, "issues": ["empty_frame_curve"]},
                "fallback_used": "empty",
            }

        frame_masks = {}
        frame_areas = []

        for frame_id in frame_ids:
            frame_rows = video_tracings[video_tracings["Frame"] == frame_id]
            mask = self.stage45.tracing_to_mask(frame_rows, height=frame_height, width=frame_width)
            area = self.stage45.mask_area(mask)
            frame_masks[int(frame_id)] = mask
            frame_areas.append((int(frame_id), area))

        frame_ids_np = [fid for fid, _ in frame_areas]
        areas_np = [area for _, area in frame_areas]
        robust_pair = self.stage45.select_robust_ed_es_pair(
            frame_ids=frame_ids_np,
            areas=areas_np,
            candidate_ed_frame=ed_frame,
            candidate_es_frame=es_frame,
            smooth_window=11,
            enforce_es_after_ed=True,
        )

        ed_frame = int(robust_pair["ed_frame"])
        es_frame = int(robust_pair["es_frame"])
        ed_area = float(robust_pair["ed_area"])
        es_area = float(robust_pair["es_area"])
        ef_from_masks = self.stage45.compute_ef_from_areas(ed_area, es_area)

        quality = self.stage45.validate_ed_es_quality(
            frame_ids=frame_ids_np,
            ed_frame=ed_frame,
            ed_area=ed_area,
            es_frame=es_frame,
            es_area=es_area,
            ed_mask=frame_masks.get(ed_frame),
            es_mask=frame_masks.get(es_frame),
        )

        return {
            "ed_frame": int(ed_frame),
            "es_frame": int(es_frame),
            "ed_area": float(ed_area),
            "es_area": float(es_area),
            "ef_from_masks": float(ef_from_masks),
            "ed_mask": frame_masks.get(int(ed_frame)),
            "es_mask": frame_masks.get(int(es_frame)),
            "quality": quality,
            "fallback_used": robust_pair.get("fallback_used", "none"),
            "fallback_attempts": robust_pair.get("attempts", []),
        }
