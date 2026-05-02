import torch.nn as nn

from pipeline.stage1_feature_extractor import Stage1FeatureExtractor
from pipeline.stage2_temporal_model import Stage2TemporalModel
from pipeline.stage3_phase_detector import Stage3PhaseDetector
from pipeline.stage45_pipeline import Stage45Pipeline


class EchoPipeline(nn.Module):
    """Orchestrates Stage 1-5 and EF regression head."""

    def __init__(self, num_frames=32, feature_dim=512, preserve_temporal_stride=True):
        super().__init__()
        self.stage1 = Stage1FeatureExtractor(preserve_temporal_stride=preserve_temporal_stride)
        self.stage2 = Stage2TemporalModel(num_frames=num_frames, feature_dim=feature_dim)
        self.stage3 = Stage3PhaseDetector(feature_dim=feature_dim, num_classes=3)
        self.stage45 = Stage45Pipeline()
        self.ef_regressor = nn.Linear(feature_dim, 1)

    def forward(self, x, stage45_context=None, return_stage_outputs=False):
        """
        Standard training/inference path returns (ef, attention, phase_logits).
        If return_stage_outputs=True, also returns a dict with intermediate stage outputs.
        """
        stage1_features = self.stage1(x)
        temporal_features, pooled_features, attention = self.stage2(stage1_features)
        phase_logits = self.stage3(temporal_features)
        pred_ed_idx, pred_es_idx = self.stage3.predict_indices(phase_logits)

        ef = self.ef_regressor(pooled_features).squeeze(1)

        if not return_stage_outputs:
            return ef, attention, phase_logits

        stage_outputs = {
            "stage1_features": stage1_features,
            "stage2_temporal_features": temporal_features,
            "stage2_attention": attention,
            "stage3_phase_logits": phase_logits,
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

        return ef, attention, phase_logits, stage_outputs

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
            }

        frame_masks = {}
        frame_areas = []

        for frame_id in frame_ids:
            frame_rows = video_tracings[video_tracings["Frame"] == frame_id]
            mask = self.stage45.tracing_to_mask(frame_rows, height=frame_height, width=frame_width)
            area = self.stage45.mask_area(mask)
            frame_masks[int(frame_id)] = mask
            frame_areas.append((int(frame_id), area))

        if ed_frame is None or es_frame is None:
            ed_frame, ed_area = max(frame_areas, key=lambda x: x[1])
            es_frame, es_area = min(frame_areas, key=lambda x: x[1])
        else:
            ed_frame = int(ed_frame)
            es_frame = int(es_frame)

            if ed_frame not in frame_masks:
                ed_rows = video_tracings[video_tracings["Frame"] == ed_frame]
                frame_masks[ed_frame] = self.stage45.tracing_to_mask(ed_rows, height=frame_height, width=frame_width)
            if es_frame not in frame_masks:
                es_rows = video_tracings[video_tracings["Frame"] == es_frame]
                frame_masks[es_frame] = self.stage45.tracing_to_mask(es_rows, height=frame_height, width=frame_width)

            ed_area = self.stage45.mask_area(frame_masks[ed_frame])
            es_area = self.stage45.mask_area(frame_masks[es_frame])

        ef_from_masks = self.stage45.compute_ef_from_areas(ed_area, es_area)

        return {
            "ed_frame": int(ed_frame),
            "es_frame": int(es_frame),
            "ed_area": float(ed_area),
            "es_area": float(es_area),
            "ef_from_masks": float(ef_from_masks),
            "ed_mask": frame_masks.get(int(ed_frame)),
            "es_mask": frame_masks.get(int(es_frame)),
        }
