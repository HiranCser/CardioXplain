import torch
import torch.nn as nn
import torch.nn.functional as F


class Stage3PhaseDetector(nn.Module):
    """Stage 3: per-frame phase classification and ED/ES index extraction."""

    def __init__(self, feature_dim=512, num_classes=3, dropout=0.1, hidden_dim=256):
        super().__init__()

        # Multi-scale temporal context before recurrent modeling.
        self.temporal_conv3 = nn.Sequential(
            nn.Conv1d(feature_dim, hidden_dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
        )
        self.temporal_conv5 = nn.Sequential(
            nn.Conv1d(feature_dim, hidden_dim, kernel_size=5, padding=2, bias=False),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
        )

        self.temporal_dropout = nn.Dropout(dropout)
        self.temporal_gru = nn.GRU(
            input_size=hidden_dim * 2,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )

        self.phase_classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, temporal_features):
        # temporal_features: (B, T, C)
        x = temporal_features.transpose(1, 2)  # (B, C, T)

        feat3 = self.temporal_conv3(x)
        feat5 = self.temporal_conv5(x)
        feat = torch.cat([feat3, feat5], dim=1)
        feat = self.temporal_dropout(feat)

        feat = feat.transpose(1, 2)  # (B, T, 2H)
        feat, _ = self.temporal_gru(feat)

        # (B, T, 2H) -> (B, T, 3)
        return self.phase_classifier(feat)

    @staticmethod
    def _smooth_scores(scores, kernel_size=5):
        kernel_size = int(max(1, kernel_size))
        if kernel_size <= 1:
            return scores
        if kernel_size % 2 == 0:
            kernel_size += 1

        pad = kernel_size // 2
        return F.avg_pool1d(scores.unsqueeze(1), kernel_size=kernel_size, stride=1, padding=pad).squeeze(1)

    @staticmethod
    def predict_indices(
        phase_logits,
        min_gap=2,
        max_gap_ratio=0.65,
        smooth_kernel=5,
    ):
        """
        Predict ED/ES indices using constrained pair decoding.

        Unlike independent argmax, this decodes ED and ES jointly with physiology-inspired
        constraints: ES should occur after ED with a plausible frame gap.
        """
        if phase_logits.ndim != 3 or phase_logits.shape[-1] < 3:
            raise ValueError("phase_logits must have shape (B, T, >=3)")

        # Relative phase confidence against background channel.
        ed_scores = phase_logits[:, :, 1] - phase_logits[:, :, 0]
        es_scores = phase_logits[:, :, 2] - phase_logits[:, :, 0]

        ed_scores = Stage3PhaseDetector._smooth_scores(ed_scores, kernel_size=smooth_kernel)
        es_scores = Stage3PhaseDetector._smooth_scores(es_scores, kernel_size=smooth_kernel)

        batch_size, num_frames = ed_scores.shape
        if num_frames <= 1:
            pred_ed = torch.argmax(ed_scores, dim=1)
            pred_es = torch.argmax(es_scores, dim=1)
            return pred_ed, pred_es

        min_gap = int(max(1, min_gap))
        max_gap = int(round(float(max_gap_ratio) * num_frames)) if max_gap_ratio is not None else (num_frames - 1)
        max_gap = int(max(min_gap, min(num_frames - 1, max_gap)))

        pair_scores = ed_scores.unsqueeze(2) + es_scores.unsqueeze(1)  # (B, T_ed, T_es)

        idx = torch.arange(num_frames, device=phase_logits.device)
        gap = idx.unsqueeze(0) - idx.unsqueeze(1)  # (T_ed, T_es): es-ed
        valid = (gap >= min_gap) & (gap <= max_gap)

        neg_inf = torch.finfo(pair_scores.dtype).min
        pair_scores = pair_scores.masked_fill(~valid.unsqueeze(0), neg_inf)

        flat = pair_scores.reshape(batch_size, -1)
        valid_any = torch.any(torch.isfinite(flat), dim=1)

        best_flat = torch.argmax(flat, dim=1)
        pred_ed = best_flat // num_frames
        pred_es = best_flat % num_frames

        # Fallback when constraints leave no valid pair (very short clips).
        fallback_ed = torch.argmax(ed_scores, dim=1)
        fallback_es = torch.argmax(es_scores, dim=1)
        pred_ed = torch.where(valid_any, pred_ed, fallback_ed)
        pred_es = torch.where(valid_any, pred_es, fallback_es)

        # Final guard: enforce ES > ED by selecting the best ES after ED when needed.
        need_fix = pred_es <= pred_ed
        if need_fix.any():
            for b in torch.where(need_fix)[0].tolist():
                ed_i = int(pred_ed[b].item())
                start = min(num_frames - 1, ed_i + min_gap)
                end = min(num_frames - 1, ed_i + max_gap)
                if end >= start:
                    local = es_scores[b, start : end + 1]
                    pred_es[b] = start + int(torch.argmax(local).item())
                else:
                    pred_es[b] = min(num_frames - 1, ed_i + 1)

        return pred_ed, pred_es
