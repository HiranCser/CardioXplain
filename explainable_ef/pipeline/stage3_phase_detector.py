import torch
import torch.nn as nn
import torch.nn.functional as F


class Stage3PhaseDetector(nn.Module):
    """
    Stage 3: predict ED/ES frame scores plus a cyclic phase embedding.

    Output layout is (B, T, 5):
    - channels 0..2: background, ED, ES logits
    - channels 3..4: unit-normalized sin/cos cardiac phase embedding
    """

    def __init__(self, feature_dim=512, dropout=0.1, hidden_dim=256):
        super().__init__()

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

        self.phase_logits = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 3),
        )
        self.phase_vector = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 2),
        )

    def forward(self, temporal_features):
        # temporal_features: (B, T, C)
        x = temporal_features.transpose(1, 2)

        feat3 = self.temporal_conv3(x)
        feat5 = self.temporal_conv5(x)
        feat = torch.cat([feat3, feat5], dim=1)
        feat = self.temporal_dropout(feat)

        feat = feat.transpose(1, 2)
        feat, _ = self.temporal_gru(feat)

        logits = self.phase_logits(feat)
        phase_vec = F.normalize(self.phase_vector(feat), p=2, dim=-1, eps=1e-6)
        return torch.cat([logits, phase_vec], dim=-1)

    @staticmethod
    def _smooth_scores(scores, kernel_size=5):
        """Apply temporal moving-average smoothing."""
        kernel_size = int(max(1, kernel_size))
        if kernel_size <= 1:
            return scores
        if kernel_size % 2 == 0:
            kernel_size += 1

        pad = kernel_size // 2
        return F.avg_pool1d(scores.unsqueeze(1), kernel_size=kernel_size, stride=1, padding=pad).squeeze(1)

    @staticmethod
    def predict_indices(
        phase_predictions,
        min_gap=15,
        max_gap_ratio=0.65,
        smooth_kernel=5,
    ):
        """
        Extract ED and ES frame indices.

        Supports current tensors shaped (B, T, C>=3), where channels 1 and 2 are
        ED/ES logits, and legacy scalar continuous phase tensors shaped (B, T).
        """
        if phase_predictions.ndim == 3 and phase_predictions.shape[-1] >= 3:
            scores = phase_predictions[:, :, :3].float()
            ed_scores = Stage3PhaseDetector._smooth_scores(scores[:, :, 1], kernel_size=smooth_kernel)
            es_scores = Stage3PhaseDetector._smooth_scores(scores[:, :, 2], kernel_size=smooth_kernel)
            pred_ed = torch.argmax(ed_scores, dim=1)
            pred_es = torch.argmax(es_scores, dim=1)
            batch_size, num_frames = ed_scores.shape
            dist_to_es = None
        elif phase_predictions.ndim == 2:
            phase_pred = Stage3PhaseDetector._smooth_scores(phase_predictions.float(), kernel_size=smooth_kernel)
            batch_size, num_frames = phase_pred.shape
            if num_frames <= 1:
                pred_ed = torch.zeros(batch_size, dtype=torch.long, device=phase_pred.device)
                pred_es = torch.zeros(batch_size, dtype=torch.long, device=phase_pred.device)
                return pred_ed, pred_es

            dist_to_ed = torch.min(torch.abs(phase_pred), torch.abs(phase_pred - 1.0))
            dist_to_es = torch.abs(phase_pred - 0.5)
            pred_ed = torch.argmin(dist_to_ed, dim=1)
            pred_es = torch.argmin(dist_to_es, dim=1)
            ed_scores = None
            es_scores = None
        else:
            raise ValueError("phase_predictions must have shape (B, T) or (B, T, C)")

        if num_frames <= 1:
            device = phase_predictions.device
            pred_ed = torch.zeros(batch_size, dtype=torch.long, device=device)
            pred_es = torch.zeros(batch_size, dtype=torch.long, device=device)
            return pred_ed, pred_es

        min_gap = int(max(1, min(min_gap, num_frames - 1)))
        max_gap = int(round(float(max_gap_ratio) * num_frames)) if max_gap_ratio is not None else (num_frames - 1)
        max_gap = int(max(min_gap, min(num_frames - 1, max_gap)))

        for b in range(batch_size):
            ed_i = int(pred_ed[b].item())
            es_i = int(pred_es[b].item())
            gap = abs(es_i - ed_i)

            if gap < min_gap or gap > max_gap:
                valid_start = min(num_frames - 1, ed_i + min_gap)
                valid_end = min(num_frames - 1, ed_i + max_gap)

                if valid_end >= valid_start:
                    if es_scores is not None:
                        local_es = es_scores[b, valid_start : valid_end + 1]
                        pred_es[b] = valid_start + torch.argmax(local_es)
                    else:
                        local_es = dist_to_es[b, valid_start : valid_end + 1]
                        pred_es[b] = valid_start + torch.argmin(local_es)
                elif es_i == ed_i:
                    pred_es[b] = min(num_frames - 1, ed_i + 1)

        return pred_ed, pred_es
