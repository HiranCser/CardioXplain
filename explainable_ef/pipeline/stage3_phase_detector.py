import torch
import torch.nn as nn
import torch.nn.functional as F


class Stage3PhaseDetector(nn.Module):
    """
    Stage 3: Predict continuous cardiac phase [0, 1] for each frame.
    
    Instead of classifying ED/ES directly, we predict a continuous cardiac phase:
    - ED (End-Diastole, maximum LV cavity) → phase ≈ 0.0
    - ES (End-Systole, minimum LV cavity) → phase ≈ 0.5
    - Full cycle: 0.0 → 0.5 → 1.0 (back to ED)
    
    This treats the cardiac cycle as a continuous progression, making it easier
    for the network to learn smooth temporal dynamics.
    """

    def __init__(self, feature_dim=512, dropout=0.1, hidden_dim=256):
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

        # Continuous phase regression head (single output per frame)
        self.phase_regressor = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),  # Output in [0, 1]
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

        # (B, T, 2H) -> (B, T, 1) -> (B, T)
        phase_pred = self.phase_regressor(feat).squeeze(-1)
        return phase_pred

    @staticmethod
    def _smooth_scores(scores, kernel_size=5):
        """Apply temporal smoothing to phase predictions."""
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
        Extract ED and ES frame indices from continuous phase predictions.
        
        ED (phase ≈ 0) is identified as the frame closest to phase 0 or 1.
        ES (phase ≈ 0.5) is identified as the frame closest to phase 0.5.
        
        Args:
            phase_predictions: (B, T) tensor of phase values in [0, 1]
            min_gap: Minimum frame gap required between ED and ES
            max_gap_ratio: Maximum frame gap as ratio of clip length
            smooth_kernel: Kernel size for temporal smoothing
        
        Returns:
            pred_ed, pred_es: (B,) tensors of predicted frame indices
        """
        if phase_predictions.ndim != 2:
            raise ValueError("phase_predictions must have shape (B, T)")

        phase_pred = Stage3PhaseDetector._smooth_scores(phase_predictions, kernel_size=smooth_kernel)
        
        batch_size, num_frames = phase_pred.shape
        if num_frames <= 1:
            pred_ed = torch.zeros(batch_size, dtype=torch.long, device=phase_pred.device)
            pred_es = torch.zeros(batch_size, dtype=torch.long, device=phase_pred.device)
            return pred_ed, pred_es

        # ED: find frame closest to phase 0 (or 1 if that's closer)
        # Distance to ED: min(|phase - 0|, |phase - 1|)
        dist_to_ed = torch.min(torch.abs(phase_pred), torch.abs(phase_pred - 1.0))
        pred_ed = torch.argmin(dist_to_ed, dim=1)
        
        # ES: find frame closest to phase 0.5
        dist_to_es = torch.abs(phase_pred - 0.5)
        pred_es = torch.argmin(dist_to_es, dim=1)

        # Enforce gap constraints
        min_gap = int(max(1, min(min_gap, num_frames - 1)))
        max_gap = int(round(float(max_gap_ratio) * num_frames)) if max_gap_ratio is not None else (num_frames - 1)
        max_gap = int(max(min_gap, min(num_frames - 1, max_gap)))

        # Ensure ES is after ED with proper gap
        for b in range(batch_size):
            ed_i = int(pred_ed[b].item())
            es_i = int(pred_es[b].item())
            gap = abs(es_i - ed_i)
            
            if gap < min_gap or gap > max_gap:
                # Find the best ES within the valid gap range
                valid_start = min(num_frames - 1, ed_i + min_gap)
                valid_end = min(num_frames - 1, ed_i + max_gap)
                
                if valid_end >= valid_start:
                    # Find ES within valid range
                    local_es = dist_to_es[b, valid_start:valid_end + 1]
                    pred_es[b] = valid_start + torch.argmin(local_es)
                else:
                    # Fallback: just ensure ES != ED
                    if es_i == ed_i:
                        pred_es[b] = min(num_frames - 1, ed_i + 1)

        return pred_ed, pred_es
