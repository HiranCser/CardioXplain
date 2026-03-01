import torch
import torch.nn as nn
import torchvision.models.video as models

class EFModel(nn.Module):
    def __init__(self):
        super().__init__()

        # Load pretrained model - handle both old and new torchvision API
        try:
            self.backbone = models.r2plus1d_18(weights="DEFAULT")
        except TypeError:
            # Fallback for older torchvision versions
            self.backbone = models.r2plus1d_18(pretrained=True)
        
        self.backbone.fc = nn.Identity()

        self.regressor = nn.Linear(512, 1)

    def forward(self, x):
        features = self.backbone(x)
        ef = self.regressor(features)
        return ef.squeeze(1)