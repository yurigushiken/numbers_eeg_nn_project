import torch.nn as nn
from .se_block import SEBlock

class EEGNetWithSE(nn.Module):
    """
    EEGNet with Squeeze-and-Excitation blocks.

    This architecture integrates SE blocks after the main temporal and spatial
    convolutional blocks of the standard EEGNet to enable feature recalibration.
    """
    def __init__(self, num_classes: int, num_electrodes: int, chunk_size: int,
                 F1=8, D=2, F2=16, kernel_1=64, kernel_2=16, dropout=0.5,
                 se_reduction_ratio=4):
        super().__init__()
        self.F1 = F1
        self.D = D
        self.F2 = F2

        # --- Block 1: Temporal Convolution + SE ---
        self.block1 = nn.Sequential(
            nn.Conv2d(1, F1, (1, kernel_1), padding=(0, kernel_1 // 2), bias=False),
            nn.BatchNorm2d(F1),
            SEBlock(channels=F1, reduction_ratio=se_reduction_ratio)
        )

        # --- Block 2: Depthwise Spatial Convolution + SE ---
        self.block2 = nn.Sequential(
            nn.Conv2d(F1, F1 * D, (num_electrodes, 1), groups=F1, bias=False),
            nn.BatchNorm2d(F1 * D),
            nn.ELU(),
            nn.AvgPool2d((1, 4)),
            nn.Dropout(dropout),
            SEBlock(channels=F1 * D, reduction_ratio=se_reduction_ratio)
        )

        # --- Block 3: Separable Convolution ---
        self.block3 = nn.Sequential(
            nn.Conv2d(F1 * D, F2, (1, kernel_2), padding=(0, kernel_2 // 2), bias=False),
            nn.BatchNorm2d(F2),
            nn.ELU(),
            nn.AvgPool2d((1, 8)),
            nn.Dropout(dropout)
        )

        # --- Classifier ---
        final_feature_map_size = chunk_size // 32
        self.classifier = nn.Linear(F2 * final_feature_map_size, num_classes)

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x