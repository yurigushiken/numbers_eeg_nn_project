import torch
import torch.nn as nn
from typing import List

class MultiScaleCNN(nn.Module):
    """
    An EEGNet-like architecture with a multi-scale temporal convolution block.

    This model applies multiple parallel 2D convolutions with different kernel
    sizes to the input EEG signal, capturing features at various time scales.
    The outputs are concatenated and processed through subsequent depthwise and
    separable convolutions.

    Args:
        num_classes (int): The number of classes to predict.
        num_electrodes (int): The number of EEG channels.
        chunk_size (int): The number of time points in each chunk.
        kernel_sizes (List[int]): A list of integers specifying the kernel
                                  sizes for the parallel temporal convolutions.
        F1 (int): The number of output filters for each temporal convolution path.
        D (int): The depth multiplier for the depthwise convolution.
        F2 (int): The number of output filters for the separable convolution.
        kernel_2 (int): The kernel size for the separable convolution.
        dropout (float): The dropout rate.
    """
    def __init__(self, num_classes: int, num_electrodes: int, chunk_size: int,
                 kernel_sizes: List[int] = [7, 11, 15, 19],
                 F1: int = 8, D: int = 2, F2: int = 16, kernel_2: int = 16,
                 dropout: float = 0.5):
        super().__init__()

        # --- 1. Multi-scale temporal convolution block ---
        self.parallel_convs = nn.ModuleList()
        for kernel_size in kernel_sizes:
            self.parallel_convs.append(
                nn.Sequential(
                    nn.Conv2d(1, F1, kernel_size=(1, kernel_size), padding='same', bias=False),
                    nn.BatchNorm2d(F1)
                )
            )

        total_F1 = F1 * len(kernel_sizes)

        # --- 2. Depthwise spatial convolution ---
        self.depthwise_conv = nn.Sequential(
            nn.Conv2d(total_F1, total_F1 * D, kernel_size=(num_electrodes, 1), groups=total_F1, bias=False),
            nn.BatchNorm2d(total_F1 * D),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=(1, 4)),
            nn.Dropout(p=dropout)
        )

        # --- 3. Separable convolution block ---
        self.separable_conv = nn.Sequential(
            nn.Conv2d(total_F1 * D, F2, kernel_size=(1, kernel_2), padding='same', bias=False),
            nn.BatchNorm2d(F2),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=(1, 8)),
            nn.Dropout(p=dropout)
        )

        # --- 4. Classifier ---
        final_feature_map_size = chunk_size // 32
        self.classifier = nn.Linear(F2 * final_feature_map_size, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch, 1, num_electrodes, chunk_size)

        # Apply parallel convolutions and concatenate along the channel dimension
        parallel_outputs = [conv(x) for conv in self.parallel_convs]
        x = torch.cat(parallel_outputs, dim=1)

        x = self.depthwise_conv(x)
        x = self.separable_conv(x)

        # Flatten for the classifier
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x 