import torch
import torch.nn as nn

class SEBlock(nn.Module):
    """
    A Squeeze-and-Excitation block that adaptively recalibrates channel-wise
    feature responses.

    Args:
        channels (int): The number of input channels.
        reduction_ratio (int): The reduction ratio for the bottleneck layer.
    """
    def __init__(self, channels: int, reduction_ratio: int = 16):
        super().__init__()
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.excitation = nn.Sequential(
            nn.Linear(channels, channels // reduction_ratio, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction_ratio, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.shape
        y = self.squeeze(x).view(b, c)
        y = self.excitation(y).view(b, c, 1, 1)
        return x * y.expand_as(x)