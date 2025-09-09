import math
import torch
import torch.nn as nn


def softplus_inv(y: torch.Tensor) -> torch.Tensor:
    """Numerically stable inverse of Softplus.

    For y > 0, softplus_inv(y) satisfies softplus(softplus_inv(y)) == y.
    """
    y = torch.clamp(y, min=1e-8)
    return torch.log(torch.expm1(y))


class ChannelGate(nn.Module):
    """Learnable non-negative per-channel gates.

    Supports input of shape (B, 1, C, T) or (B, C, T) and multiplies
    each channel by a non-negative scalar.
    """

    def __init__(self, num_channels: int, init: float = 1.0):
        super().__init__()
        init_t = torch.full((num_channels,), float(init))
        self.log_g = nn.Parameter(softplus_inv(init_t))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gates = torch.nn.functional.softplus(self.log_g)
        if x.dim() == 4:  # (B, 1, C, T)
            return x * gates.view(1, 1, -1, 1)
        if x.dim() == 3:  # (B, C, T)
            return x * gates.view(1, -1, 1)
        raise ValueError(f"Unsupported input shape for ChannelGate: {tuple(x.shape)}")

    def l1_penalty(self) -> torch.Tensor:
        return torch.nn.functional.softplus(self.log_g).sum()


class ChannelGatedModel(nn.Module):
    """Wraps a backbone model and applies ChannelGate before it.

    The wrapper exposes two helpers for training and reporting:
    - gate_l1_penalty(): L1 penalty on the current gate values
    - get_gate_values(): detached tensor of current gate values
    """

    def __init__(self, backbone: nn.Module, num_channels: int, gate_init: float = 1.0):
        super().__init__()
        self.gate = ChannelGate(num_channels=num_channels, init=gate_init)
        self.backbone = backbone

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.gate(x)
        return self.backbone(x)

    def gate_l1_penalty(self) -> torch.Tensor:
        return self.gate.l1_penalty()

    def get_gate_values(self) -> torch.Tensor:
        return torch.nn.functional.softplus(self.gate.log_g).detach()


