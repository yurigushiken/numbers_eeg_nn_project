import torch
import torch.nn as nn
import torch.nn.functional as F


def softplus_inv(y: torch.Tensor) -> torch.Tensor:
    y = torch.clamp(y, min=1e-8)
    return torch.log(torch.expm1(y))


class TemporalGate(nn.Module):
    """Learnable non-negative per-timepoint gates.

    Supports input of shape (B, 1, C, T) or (B, C, T) and multiplies along T.
    """

    def __init__(self, num_timepoints: int, init: float = 1.0):
        super().__init__()
        init_t = torch.full((num_timepoints,), float(init))
        self.log_g_time = nn.Parameter(softplus_inv(init_t))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        g = F.softplus(self.log_g_time)
        if x.dim() == 4:  # (B, 1, C, T)
            return x * g.view(1, 1, 1, -1)
        if x.dim() == 3:  # (B, C, T)
            return x * g.view(1, 1, -1)
        raise ValueError(f"Unsupported input shape for TemporalGate: {tuple(x.shape)}")

    def l1_penalty(self) -> torch.Tensor:
        return F.softplus(self.log_g_time).sum()

    def tv_penalty(self) -> torch.Tensor:
        g = F.softplus(self.log_g_time)
        if g.shape[0] <= 1:
            return g.sum() * 0.0
        return (g[1:] - g[:-1]).abs().sum()


class TemporalGatedModel(nn.Module):
    """Wraps a backbone model and applies TemporalGate before it.

    Exposes:
      - time_gate_l1_penalty()
      - time_gate_tv_penalty()
      - get_time_gate_values()
    Also passes through channel-gate helpers if the backbone exposes them.
    """

    def __init__(self, backbone: nn.Module, num_timepoints: int, gate_init: float = 1.0):
        super().__init__()
        self.time_gate = TemporalGate(num_timepoints=num_timepoints, init=gate_init)
        self.backbone = backbone

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(self.time_gate(x))

    # Temporal helpers
    def time_gate_l1_penalty(self) -> torch.Tensor:
        return self.time_gate.l1_penalty()

    def time_gate_tv_penalty(self) -> torch.Tensor:
        return self.time_gate.tv_penalty()

    def get_time_gate_values(self) -> torch.Tensor:
        return F.softplus(self.time_gate.log_g_time).detach()

    # Channel-gate passthroughs (if backbone has them)
    def gate_l1_penalty(self) -> torch.Tensor:
        if hasattr(self.backbone, "gate_l1_penalty"):
            return getattr(self.backbone, "gate_l1_penalty")()
        return torch.tensor(0.0, device=self.time_gate.log_g_time.device)

    def get_gate_values(self) -> torch.Tensor:
        if hasattr(self.backbone, "get_gate_values"):
            return getattr(self.backbone, "get_gate_values")()
        return torch.empty(0, device=self.time_gate.log_g_time.device)


