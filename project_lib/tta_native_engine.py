# code/tta_native_engine.py
"""
A self-contained, native implementation of the Test-Time Adaptation (TTA)
algorithm based on entropy minimization (T-TIME).

This module has no dependencies on the third_party/DeepTransferEEG repository.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class HLoss(nn.Module):
    """
    Entropy minimization loss.

    Calculates the entropy of the model's output probabilities and encourages
    the model to make more confident (lower entropy) predictions.
    """
    def __init__(self):
        super(HLoss, self).__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Apply softmax to get probabilities, then compute entropy
        b = F.softmax(x, dim=1) * F.log_softmax(x, dim=1)
        b = -1.0 * b.sum(dim=1)
        return b.mean()

def adapt_model_on_batch(model: nn.Module,
                         optimizer: torch.optim.Optimizer,
                         loss_fn: nn.Module,
                         data_batch: torch.Tensor,
                         steps: int):
    """
    Performs one adaptation cycle on a single batch of unlabeled test data.

    Args:
        model: The PyTorch model to adapt.
        optimizer: The optimizer for TTA.
        loss_fn: The entropy minimization loss function.
        data_batch: A tensor of unlabeled data for adaptation.
        steps (int): The number of gradient updates to perform.
    """
    for _ in range(steps):
        optimizer.zero_grad()
        outputs = model(data_batch)
        loss = loss_fn(outputs)
        loss.backward()
        optimizer.step()