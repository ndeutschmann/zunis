"""Trainable layers
"""
import torch

class OverallAffineLayer(torch.nn.Module):
    """Learnable overall affine transformation
    f(x) = alpha x + delta
    """

    def __init__(self, alpha=10., delta=0.):
        super().__init__()
        self.alpha = torch.nn.Parameter(torch.tensor(alpha), requires_grad=True)
        self.delta = torch.nn.Parameter(torch.tensor(delta), requires_grad=True)

    def forward(self, input):
        return input * self.alpha + self.delta