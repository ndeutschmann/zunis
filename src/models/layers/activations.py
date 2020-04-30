"""Activation functions
Element-wise transformations without trainable parameters
"""
import torch

class BiLU(torch.nn.Module):
    """Bijective linear unit
    f(x) = alpha + delta*sign(x)
    """

    def __init__(self, alpha=1., delta=.9):
        super(BiLU, self).__init__()
        self.alpha = torch.nn.Parameter(torch.tensor(alpha), requires_grad=False)
        self.delta = torch.nn.Parameter(torch.tensor(delta), requires_grad=False)

    def forward(self, input):
        """Output of the BiLU activation function"""
        return input * self.alpha + self.delta * torch.abs(input)


class NormBiTanh(torch.nn.Module):
    """Bijective Normalized Tanh layer
     f(x) = alpha*Tanh + (1-alpha)*x activation
     """

    def __init__(self, alpha=0.3):
        super(NormBiTanh, self).__init__()
        self.alpha = torch.nn.Parameter(torch.tensor(alpha), requires_grad=False)

    def forward(self, input):
        """Output of the NormBiTanh activation function"""
        return self.alpha * input + (1 - self.alpha) * torch.tanh(input)
