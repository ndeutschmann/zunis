"""Model utilities"""

import torch


class Reshift(torch.nn.Module):
    """Un-trainable activation: shift and scale data

    The input x is transformed as x*scale + offset. Scale and offset are untrainable scalars.
    """
    def __init__(self, scale=2., offset=-1.):
        """

        Parameters
        ----------
        scale: float
        offset: float
        """
        super(Reshift, self).__init__()
        self.scale = torch.nn.Parameter(data=torch.scalar_tensor(scale), requires_grad=False)
        self.offset = torch.nn.Parameter(data=torch.scalar_tensor(offset), requires_grad=False)

    def forward(self, x):
        return x*self.scale + self.offset
