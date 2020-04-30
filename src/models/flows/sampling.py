"""Sampling layers
The input of flows as we use them is nearly always generated data from some distribution
provided with its log-inverse PDF. As a result, it can be convenient to plug the first layer
of a flow as a sampling layers that draws points and computes the required PDF
"""

import torch


class FactorizedFlowPrior(torch.nn.Module):
    """Sample d-dimensional data from a factorized 1D PDF over each dimension
    The 1D PDF is expected to be a pytorch.probability.Distribution object
    but it can be any object that implements the `sample`and `log_prob` methods.

    NB: we provide the 1D prior object explicitly as pytorch distributions don't
    respond appropriately to the `.to(device)`. To sample on a device, provide a
    prior initialized with parameters already on the correct device.
    """
    def __init__(self, *, d, prior_1d):
        super(FactorizedFlowPrior, self).__init__()
        self.d = d
        self.prior = prior_1d

    def forward(self, n_batch):
        # Generate normally distributed points
        x = self.prior.sample((n_batch, self.d))
        # Compute their per-dimension log PDF and sum them across dimenions
        # to get the log PDF then add a batch dimension
        log_j = -torch.sum(self.prior.log_prob(x), -1).unsqueeze(-1)
        return torch.cat([x, log_j], -1)