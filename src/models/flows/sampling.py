"""Sampling layers
The input of flows as we use them is nearly always generated data from some distribution
provided with its log-inverse PDF. As a result, it can be convenient to plug the first layer
of a flow as a sampling layers that draws points and computes the required PDF
"""

import torch


class FactorizedFlowSampler(torch.nn.Module):
    """Sample d-dimensional data from a factorized 1D PDF over each dimension
    The 1D PDF is expected to be a pytorch.probability.Distribution object
    but it can be any object that implements the `sample`and `log_prob` methods.

    NB: we provide the 1D prior object explicitly as pytorch distributions don't
    respond appropriately to the `.to(device)`. To sample on a device, provide a
    prior initialized with parameters already on the correct device.
    """
    def __init__(self, *, d, prior_1d):
        super(FactorizedFlowSampler, self).__init__()
        self.d = d
        self.prior = prior_1d

    def log_prob(self, x):
        """Compute, point-per-point, the log-PDF of a batch of points"""
        assert len(x.shape) == 2 and x.shape[1] == self.d, f"Expected shape: (:, {self.d})"
        return torch.sum(self.prior.log_prob(x), -1)

    def forward(self, n_batch):
        """Sample n_batch points and stack them with their jacobians"""
        # Generate normally distributed points
        x = self.prior.sample((n_batch, self.d))
        # Compute their per-dimension log PDF and sum them across dimenions
        # to get the log PDF then add a batch dimension
        log_j = - self.log_prob(x)
        return torch.cat([x, log_j.unsqueeze(-1)], -1)


class FactorizedGaussianSampler(FactorizedFlowSampler):
    """Factorized gaussian prior
    Note that tensorflow distribution objects cannot easily be moved devices so specify the right
    device at initialization.
    """
    def __init__(self, *, d, mu=0., sig=1., device=None):
        # Copy data
        sig_ = torch.tensor(sig)
        mu_ = torch.tensor(mu)

        if device is not None:
            sig_ = sig_.to(device)
            mu_ = mu_.to(device)

        prior_1d = torch.distributions.normal.Normal(mu_, sig_)
        super(FactorizedGaussianSampler, self).__init__(d=d, prior_1d=prior_1d)


class UniformSampler(FactorizedFlowSampler):
    """Factorized uniform prior
    Note that tensorflow distribution objects cannot easily be moved devices so specify the right
    device at initialization.
    """
    def __init__(self, *, d, low=0., high=1., device=None):
        # Copy data
        low_ = torch.tensor(low, dtype=torch.get_default_dtype())
        high_ = torch.tensor(high, dtype=torch.get_default_dtype())

        if device is not None:
            low_ = low_.to(device)
            high_ = high_.to(device)

        prior_1d = torch.distributions.Uniform(low_, high_)
        super(UniformSampler, self).__init__(d=d, prior_1d=prior_1d)
