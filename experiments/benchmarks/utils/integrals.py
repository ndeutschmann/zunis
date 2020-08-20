"""Facilities for evaluating integrals from a batch of points"""
import torch

def mean_std_integrand(fx, px):
    """Compute the expectation value and the standard deviation of a function evaluated
    on a sample of points taken from a known distribution.

    Parameters
    ----------
    fx: torch.Tensor
        batch of function values
    px: torch.tensor
        batch of PDF values for each of the sampled x

    Returns
    -------
    tuple of float
    """

    assert len(px.shape) == 1
    assert fx.shape == fx.shape

    v, m = torch.var_mean(fx / px)
    return m.detach().item(), v.detach().sqrt().item()
