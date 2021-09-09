"""Facilities for evaluating integrals from a batch of points"""
import torch


def compute_mc_statistics(fx, px):
    """Compute the relevant Monte Carlo evaluation statistics expectation for a function evaluated
    on a sample of points taken from a known distribution: integral value, integral value standard deviation, unweighting efficiency

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

    weights = (fx / px)
    int_var, int_mean = torch.var_mean(weights)
    int_mean = int_mean.detach().item()
    int_std = int_var.sqrt().detach().item()

    unweighting_efficiency = int_mean / weights.max().item()
    unweighting_efficiency = unweighting_efficiency

    return int_mean, int_std, unweighting_efficiency
