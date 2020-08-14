"""Facilities for evaluating integrals from a batch of points"""
import torch
from zunis.training.weighted_dataset.training_record import DictWrapper

def mean_std_integrand(f, x, px):
    """Compute the expectation value and the standard deviation of a function evaluated
    on a sample of points taken from a known distribution.

    Parameters
    ----------
    f: function
    x: torch.Tensor
        batch of points on which to evaluate the function with shape (N,d)
    px: torch.tensor
        batch of PDF values for each of the sampled x

    Returns
    -------
    tuple of float
    """
    assert len(x.shape) == 2
    assert len(px.shape) == 1
    assert x.shape[0] == px.shape[0]

    v, m = torch.var_mean(f(x) / px)
    return m.detach().item(), (v.detach()).sqrt().item()
