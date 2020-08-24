"""Computing integrals using Naive Monte Carlo"""
import torch
from utils.integral_validation import validate_integral, evaluate_integral, Sampler
from utils.integrands.abstract import Integrand


class FlatSampler(Sampler):
    """Sampler for uniform sampling in the d-dimensional hypercube"""

    def __init__(self, d, device=torch.device("cpu")):
        """

        Parameters
        ----------
        d: int
        """
        self.d = d
        self.device = device

    def sample(self, f, n_batch=10000, *args, **kwargs):
        """

        Parameters
        ----------
        f: function or utils.Integrand
            function that evaluate batches of points and returns batches of points
        n_batch: int
        device: torch.device

        Returns
        -------
            tuple of torch.Tensor
            x,px,fx: points, pdfs, function values
        """

        if isinstance(f, Integrand):
            assert self.d == f.d

        device = self.device
        if "device" in kwargs:
            device = kwargs["device"]

        x = torch.zeros(n_batch, self.d, device=device).uniform_(0., 1.)
        px = torch.ones(n_batch, device=device)
        fx = f(x)
        return x, px, fx


def validate_known_integrand_flat(f, d, n_batch=10000, sigma_cutoff=2, device=torch.device("cpu")):
    """Validate a known integral using uniform sampling

    Parameters
    ----------
    f: utils.integrands.KnownIntegrand
    d: int
    n_batch: int
    sigma_cutoff: float
    device: torch.device

    Returns
    -------
       utils.record.ComparisonRecord
    """

    return validate_integral(f, FlatSampler(d, device), n_batch, sigma_cutoff)


def evaluate_integral_flat(f, d, n_batch=10000, device=torch.device("cpu")):
    """Evaluate an integral using uniform sampling

    Parameters
    ----------
    f: utils.integrands.KnownIntegrand
    d: int
    n_batch: int
    device: torch.device

    Returns
    -------
       utils.record.EvaluationRecord
    """

    return evaluate_integral(f, FlatSampler(d, device), n_batch)