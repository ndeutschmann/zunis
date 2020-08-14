"""Computing integrals using Naive Monte Carlo"""
import logging
import torch
from math import sqrt
from utils.integrals import mean_std_integrand
from utils.comparison_record import ComparisonRecord

logger = logging.getLogger(__name__)


def sample_flat(d, n_batch=10000, device=torch.device("cpu")):
    """Sample a batch of points in the d-dimensional hypercube

    Parameters
    ----------
    d: int
    n_batch: int
    device: torch.device

    Returns
    -------
    torch.Tensor
    """

    return torch.zeros(n_batch, d, device=device).uniform_(0., 1.)


def validate_known_integrand_flat(f, d, n_batch=10000, sigma_cutoff=2):
    """

    Parameters
    ----------
    f
    d
    n_batch
    sigma_cutoff

    Returns
    -------

    """
    x = sample_flat(d, n_batch)
    px = torch.ones(n_batch)
    integral, std = mean_std_integrand(f, x, px)
    unc = std / sqrt(n_batch)
    correct_integral = f.integral()
    logger.info(f"Estimated result: {integral:.2e}+/-{unc:.2e}")
    logger.info(f"Correct result:   {correct_integral:.2e}")
    logger.info(f'Difference:       {100 * f.compare_relative(integral):.2f}%')
    logger.info(f"Significance:     {f.compare_relative(integral) / unc:.2f}σ")
    result = ComparisonRecord(
        value=integral,
        value_std=unc,
        target=correct_integral,
        target_std=0.,
        sigma_cutoff=sigma_cutoff,
        sigmas_off=f.compare_absolute(integral) / unc,
        percent_difference=100 * f.compare_relative(integral),
        match=f.compare_absolute(integral) / unc <= sigma_cutoff
    )

    return result
