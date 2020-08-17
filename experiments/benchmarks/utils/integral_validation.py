"""Generic facilities to validate integrals"""
from better_abc import ABC, abstractmethod
from math import sqrt

import logging

from utils.integrals import mean_std_integrand
from utils.comparison_record import ComparisonRecord
from utils.integrands import KnownIntegrand

logger = logging.getLogger(__name__)


class Sampler(ABC):
    """Sampling tool for integral validation
     that mimics the behavior of sample_survey/sample_refine from Integrators"""

    @abstractmethod
    def sample(self, f, n_batch=10000, *args, **kwargs):
        """

        Parameters
        ----------
        f: utils.integrands.Integrand
        n_batch: int

        Returns
        -------
            tuple of torch.Tensor
            x,px,fx: points, pdfs, function values
        """


def validate_integral(integrand, sampler, n_batch=10000, sigma_cutoff=2):
    """Compute the integral and check whether it matches the known value

    Parameters
    ----------
    integrand: utils.integrands.KnownIntegrand
    sampler: Sampler
    n_batch: int
    sigma_cutoff: numbers.Number

    Returns
    -------
        ComparisonRecord
    """
    assert isinstance(integrand, KnownIntegrand)

    _, px, fx = sampler.sample(integrand, n_batch=n_batch)
    integral, std = mean_std_integrand(fx, px)
    unc = std / sqrt(n_batch)
    correct_integral = integrand.integral()
    logger.info(f"Estimated result: {integral:.2e}+/-{unc:.2e}")
    logger.info(f"Correct result:   {correct_integral:.2e}")
    logger.info(f'Difference:       {100 * integrand.compare_relative(integral):.2f}%')
    logger.info(f"Significance:     {integrand.compare_relative(integral) / unc:.2f}σ")
    result = ComparisonRecord(
        value=integral,
        value_std=unc,
        target=correct_integral,
        target_std=0.,
        sigma_cutoff=sigma_cutoff,
        sigmas_off=integrand.compare_absolute(integral) / unc,
        percent_difference=100 * integrand.compare_relative(integral),
        match=integrand.compare_absolute(integral) / unc <= sigma_cutoff
    )

    return result
