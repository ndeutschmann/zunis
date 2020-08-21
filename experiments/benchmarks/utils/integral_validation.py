"""Generic facilities to validate integrals"""
from better_abc import ABC, abstractmethod
from math import sqrt

import logging

from utils.integrals import mean_std_integrand
from utils.record import ComparisonRecord, EvaluationRecord
from utils.integrands.abstract import KnownIntegrand
from utils.vanity import sigma

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


def evaluate_integral(integrand, sampler, n_batch=10000):
    """Evaluate an integral

    Parameters
    ----------
    integrand: utils.integrands.KnownIntegrand
    sampler: Sampler
    n_batch: int

    Returns
    -------
        utils.record.EvaluationRecord
    """

    _, px, fx = sampler.sample(integrand, n_batch=n_batch)
    integral, std = mean_std_integrand(fx, px)
    unc = std / sqrt(n_batch)

    logger.info(f"Estimated result: {integral:.2e}+/-{unc:.2e}")
    result = EvaluationRecord(
        value=integral,
        value_std=unc,
    )

    return result


def validate_integral(integrand, sampler, n_batch=10000, sigma_cutoff=2):
    """Compute the integral and check whether it matches the known value

    Parameters
    ----------
    integrand: utils.integrands.KnownIntegrand
    sampler: Sampler
    n_batch: int
    sigma_cutoff: float

    Returns
    -------
        utils.record.EvaluationRecord
    """
    assert isinstance(integrand, KnownIntegrand)

    eval_result = evaluate_integral(integrand, sampler, n_batch)
    integral = eval_result["value"]
    unc = eval_result["value_std"]
    if unc == 0.:
        unc = 1.e-16
    correct_integral = integrand.integral()
    logger.info(f"Estimated result: {integral:.2e}+/-{unc:.2e}")
    logger.info(f"Correct result:   {correct_integral:.2e}")
    logger.info(f'Difference:       {100 * integrand.compare_relative(integral):.2f}%')
    logger.info(f"Significance:     {integrand.compare_absolute(integral) / unc:.2f}{sigma}")
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


def compare_integral_result(result1, result2, sigma_cutoff=2):
    """Compute an integral in two different ways and compare

    Parameters
    ----------
    result1: utils.record.EvaluationRecord
    result2: utils.record.EvaluationRecord
    sigma_cutoff: float

    Returns
    -------
        utils.record.ComparisonRecord
    """
    integral1 = result1["value"]
    unc1 = result1["value_std"]

    integral2 = result2["value"]
    unc2 = result2["value_std"]

    absum = abs(integral1) + abs(integral2)
    absdiff = abs(integral1 - integral2)
    percent_diff = 2 * absdiff / absum
    diff_unc = sqrt(unc1 ** 2 + unc2 ** 2)
    sigmas = absdiff / diff_unc

    logger.info(f"Result 1:     {integral1:.2e}+/-{unc1:.2e}")
    logger.info(f"Result 2:     {integral2:.2e}+/-{unc2:.2e}")
    logger.info(f"Difference:   {100. * percent_diff:.2f}%")
    logger.info(f"Significance: {sigmas:.2f}{sigma}")

    return ComparisonRecord(
        value=result1,
        target=result2,
        value_std=unc1,
        target_std=unc2,
        sigma_cutoff=sigma_cutoff,
        sigmas_off=sigmas,
        percent_difference=100 * percent_diff,
        match=sigmas <= sigma_cutoff
    )


def compare_integrals(integrand1, sampler1, sampler2, integrand2=None, n_batch=10000, sigma_cutoff=2):
    """Compute an integral in two different ways and compare

    Parameters
    ----------
    integrand1: callable
    integrand2: None callable
        if not None, use `integrand1` with `sampler1` and `integrand2` with `sampler2`, else use `integrand1` for both
    sampler1
    sampler2
    n_batch
    sigma_cutoff

    Returns
    -------
        utils.record.ComparisonRecord
    """
    result1 = evaluate_integral(integrand1, sampler2, n_batch)

    if integrand2 is None:
        integrand2 = integrand1
    result2 = evaluate_integral(integrand2, sampler1, n_batch)
    return compare_integral_result(result1, result2, sigma_cutoff)
