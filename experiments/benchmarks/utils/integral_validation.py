"""Generic facilities to validate integrals"""
from better_abc import ABC, abstractmethod
from math import sqrt
import pandas as pd
import numpy as np

import logging

from utils.integrals import compute_mc_statistics
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

    def get_history(self):
        """Return stored training history from the sampler
        This abstract class has no training history and returns an empty history

        Returns
        -------
            pandas.Dataframe
                empty dataframe
        """
        return pd.DataFrame()


def evaluate_integral(integrand, sampler, n_batch=10000, keep_history=False):
    """Evaluate an integral

    Parameters
    ----------
    integrand: utils.integrands.KnownIntegrand
    sampler: Sampler
    n_batch: int
    keep_history: bool

    Returns
    -------
        utils.record.EvaluationRecord
    """

    _, px, fx = sampler.sample(integrand, n_batch=n_batch)
    integral, std, unweighting_efficiency = compute_mc_statistics(fx, px)
    unc = std / sqrt(n_batch)

    logger.info(f"Estimated result: {integral:.2e}+/-{unc:.2e}")
    result = EvaluationRecord(
        value=integral,
        value_std=unc,
        unweighting_eff=unweighting_efficiency
    )

    if keep_history:
        result["history"] = sampler.get_history()

    return result


def evaluate_integral_stratified(integrand, sampler, n_batch=10000, keep_history=False):
    """Evaluate an integral with a stratified sampling sampler

    Parameters
    ----------
    integrand: utils.integrands.KnownIntegrand
    sampler: Sampler
    n_batch: int
    keep_history: bool

    Returns
    -------
        utils.record.EvaluationRecord
    """
    _, (wx, hcs), fx = sampler.sample(integrand, n_batch=n_batch)

    integral = wx.dot(fx).item()
    variance = 0

    for hc in np.unique(hcs):
        idx = (hcs == hc)
        px = 1 / wx[idx] / np.sum(idx)

        var_hc = np.var(fx[idx] / px)
        variance += var_hc

    unc = sqrt(variance.item()/n_batch)

    logger.info(f"Estimated result: {integral:.2e}+/-{unc:.2e}")
    result = EvaluationRecord(
        value=integral,
        value_std=unc,
    )

    if keep_history:
        result["history"] = sampler.get_history()

    return result


def validate_integral(integrand, sampler, n_batch=10000, sigma_cutoff=2, keep_history=False):
    """Compute the integral and check whether it matches the known value

    Parameters
    ----------
    integrand: utils.integrands.KnownIntegrand
    sampler: Sampler
    n_batch: int
    sigma_cutoff: float
    keep_history: bool

    Returns
    -------
        utils.record.EvaluationRecord
    """
    assert isinstance(integrand, KnownIntegrand)

    eval_result = evaluate_integral(integrand, sampler, n_batch, keep_history=keep_history)
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

    if keep_history:
        result["history"] = eval_result["history"]

    return result


def compare_integral_result(result1, result2, sigma_cutoff=2, keep_history=False):
    """Compute an integral in two different ways and compare

    Parameters
    ----------
    result1: utils.record.EvaluationRecord
    result2: utils.record.EvaluationRecord
    sigma_cutoff: float
    keep_history: bool

    Returns
    -------
        utils.record.ComparisonRecord
    """
    integral1 = result1["value"]
    unc1 = result1["value_std"]
    try:
        unweight_eff1 = result1['unweighting_eff']
    except KeyError:
        unweight_eff1 = float('nan')

    integral2 = result2["value"]
    unc2 = result2["value_std"]
    try:
        unweight_eff2 = result2['unweighting_eff']
    except KeyError:
        unweight_eff2 = float('nan')

    absum = abs(integral1) + abs(integral2)
    absdiff = abs(integral1 - integral2)
    percent_diff = 2 * absdiff / absum
    diff_unc = sqrt(unc1 ** 2 + unc2 ** 2)
    sigmas = absdiff / diff_unc

    logger.info(f"Result 1:             {integral1:.2e}+/-{unc1:.2e}")
    logger.info(f"Result 2:             {integral2:.2e}+/-{unc2:.2e}")
    logger.info(f"Difference:           {100. * percent_diff:.2f}%")
    logger.info(f"Significance:         {sigmas:.2f}{sigma}")
    logger.info(f"Variance ratio (2/1): {(unc2 / unc1) ** 2:.2e}")

    result = ComparisonRecord(
        value=integral1,
        target=integral2,
        value_std=unc1,
        target_std=unc2,
        sigma_cutoff=sigma_cutoff,
        sigmas_off=sigmas,
        percent_difference=100 * percent_diff,
        variance_ratio=(unc2 / unc1) ** 2,
        match=sigmas <= sigma_cutoff,
        value_unweighting_eff=unweight_eff1,
        target_unweighting_eff=unweight_eff2,
        unweighting_eff_ratio=(unweight_eff1/unweight_eff2)
    )

    if keep_history:
        try:
            history1 = result1["history"]
        except KeyError:
            logger.warning(f"No history available in the first record: {result1}")
            history1 = None
        try:
            history2 = result2["history"]
        except KeyError:
            logger.warning(f"No history available in the second record: {result2}")
            history2 = None
        result["value_history"] = history1
        result["target_history"] = history2

    return result


def compare_integrals(integrand1, sampler1, sampler2, integrand2=None, n_batch=10000, sigma_cutoff=2,
                      keep_history=False):
    """Compute an integral in two different ways and compare

    Parameters
    ----------
    integrand1: callable
    integrand2: None callable
        if not None, use `integrand1` with `sampler1` and `integrand2` with `sampler2`, else use `integrand1` for both
    sampler1: Sampler
    sampler2: Sampler
    n_batch: int
    sigma_cutoff: float
    keep_history: bool

    Returns
    -------
        utils.record.ComparisonRecord
    """
    result1 = evaluate_integral(integrand1, sampler2, n_batch, keep_history=keep_history)

    if integrand2 is None:
        integrand2 = integrand1
    result2 = evaluate_integral(integrand2, sampler1, n_batch, keep_history=keep_history)
    return compare_integral_result(result1, result2, sigma_cutoff, keep_history=keep_history)
