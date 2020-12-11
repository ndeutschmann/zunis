import logging

import torch
import vegas
from dictwrapper import NestedMapping

from utils.benchmark.benchmarker import Benchmarker, GridBenchmarker, RandomHyperparameterBenchmarker, \
    SequentialBenchmarker
from zunis.utils.config.loaders import get_default_integrator_config, create_integrator_args
from utils.flat_integrals import evaluate_integral_flat
from utils.integral_validation import compare_integral_result
from utils.integrator_integrals import evaluate_integral_integrator
from utils.vegas_integrals import evaluate_integral_vegas
from zunis.integration import Integrator

logger = logging.getLogger(__name__)


class VegasBenchmarker(Benchmarker):
    """Benchmark by comparing with VEGAS"""

    def benchmark_method(self, d, integrand, integrator_config=None, integrand_params=None, n_batch=100000,
                         keep_history=False, device=torch.device("cpu")):
        logger.debug("=" * 72)
        logger.info("Defining integrand")
        if integrand_params is None:
            integrand_params = dict()
        f = integrand(d=d, device=device, **integrand_params)
        vf = f.vegas(device=device)

        logger.debug("=" * 72)
        logger.info("Defining integrator")
        if integrator_config is None:
            integrator_config = get_default_integrator_config()
        integrator_args = create_integrator_args(integrator_config)
        integrator = Integrator(f=f, d=d, device=device, **integrator_args)

        vintegrator = vegas.Integrator([[0, 1]] * d, max_nhcube=1)

        integrator_result = evaluate_integral_integrator(f, integrator, n_batch=n_batch, keep_history=keep_history)
        vegas_result = evaluate_integral_vegas(vf, vintegrator, n_batch=n_batch,
                                               n_batch_survey=integrator_args["n_points_survey"])
        flat_result = evaluate_integral_flat(f, d, n_batch=n_batch, device=device)

        result = compare_integral_result(integrator_result, vegas_result, sigma_cutoff=3, keep_history=keep_history)
        result["flat_variance_ratio"] = (flat_result["value_std"] / result["value_std"]) ** 2

        if isinstance(integrator_config, NestedMapping):
            result.update(integrator_config.as_flat_dict())
        else:
            result.update(integrator_config)

        result.update(integrand_params)

        result["d"] = d

        return result, integrator


class VegasGridBenchmarker(GridBenchmarker, VegasBenchmarker):
    """Benchmark against VEGAS by sampling parameters on a grid"""


class VegasRandomHPBenchmarker(RandomHyperparameterBenchmarker, VegasBenchmarker):
    """Benchmark against VEGAS by sampling integrator hyperparameters randomly"""


class VegasSequentialBenchmarker(SequentialBenchmarker, VegasBenchmarker):
    """Benchmark against VEGAS by testing on a sequence of (dimension, integrand, integrator) triplets"""
