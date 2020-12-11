import logging

import torch
from dictwrapper import NestedMapping

from utils.benchmark.benchmarker import Benchmarker, GridBenchmarker, RandomHyperparameterBenchmarker
from zunis.utils.config.loaders import get_default_integrator_config, create_integrator_args
from utils.flat_integrals import validate_known_integrand_flat
from utils.integrator_integrals import validate_integral_integrator
from zunis.integration import Integrator

logger = logging.getLogger(__name__)


class KnownIntegrandBenchmarker(Benchmarker):
    """Benchmark by comparing to a known integrand with an exact integral value"""

    def benchmark_method(self, d, integrand, integrator_config=None, integrand_params=None, n_batch=100000,
                         keep_history=False, device=torch.device("cpu")):
        """Integrate an known integrand and compare with the theoretical result

        Parameters
        ----------
        d: int
            number of dimensions
        integrand: constructor for utils.integrands.abstract.KnownIntegrand
            integrand class to be tested. Expects a constuctor for that class, i.e. a callable that returns an instance.
        integrator_config: dictwrapper.NestedMapping, None
            configuration to be passed to :py:func:`Integrator <zunis.integration.default_integrator.Integrator>`. If `None`, use the
            :py:func:`default <utils.config.loaders.get_default_integrator_config>`.
        n_batch: int
            batch size used for the benchmarking (after training)
        integrand_params: dict
            dictionary of parameters provided to `integrand` through `integrand(**integrand_params)`.
        device: torch.device
            torch device on which to train and run the `Integrator`.

        Returns
        -------

        """
        logger.debug("=" * 72)
        logger.info("Defining integrand")
        if integrand_params is None:
            integrand_params = dict()
        f = integrand(d=d, device=device, **integrand_params)

        logger.debug("=" * 72)
        logger.info("Defining integrator")
        if integrator_config is None:
            integrator_config = get_default_integrator_config()
        integrator_args = create_integrator_args(integrator_config)
        integrator = Integrator(f=f, d=d, device=device, **integrator_args)

        logger.debug("=" * 72)
        logger.info("Running integrator")
        integrator_result = validate_integral_integrator(f, integrator, n_batch=n_batch, keep_history=keep_history)

        logger.debug("=" * 72)
        logger.info("Running flat sampler")
        flat_result = validate_known_integrand_flat(f, d=d, n_batch=n_batch, device=device)

        logger.debug("=" * 72)
        integrator_result["speedup"] = (flat_result["value_std"] / integrator_result["value_std"]) ** 2
        logger.info(f"speedup: {integrator_result['speedup']}")
        logger.debug("=" * 72)
        logger.debug(" " * 72)
        integrator_result["d"] = d

        if isinstance(integrator_config, NestedMapping):
            integrator_result.update(integrator_config.as_flat_dict())
        else:
            integrator_result.update(integrator_config)

        integrator_result.update(integrand_params)

        integrator_result["d"] = d

        return integrator_result, integrator


class KnownIntegrandGridBenchmarker(GridBenchmarker, KnownIntegrandBenchmarker):
    """Benchmark against a known integrand by sampling configurations from a grid"""


class KnownIntegrandRandomHPBenchmarker(RandomHyperparameterBenchmarker, KnownIntegrandBenchmarker):
    """Benchmark against a known integrand by sampling integrator hyperparameters randomly"""
