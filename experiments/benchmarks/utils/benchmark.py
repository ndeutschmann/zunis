"""Benchmarking functions"""
import logging
import torch
from zunis.integration import Integrator
from utils.integrator_integrals import validate_integral_integrator
from utils.flat_integrals import validate_known_integrand_flat
from utils.config.loaders import create_integrator_args

logger = logging.getLogger(__name__)


def benchmark_known_integrand(d, integrand, integrator_config, n_batch=100000, integrand_params=None,
                              device=torch.device("cpu")):
    """Integrate an known integrand and compare with the theoretical result

    Parameters
    ----------
    d
    integrand
    integrator_config
    n_batch
    integrand_params
    device

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
        integrator_config = dict()
    integrator_args = create_integrator_args(integrator_config)
    integrator = Integrator(f=f, d=d, device=device, **integrator_args)

    logger.debug("=" * 72)
    logger.info("Running integrator")
    integrator_result = validate_integral_integrator(integrand, integrator, n_batch=n_batch)

    logger.debug("=" * 72)
    logger.info("Running flat sampler")
    flat_result = validate_known_integrand_flat(integrand, d=d, n_batch=n_batch, device=device)

    logger.debug("=" * 72)
    integrator_result["speedup"] = (flat_result["value_std"] / integrator_result["value_std"]) ** 2
    logger.info(f"speedup: {integrator_result['speedup']}")
    logger.debug("=" * 72)
    logger.debug(" " * 72)
    integrator_result["d"] = d

    integrator_result.update(integrator_config.as_flat_dict())
    integrator_result.update(integrand_params)

    return integrator_result

