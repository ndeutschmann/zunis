"""Benchmarking functions"""
import logging
from copy import deepcopy
from itertools import product
from collections.abc import Sequence

import pandas as pd
import torch
import vegas

from dictwrapper import NestedMapping
from zunis.integration import Integrator
from utils.integrator_integrals import validate_integral_integrator
from utils.integrator_integrals import evaluate_integral_integrator
from utils.vegas_integrals import evaluate_integral_vegas
from utils.flat_integrals import evaluate_integral_flat
from utils.flat_integrals import validate_known_integrand_flat
from utils.integral_validation import compare_integral_result
from utils.config.loaders import create_integrator_args, get_default_integrator_config, get_sql_types
from utils.logging import get_benchmark_logger, get_benchmark_logger_debug
from utils.torch_utils import get_device
from utils.data_storage.dataframe2sql import append_dataframe_to_sqlite

# TODO: this should be turned into classes

def benchmark_known_integrand(d, integrand, integrator_config=None, n_batch=100000, integrand_params=None, logger=None,
                              device=torch.device("cpu")):
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
    if logger is None:
        logger = logging.getLogger(__name__)

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
    integrator_result = validate_integral_integrator(f, integrator, n_batch=n_batch)

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

    return integrator_result


def run_benchmark_grid(dimensions, integrand, *,
                       base_integrand_params, base_integrator_config=None,
                       integrand_params_grid=None, integrator_config_grid=None,
                       n_batch=100000, debug=True, cuda=0, benchmark_method,
                       sql_dtypes=None, dbname="benchmarks.db", experiment_name="benchmark"):
    """Run benchmarks over a grid of parameters for the integrator and the integrand."""

    if debug:
        logger = get_benchmark_logger_debug(experiment_name, zunis_integration_level=logging.DEBUG,
                                            zunis_training_level=logging.DEBUG, zunis_level=logging.DEBUG)
    else:
        logger = get_benchmark_logger(experiment_name)

    device = get_device(cuda_ID=cuda)
    results = pd.DataFrame()

    if isinstance(dimensions, int):
        dimensions = [dimensions]
    assert isinstance(dimensions, Sequence) and all([isinstance(d, int) for d in dimensions]) and len(dimensions)>0,\
        "argument dimensions must be an integer or a list of integers"

    if integrand_params_grid is None and integrator_config_grid is None and len(dimensions) == 1:
        result = benchmark_method(dimensions[0], integrand=integrand,
                                  integrand_params=base_integrand_params, integrator_config=base_integrator_config,
                                  n_batch=n_batch, logger=logger, device=device).as_dataframe()
        results = result

    else:
        if integrand_params_grid is None:
            integrand_params_grid = dict()
        if integrator_config_grid is None:
            integrator_config_grid = dict()

        if base_integrator_config is None:
            base_integrator_config = get_default_integrator_config()

        integrator_config = deepcopy(base_integrator_config)
        integrand_params = deepcopy(base_integrand_params)

        integrator_grid_keys = integrator_config_grid.keys()
        integrand_grid_keys = integrand_params_grid.keys()

        integrator_full_grid = product(*integrator_config_grid.values())
        integrand_full_grid = product(*integrand_params_grid.values())

        for d in dimensions:
            for integrator_update in integrator_full_grid:
                for integrand_update in integrand_full_grid:
                    integrator_config.update(dict(zip(integrator_grid_keys, integrator_update)))
                    integrand_params.update(dict(zip(integrand_grid_keys, integrand_update)))
                    result = benchmark_method(d, integrand=integrand,
                                              integrand_params=integrand_params, integrator_config=integrator_config,
                                              n_batch=n_batch, logger=logger, device=device).as_dataframe()

                    results = pd.concat([results, result], ignore_index=True)

    if not debug:
        if sql_dtypes is None:
            sql_dtypes = get_sql_types()
        append_dataframe_to_sqlite(results, dbname=dbname, tablename=experiment_name, dtypes=sql_dtypes)


def run_benchmark_grid_know_integrand(dimensions, integrand, *,
                                      base_integrand_params, base_integrator_config=None,
                                      integrand_params_grid=None, integrator_config_grid=None,
                                      n_batch=100000, debug=True, cuda=0,
                                      sql_dtypes=None, dbname="benchmarks.db", experiment_name="benchmark"):
    return run_benchmark_grid(dimensions=dimensions, integrand=integrand, base_integrand_params=base_integrand_params,
                              base_integrator_config=base_integrator_config,
                              integrand_params_grid=integrand_params_grid,
                              integrator_config_grid=integrator_config_grid, n_batch=n_batch, debug=debug, cuda=cuda,
                              sql_dtypes=sql_dtypes, dbname=dbname, experiment_name=experiment_name,
                              benchmark_method=benchmark_known_integrand)
