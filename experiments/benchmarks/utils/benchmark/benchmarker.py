import logging
import random
from collections import Sequence, MutableMapping
from copy import deepcopy
from itertools import product
from typing import Dict, List

import torch
from abc import ABC, abstractmethod
from better_abc import ABCMeta, abstract_attribute
from dictwrapper import NestedMapping

from zunis.utils.config.configuration import Configuration
from utils.config.loaders import get_sql_types
from zunis.utils.config.loaders import get_default_integrator_config
from utils.data_storage.dataframe2sql import append_dataframe_to_sqlite
from utils.logging import set_benchmark_logger_debug, set_benchmark_logger
from utils.torch_utils import get_device

logger = logging.getLogger(__name__)


class Benchmarker(ABC):
    """Benchmarker class used to run benchmarks of our integrator comparing to others and perform parameter scans"""

    @abstractmethod
    def benchmark_method(self, d, integrand, integrator_config=None, integrand_params=None, n_batch=100000,
                         keep_history=False, device=torch.device("cpu")):
        """Run a single benchmark evaluation of a given integrand"""

    @abstractmethod
    def generate_config_samples(self, dimensions, integrator_grid, integrand_grid):
        """Sample over dimensions, integrator and integrand configurations from lists of possible option values.

        Parameters
        ----------
        dimensions : List[int]
            list of dimensions to sample from
        integrator_grid : Dict[str, List[Any]]
            mapping (option name) -> (list of values) for the integrator
        integrand_grid: Dict[str, List[Any]]
            mapping (option name) -> (list of values) for the integrand

        Yields
        -------
        Tuple[int, Dict[str, Any], Dict[str, Any]]
            triplets (d, integrator_config, integrand_params) that can be used to sample configurations
        """

    @staticmethod
    def set_benchmark_grid_config_param(benchmark_grid_config, param, param_value, config):
        """Set an argument in the configuration dictionary to be provided to `run_benchmark_grid` according to the
        established hierarchy:

        1. direct argument value (typically from CLI)
        2. config file value
        3. default value - what is already in the config dictionary when entering this function

        Parameters
        ----------
        benchmark_grid_config : dict
            configuration dictionary being constructed. Keys should match the arguments of `run_benchmark_grid`
        param : str
            name of the parameter to be set
        param_value : optional
            direct argument value for the parameter. Use the configuration dictionary if not provided
        config : Configuration, optional
            Configuration nested dictionary for the benchmarking run at hand. If it is not provided or does not specify
            a value of `param`, use the default value instead.
        """
        assert param in benchmark_grid_config, f"A default value must be set for parameter {param}"
        if param_value is not None:
            if isinstance(benchmark_grid_config[param], MutableMapping):
                benchmark_grid_config[param].update(param_value)
            else:
                benchmark_grid_config[param] = param_value
        elif config is not None and param in config:
            if isinstance(benchmark_grid_config[param], MutableMapping):
                benchmark_grid_config[param].update(config[param])
            else:
                benchmark_grid_config[param] = config[param]

    def set_benchmark_grid_config(self, config=None, dimensions=None, n_batch=None, keep_history=None,
                                  dbname=None, experiment_name=None, cuda=None, debug=None,
                                  default_dimension=(2,), base_integrand_params=()):
        """Prepare standard arguments for `run_benchmark_grid`
        The parameter importance hierarchy is:

        1. CLI arguments (direct arguments to this function)
        2. config file (filepath given as `config`)
        3. default values provided as argument
        """
        benchmark_config = {
            "dimensions": default_dimension,
            "base_integrand_params": base_integrand_params,
            "base_integrator_config": get_default_integrator_config(),
            "integrand_params_grid": None,
            "integrator_config_grid": None,
            "n_batch": 100000,
            "keep_history": False,
            "dbname": None,
            "experiment_name": "benchmark",
            "cuda": 0,
            "debug": True
        }
        if config is not None and not isinstance(config, Configuration):
            config = Configuration.from_yaml(config, check=False)

        self.set_benchmark_grid_config_param(benchmark_config, "dimensions", dimensions, config)
        self.set_benchmark_grid_config_param(benchmark_config, "keep_history", keep_history, config)
        # the base_integrand_params is too problem specific to be provided by direct argument/CLI.
        # Either a default or a config-file input
        self.set_benchmark_grid_config_param(benchmark_config, "base_integrand_params", None, config)
        self.set_benchmark_grid_config_param(benchmark_config, "base_integrator_config", None, config)
        # The integrator grid is too unwieldy to be used through direct argument/CLI. Use a config file
        # Either a default or a config-file input
        self.set_benchmark_grid_config_param(benchmark_config, "integrator_config_grid", None, config)
        # The integrand_params_grid is too problem specific to be provided by direct argument/CLI.
        # Either a default or a config-file input
        self.set_benchmark_grid_config_param(benchmark_config, "integrand_params_grid", None, config)

        self.set_benchmark_grid_config_param(benchmark_config, "n_batch", n_batch, config)
        self.set_benchmark_grid_config_param(benchmark_config, "dbname", dbname, config)
        self.set_benchmark_grid_config_param(benchmark_config, "experiment_name", experiment_name, config)
        self.set_benchmark_grid_config_param(benchmark_config, "cuda", cuda, config)
        self.set_benchmark_grid_config_param(benchmark_config, "debug", debug, config)

        return benchmark_config

    def run(self, dimensions, integrand, *,
            base_integrand_params, base_integrator_config=None,
            integrand_params_grid=None, integrator_config_grid=None,
            n_batch=100000, debug=True, cuda=0,
            sql_dtypes=None, dbname=None, experiment_name="benchmark", keep_history=False):
        """Run benchmarks over a grid of parameters for the integrator and the integrand."""

        if debug:
            set_benchmark_logger_debug(zunis_integration_level=logging.DEBUG,
                                       zunis_training_level=logging.DEBUG, zunis_level=logging.DEBUG)
        else:
            set_benchmark_logger(experiment_name)

        device = get_device(cuda_ID=cuda)

        if isinstance(dimensions, int):
            dimensions = [dimensions]
        assert isinstance(dimensions, Sequence) and all([isinstance(d, int) for d in dimensions]) and len(
            dimensions) > 0, \
            "argument dimensions must be an integer or a list of integers"

        if sql_dtypes is None:
            sql_dtypes = get_sql_types()

        if integrand_params_grid is None and integrator_config_grid is None and len(dimensions) == 1:
            result, integrator = self.benchmark_method(dimensions[0], integrand=integrand,
                                                       integrand_params=base_integrand_params,
                                                       integrator_config=base_integrator_config,
                                                       n_batch=n_batch, device=device, keep_history=keep_history)
            result = result.as_dataframe()

            if dbname is not None:
                append_dataframe_to_sqlite(result, dbname=dbname, tablename=experiment_name, dtypes=sql_dtypes)
            return result, integrator

        else:
            if integrand_params_grid is None:
                integrand_params_grid = dict()
            if integrator_config_grid is None:
                integrator_config_grid = dict()
            if base_integrator_config is None:
                base_integrator_config = get_default_integrator_config()

            integrator_config = deepcopy(base_integrator_config)
            integrand_params = deepcopy(base_integrand_params)

            benchmarks = self.generate_config_samples(dimensions, integrator_config_grid, integrand_params_grid)

            for d, integrator_config_update, integrand_params_update in benchmarks:
                logger.info("Benchmarking with:")
                logger.info(f"d = {d}")
                logger.info(f"integrator update: {integrator_config_update}")
                logger.info(f"integrand update: {integrand_params_update}")
                integrator_config.update(integrator_config_update)
                integrand_params.update(integrand_params_update)

                try:
                    result, _ = self.benchmark_method(d, integrand=integrand,
                                                      integrand_params=integrand_params,
                                                      integrator_config=integrator_config,
                                                      n_batch=n_batch, device=device,
                                                      keep_history=keep_history)
                    result = result.as_dataframe()


                except Exception as e:
                    logger.exception(e)
                    result = NestedMapping()
                    result["d"] = d
                    result.update(integrator_config)
                    result.update(integrand_params)
                    result["extra_data"] = e
                    result = result.as_dataframe()

                if dbname is not None:
                    append_dataframe_to_sqlite(result, dbname=dbname, tablename=experiment_name,
                                               dtypes=sql_dtypes)


class GridBenchmarker(Benchmarker):
    """Benchmark by sampling configurations like a grid. Can repeat each configurations multiple times"""

    def __init__(self, n_repeat=1):
        """

        Parameters
        ----------
        n: int
            Optional: How often the grid is sampled.
        """
        self.n_repeat = n_repeat

    def generate_config_samples(self, dimensions, integrator_grid, integrand_grid):
        integrator_grid_keys = integrator_grid.keys()
        integrand_grid_keys = integrand_grid.keys()
        integrator_grid_values = integrator_grid.values()
        integrand_grid_values = integrand_grid.values()

        for d in dimensions:
            # Need to reset our cartesian product iterator at each pass through
            integrand_full_grid = product(*integrand_grid_values)
            for integrand_update in integrand_full_grid:
                integrator_full_grid = product(*integrator_grid_values)
                for integrator_update in integrator_full_grid:
                    for _ in range(self.n_repeat):
                        integrator_config_update = dict(zip(integrator_grid_keys, integrator_update))
                        integrand_config_update = dict(zip(integrand_grid_keys, integrand_update))
                        yield d, integrator_config_update, integrand_config_update


class RandomHyperparameterBenchmarker(Benchmarker):
    """Benchmark by sampling integrator configurations randomly, a fixed number of times.
    Can repeat each configurations multiple times"""

    def __init__(self, n_samples=5, n_repeat=1):
        """

        Parameters
        ----------
        n_samples : int
            Number of random integrator configurations to draw

        n_repeat: int
            Optional: How often the grid is sampled.
        """
        self.n_samples = n_samples
        self.n_repeat = n_repeat

    def generate_config_samples(self, dimensions, integrator_grid, integrand_grid):
        integrand_grid_keys = integrand_grid.keys()
        integrand_grid_values = integrand_grid.values()

        for d in dimensions:
            # Need to reset our cartesian product iterator at each pass through
            integrand_full_grid = product(*integrand_grid_values)
            for integrand_update in integrand_full_grid:
                for _ in range(self.n_samples):
                    integrator_config_update = dict()
                    for param_name, param_grid in integrator_grid.items():
                        integrator_config_update[param_name] = random.choice(param_grid)
                    integrand_config_update = dict(zip(integrand_grid_keys, integrand_update))
                    for _ in range(self.n_repeat):
                        yield d, integrator_config_update, integrand_config_update


class SequentialIntegratorBenchmarker(Benchmarker):
    """Benchmark by going through a list of full integrator configurations (as opposed to going through parameters
    in the configuration independently) and scan over a grid of possible integrands.
    Can repeat each configurations multiple times"""

    def __init__(self, n_repeat=1):
        """

        Parameters
        ----------
        n: int
            Optional: How often the grid is sampled.
        """
        self.n_repeat = n_repeat

    def generate_config_samples(self, dimensions, integrator_grid, integrand_grid):
        """Sample over dimensions, integrator and integrand configurations from lists of possible option values.

        Parameters
        ----------
        dimensions : List[int]
            list of dimensions to sample from
        integrator_grid : Dict[str, List[Any]]
            mapping (option name) -> (list of values) for the integrator
        integrand_grid: Dict[str, List[Any]]
            mapping (option name) -> (list of values) for the integrand

        Yields
        -------
            Tuple[int, Dict[str, Any], Dict[str, Any]]
                triplets (d, integrator_config, integrand_params) that can be used to sample configurations

        Notes
        -----

        The integrator grid is interpreted as a sequence:

        .. code-block:: python

            integrator_grid = {
              param1: [v1, v2, ..., vn]
              param2: [w1, w2, ..., wn]
            }

        is scanned as the list of configurations

        .. code-block:: python

            config1 = {param1: v1, param2: w1}
            config2 = {param1: v2, param2: w2}
        """

        grid_lengths = list(set([len(param) for param in integrator_grid.values()]))
        # We use max to account for the possibility of an empty grid
        assert max(1, len(grid_lengths)) == 1, "All integrator parameter lists must have the same length"
        try:
            grid_length = grid_lengths[0]
        except IndexError:
            # If no grid, use 1 to evaluate the default configuration once
            grid_length = 1

        integrand_grid_keys = integrand_grid.keys()
        integrand_grid_values = integrand_grid.values()

        for d in dimensions:
            # Need to reset our cartesian product iterator at each pass through
            integrand_full_grid = product(*integrand_grid_values)
            for integrand_update in integrand_full_grid:
                for i in range(grid_length):
                    integrator_config_update = dict()
                    for param_name, param_grid in integrator_grid.items():
                        integrator_config_update[param_name] = param_grid[i]
                    integrand_config_update = dict(zip(integrand_grid_keys, integrand_update))
                    for _ in range(self.n_repeat):
                        yield d, integrator_config_update, integrand_config_update


class SequentialBenchmarker(Benchmarker):
    """Benchmark by going through a list of integrands and matching integrator configurations
    (as opposed to scanning through parameters for either).

    This is intended for benchmarking integrands on "optimal configurations" found through previous
    hyper parameter searching.
    Can repeat each configurations multiple times"""

    def __init__(self, n_repeat=1):
        """

        Parameters
        ----------
        n: int
            Optional: How often the grid is sampled.
        """
        self.n_repeat = n_repeat

    def generate_config_samples(self, dimensions, integrator_grid, integrand_grid):
        """Sample over dimensions, integrator and integrand configurations from lists of possible option values.

        Parameters
        ----------
        dimensions : List[int]
            list of dimensions to sample from
        integrator_grid : Dict[str, List[Any]]
            mapping (option name) -> (list of values) for the integrator
        integrand_grid: Dict[str, List[Any]]
            mapping (option name) -> (list of values) for the integrand

        Yields
        -------
            Tuple[int, Dict[str, Any], Dict[str, Any]]
                triplets (d, integrator_config, integrand_params) that can be used to sample configurations

        Notes
        -----

        The integrator grid is interpreted as a sequence:

        .. code-block:: python

            integrator_grid = {
              param1: [v1, v2, ..., vn]
              param2: [w1, w2, ..., wn]
            }

        is scanned as the list of configurations

        .. code-block:: python

            config1 = {param1: v1, param2: w1}
            config2 = {param1: v2, param2: w2}
            ...
        """

        integrator_grid_lengths = list(set([len(param) for param in integrator_grid.values()]))
        integrand_grid_lengths = list(set([len(param) for param in integrand_grid.values()]))

        assert \
            len(integrator_grid_lengths) == 1 and \
            len(integrand_grid_lengths) == 1 and \
            integrand_grid_lengths[0] == integrand_grid_lengths[0] == len(dimensions), \
            "The length of all parameter grids must match (integrator, integrand, dimensions)"

        grid_length = integrand_grid_lengths[0]

        for i in range(grid_length):
            integrator_config_update = dict()
            integrand_config_update = dict()
            d = dimensions[i]
            for param_name, param_grid in integrator_grid.items():
                integrator_config_update[param_name] = param_grid[i]
            for param_name, param_grid in integrand_grid.items():
                integrand_config_update[param_name] = param_grid[i]
            for _ in range(self.n_repeat):
                yield d, integrator_config_update, integrand_config_update
