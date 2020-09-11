"""Comparing ZuNIS to VEGAS on camel integrals"""
import click
import numpy as np
import os

from utils.command_line_tools import PythonLiteralOption
from utils.benchmark import run_benchmark_grid_vegas
from utils.config.loaders import get_default_integrator_config, get_sql_types
from utils.integrands.gaussian import SymmetricCamelIntegrand
from utils.config.configuration import Configuration


def benchmark_camel(dimensions=(2, 4, 6, 8), sigmas=(0.1, 0.3, 0.5, 0.7), db="benchmarks.db",
                    experiment_name="camel-grid-run", debug=True, cuda=0,
                    config=None):
    base_integrand_params = {
        "s": 0.5,
        "norm": 1.
    }

    integrator_config_grid = None

    base_integrator_config = get_default_integrator_config()
    dtypes = get_sql_types()

    # TODO this should go in utils.benchmark.run_benchmark_grid
    if config is not None:
        config = Configuration.from_yaml(config, check=False)
        if "dimensions" in config:
            dimensions = config["dimensions"]
        if "sigmas" in config:
            sigmas = config["sigmas"]
        if "database" in config:
            db = config["database"]
        if "cuda" in config:
            cuda = config["cuda"]
        if "experiment_name" in config:
            experiment_name = config["experiment_name"]

        integrand_params_grid = {
            "s": sigmas
        }

        if "integrator_config" in config:
            base_integrator_config.update(config["integrator_config"])
        if "integrand_params" in config:
            base_integrand_params.update(config["integrand_params"])
        if "integrator_grid" in config:
            integrator_config_grid = config["integrator_grid"]
        if "integrand_grid" in config:
            integrand_params_grid.update(config["integrand_grid"])

    run_benchmark_grid_vegas(dimensions=dimensions, integrand=SymmetricCamelIntegrand,
                             base_integrand_params=base_integrand_params,
                             base_integrator_config=base_integrator_config,
                             integrand_params_grid=integrand_params_grid,
                             integrator_config_grid=integrator_config_grid,
                             n_batch=100000, debug=debug, cuda=cuda, sql_dtypes=dtypes,
                             dbname=db, experiment_name=experiment_name)


cli = click.Command("cli", callback=benchmark_camel, params=[
    PythonLiteralOption(["--dimensions"], default=list(range(2, 17, 2))),
    PythonLiteralOption(["--sigmas"], default=list(np.linspace(0.1, 1., 30))),
    click.Option(["--debug/--no-debug"], default=True),
    click.Option(["--cuda"], default=0, type=int),
    click.Option(["--db"], default="benchmarks.db", type=str),
    click.Option(["--experiment_name"], default="camel-grid-run", type=str),
    click.Option(["--config"], default=None, type=str)
])

if __name__ == '__main__':
    cli()
