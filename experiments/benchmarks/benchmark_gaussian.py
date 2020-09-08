"""Comparing ZuNIS to VEGAS on gaussian integrals (VEGAS expected to be better)"""

import click

from utils.command_line_tools import PythonLiteralOption
from utils.benchmark import run_benchmark_grid_vegas
from utils.config.loaders import get_default_integrator_config, get_sql_types
from utils.integrands.gaussian import DiagonalGaussianIntegrand


def benchmark_gaussian(dimensions=(2, 4, 6, 8), sigmas=(0.1, 0.3, 0.5, 0.7), db="benchmarks.db", debug=True, cuda=0):
    base_integrand_params = {
        "s1": 0.5,
        "s2": 0.5,
        "norm1": 1.,
        "norm2": 1.
    }
    integrands_params_grid = {
        "s": sigmas
    }

    base_integrator_config = get_default_integrator_config()
    dtypes = get_sql_types()

    if debug:
        base_integrator_config["n_epochs"] = 1
        base_integrator_config["n_iter"] = 1

    run_benchmark_grid_vegas(dimensions=dimensions, integrand=DiagonalGaussianIntegrand,
                             base_integrand_params=base_integrand_params,
                             base_integrator_config=base_integrator_config,
                             integrand_params_grid=integrands_params_grid, integrator_config_grid=None,
                             n_batch=100000, debug=debug, cuda=cuda, sql_dtypes=dtypes,
                             dbname=db, experiment_name="gaussian")


cli = click.Command("cli", callback=benchmark_gaussian, params=[
    PythonLiteralOption(["--dimensions"], default=[2, 4, 6, 8, 10]),
    PythonLiteralOption(["--sigmas"], default=[0.5, 0.3, 0.1]),
    click.Option(["--debug/--no-debug"], default=True),
    click.Option(["--cuda"], default=0, type=int),
    click.Option(["--db"], default="benchmarks.db", type=str)
])

if __name__ == '__main__':
    cli()
