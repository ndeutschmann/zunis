"""Comparing ZuNIS to VEGAS on gaussian integrals"""
import click

from utils.benchmark.vegas_benchmarks import VegasRandomHPBenchmarker
from utils.command_line_tools import PythonLiteralOption
from utils.config.loaders import get_sql_types
from utils.integrands.gaussian import DiagonalGaussianIntegrand


def benchmark_gaussian(dimensions=None, sigmas=None, db=None,
                    experiment_name=None, debug=None, cuda=None, keep_history=None,
                    config=None,
                    n_search=10,
                    stratified=True):
    dtypes = get_sql_types()

    # Integrand specific defaults
    base_integrand_params = {
        "s": 0.3,
        "norm": 1.
    }

    benchmarker = VegasRandomHPBenchmarker(n=n_search, stratified=stratified)

    benchmark_config = benchmarker.set_benchmark_grid_config(config=config, dimensions=dimensions,
                                                             keep_history=keep_history,
                                                             dbname=db, experiment_name=experiment_name, cuda=cuda,
                                                             debug=debug,
                                                             base_integrand_params=base_integrand_params)

    # Integrand specific CLI argument mapped to standard API
    if sigmas is not None:
        benchmark_config["integrand_params_grid"]["s"] = sigmas

    benchmarker.run(integrand=DiagonalGaussianIntegrand, sql_dtypes=dtypes,
                    **benchmark_config)


cli = click.Command("cli", callback=benchmark_gaussian, params=[
    PythonLiteralOption(["--dimensions"], default=None),
    PythonLiteralOption(["--sigmas"], default=None),
    click.Option(["--debug/--no-debug"], default=None, type=bool),
    click.Option(["--cuda"], default=None, type=int),
    click.Option(["--db"], default=None, type=str),
    click.Option(["--experiment_name"], default=None, type=str),
    click.Option(["--config"], default=None, type=str),
    click.Option(["--n_search"], default=10, type=int),
    click.Option(["--stratified"], is_flag=True)
])

if __name__ == '__main__':
    cli()
