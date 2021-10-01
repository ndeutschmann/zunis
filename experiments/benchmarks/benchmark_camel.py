"""Comparing ZuNIS to VEGAS on camel integrals"""
import click

from utils.benchmark.vegas_benchmarks import VegasSequentialIntegratorBenchmarker
from utils.command_line_tools import PythonLiteralOption
from utils.config.loaders import get_sql_types
from utils.integrands.camel import KnownSymmetricCamelIntegrand


def benchmark_camel(dimensions=None, sigmas=None, db=None,
                    experiment_name=None, debug=None, cuda=None, keep_history=None,
                    config=None,
                    n_repeat=10,
                    stratified=False):
    dtypes = get_sql_types()

    # Integrand specific defaults
    base_integrand_params = {
        "s": 0.3,
        "norm": 1.
    }

    benchmarker = VegasSequentialIntegratorBenchmarker(n_repeat=n_repeat, stratified=stratified, benchmark_time=False)

    benchmark_config = benchmarker.set_benchmark_grid_config(config=config, dimensions=dimensions,
                                                             keep_history=keep_history,
                                                             dbname=db, experiment_name=experiment_name, cuda=cuda,
                                                             debug=debug,
                                                             base_integrand_params=base_integrand_params)

    # Integrand specific CLI argument mapped to standard API
    if sigmas is not None:
        benchmark_config["integrand_params_grid"]["s"] = sigmas

    benchmarker.run(integrand=KnownSymmetricCamelIntegrand, sql_dtypes=dtypes,
                    **benchmark_config)


cli = click.Command("cli", callback=benchmark_camel, params=[
    PythonLiteralOption(["--dimensions"], default=None),
    PythonLiteralOption(["--sigmas"], default=None),
    click.Option(["--debug/--no-debug"], default=None, type=bool),
    click.Option(["--cuda"], default=None, type=int),
    click.Option(["--db"], default=None, type=str),
    click.Option(["--experiment_name"], default=None, type=str),
    click.Option(["--config"], default=None, type=str),
    click.Option(["--n_repeat"], default=10, type=int),
    click.Option(["--stratified"], is_flag=True)
])

if __name__ == '__main__':
    cli()
