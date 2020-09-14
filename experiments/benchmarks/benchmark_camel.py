"""Comparing ZuNIS to VEGAS on camel integrals"""
import click

from utils.command_line_tools import PythonLiteralOption
from utils.benchmark import run_benchmark_grid_vegas, set_benchmark_grid_config
from utils.config.loaders import get_default_integrator_config, get_sql_types
from utils.integrands.gaussian import SymmetricCamelIntegrand


def benchmark_camel(dimensions=None, sigmas=None, db=None,
                    experiment_name=None, debug=None, cuda=None, keep_history=None,
                    config=None):
    dtypes = get_sql_types()

    # Integrand specific defaults
    base_integrand_params = {
        "s": 0.3,
        "norm": 1.
    }
    benchmark_config = set_benchmark_grid_config(config=config, dimensions=dimensions, keep_history=keep_history,
                                                 dbname=db, experiment_name=experiment_name, cuda=cuda, debug=debug,
                                                 base_integrand_params=base_integrand_params)

    # Integrand specific CLI argument mapped to standard API
    if sigmas is not None:
        benchmark_config["integrand_params_grid"]["s"] = sigmas

    run_benchmark_grid_vegas(integrand=SymmetricCamelIntegrand, sql_dtypes=dtypes,
                             **benchmark_config)


cli = click.Command("cli", callback=benchmark_camel, params=[
    PythonLiteralOption(["--dimensions"], default=None),
    PythonLiteralOption(["--sigmas"], default=None),
    click.Option(["--debug/--no-debug"], default=True),
    click.Option(["--cuda"], default=None, type=int),
    click.Option(["--db"], default=None, type=str),
    click.Option(["--experiment_name"], default=None, type=str),
    click.Option(["--config"], default=None, type=str)
])

if __name__ == '__main__':
    cli()
