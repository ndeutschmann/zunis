"""Comparing ZuNIS to VEGAS on camel integrals"""
import click

from utils.benchmark.vegas_benchmarks import VegasSequentialBenchmarker
from utils.config.loaders import get_sql_types
from utils.integrands.camel import SymmetricCamelIntegrand


def benchmark_camel(db=None, experiment_name=None, debug=None, cuda=None, keep_history=None, config=None):
    dtypes = get_sql_types()

    # Integrand specific defaults
    base_integrand_params = {
        "s": 0.3,
        "norm": 1.
    }

    benchmarker = VegasSequentialBenchmarker()

    benchmark_config = benchmarker.set_benchmark_grid_config(config=config,
                                                             keep_history=keep_history,
                                                             dbname=db, experiment_name=experiment_name, cuda=cuda,
                                                             debug=debug,
                                                             base_integrand_params=base_integrand_params)

    # Integrand specific CLI argument mapped to standard API

    benchmarker.run(integrand=SymmetricCamelIntegrand, sql_dtypes=dtypes,
                    **benchmark_config)


cli = click.Command("cli", callback=benchmark_camel, params=[
    click.Option(["--debug/--no-debug"], default=None, type=bool),
    click.Option(["--cuda"], default=None, type=int),
    click.Option(["--db"], default=None, type=str),
    click.Option(["--experiment_name"], default=None, type=str),
    click.Option(["--config"], default=None, type=str)
])

if __name__ == '__main__':
    cli()
