import click

from utils.command_line_tools import PythonLiteralOption
from utils.benchmark import run_benchmark_grid_known_integrand
from utils.integrands.volume import HyperrectangleVolumeIntegrand
from utils.config.loaders import get_sql_types
from zunis.utils.config.loaders import get_default_integrator_config


def hyperrect_benchmark(dimensions=(2, 4, 6, 8), fracs=(0.3, 0.5, 0.7), debug=True, cuda=0):
    base_integrand_params = {
        "frac": 0.5
    }
    integrands_params_grid = {
        "frac": fracs
    }

    base_integrator_config = get_default_integrator_config()
    dtypes = get_sql_types()

    if debug:
        base_integrator_config["n_epochs"] = 1
        base_integrator_config["n_iter"] = 1

    run_benchmark_grid_known_integrand(dimensions=dimensions, integrand=HyperrectangleVolumeIntegrand,
                                       base_integrand_params=base_integrand_params,
                                       base_integrator_config=base_integrator_config,
                                       integrand_params_grid=integrands_params_grid, integrator_config_grid=None,
                                       n_batch=100000, debug=debug, cuda=cuda, sql_dtypes=dtypes,
                                       dbname="benchmarks-debug.db", experiment_name="hyperrect")


cli = click.Command("cli", callback=hyperrect_benchmark, params=[
    PythonLiteralOption(["--dimensions"], default=[2]),
    PythonLiteralOption(["--fracs"], default=[0.3, 0.5, 0.7]),
    click.Option(["--debug/--no-debug"], default=True),
    click.Option(["--cuda"], default=0, type=int)
])

if __name__ == '__main__':
    cli()
