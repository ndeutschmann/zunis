import click

from utils.command_line_tools import PythonLiteralOption
from utils.benchmark import run_benchmark_grid_known_integrand
from utils.integrands.volume import RegulatedHyperSphereIntegrand
from utils.config.loaders import get_sql_types
from zunis.utils.config.loaders import get_default_integrator_config


def hypersphere_benchmark(dimensions=(2, 4, 6, 8), rs=(0.3, 0.4, 0.5), cs=(0.5,), regs=(1.e-6,), debug=True, cuda=0):
    base_integrand_params = {
        "r": 0.49,
        "c": 0.5,
        "reg": 1.e-6
    }
    integrands_params_grid = {
        "r": rs,
        "c": cs,
        "reg": regs
    }

    base_integrator_config = get_default_integrator_config()
    dtypes = get_sql_types()

    if debug:
        base_integrator_config["n_epochs"] = 1
        base_integrator_config["n_iter"] = 1

    run_benchmark_grid_known_integrand(dimensions=dimensions, integrand=RegulatedHyperSphereIntegrand,
                                       base_integrand_params=base_integrand_params,
                                       base_integrator_config=base_integrator_config,
                                       integrand_params_grid=integrands_params_grid, integrator_config_grid=None,
                                       n_batch=100000, debug=debug, cuda=cuda, sql_dtypes=dtypes,
                                       dbname="benchmarks.db", experiment_name="hypersphere")


cli = click.Command("cli", callback=hypersphere_benchmark, params=[
    PythonLiteralOption(["--dimensions"], default=[2]),
    PythonLiteralOption(["--rs"], default=[0.3, 0.49]),
    PythonLiteralOption(["--cs"], default=[0.5]),
    PythonLiteralOption(["--regs"], default=[1.e-6]),
    click.Option(["--debug/--no-debug"], default=True),
    click.Option(["--cuda"], default=0, type=int)
])

if __name__ == '__main__':
    cli()
