import click

from utils.command_line_tools import PythonLiteralOption
from utils.benchmark import run_benchmark_grid_known_integrand
from utils.integrands.volume import RegulatedHyperSphericalCamel
from utils.config.loaders import get_default_integrator_config, get_sql_types


def hypercamel_benchmark(dimensions=(2, 4, 6, 8), rs=(0.25,), regs=(1.e-6,), debug=True, cuda=0):
    base_integrand_params = {
        "r1": 0.25,
        "r2": 0.25,
        "reg": 1.e-6
    }
    integrands_params_grid = {
        "r1": rs,
        "r2": rs,
        "reg": regs
    }

    base_integrator_config = get_default_integrator_config()
    dtypes = get_sql_types()

    if debug:
        base_integrator_config["n_epochs"] = 1
        base_integrator_config["n_iter"] = 1

    run_benchmark_grid_known_integrand(dimensions=dimensions, integrand=RegulatedHyperSphericalCamel,
                                       base_integrand_params=base_integrand_params,
                                       base_integrator_config=base_integrator_config,
                                       integrand_params_grid=integrands_params_grid, integrator_config_grid=None,
                                       n_batch=100000, debug=debug, cuda=cuda, sql_dtypes=dtypes,
                                       dbname="benchmarks-debug.db", experiment_name="hypercamel")


cli = click.Command("cli", callback=hypercamel_benchmark, params=[
    PythonLiteralOption(["--dimensions"], default=[2]),
    PythonLiteralOption(["--fracs"], default=[0.3, 0.5, 0.7]),
    click.Option(["--debug/--no-debug"], default=True),
    click.Option(["--cuda"], default=0, type=int)
])

if __name__ == '__main__':
    cli()
