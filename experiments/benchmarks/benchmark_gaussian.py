"""Comparing ZuNIS to VEGAS on gaussian integrals (VEGAS expected to be better)"""

import vegas
import click
import logging
import pandas as pd
import torch

from utils.command_line_tools import PythonLiteralOption
from utils.integrands.gaussian import DiagonalGaussianIntegrand
from utils.logging import get_benchmark_logger, get_benchmark_logger_debug
from utils.torch_utils import get_device
from utils.integrator_integrals import evaluate_integral_integrator
from utils.vegas_integrals import evaluate_integral_vegas
from utils.flat_integrals import evaluate_integral_flat
from utils.integral_validation import compare_integral_result
import utils.config as conf
from utils.data_storage.dataframe2sql import append_dataframe_to_sqlite
from zunis.integration import Integrator


def benchmark_gaussian(d, s=0.3, n_batch=100000, logger=None, device=torch.device("cpu")):
    if logger is None:
        get_benchmark_logger_debug("benchmark_gaussian")
    logger.info(f"Benchmarking a gaussian with d={d} and s={s:.1f}")

    gaussian = DiagonalGaussianIntegrand(d=d, device=device, s=s)

    @vegas.batchintegrand
    def vgaussian(x):
        return gaussian(torch.tensor(x).to(device)).cpu().numpy()

    integrator_config = conf.get_default_integrator_config()
    integrator_config["n_points_survey"] = 100000
    integrator_config["n_bins"] = 50
    integrator_config["n_epochs"] = 50
    integrator_args = conf.create_integrator_args(integrator_config)

    integrator = Integrator(d=d, f=gaussian, device=device, **integrator_args)
    vintegrator = vegas.Integrator([[0, 1]] * d, max_nhcube=1)

    integrator_result = evaluate_integral_integrator(gaussian, integrator, n_batch=n_batch)
    vegas_result = evaluate_integral_vegas(vgaussian, vintegrator, n_batch=n_batch,
                                           n_batch_survey=integrator_args["n_points_survey"])
    flat_result = evaluate_integral_flat(gaussian, d, n_batch=n_batch, device=device)

    result = compare_integral_result(integrator_result, vegas_result, sigma_cutoff=3).as_dataframe()
    result["flat_variance_ratio"] = (flat_result["value_std"] / result["value_std"]) ** 2

    result["d"] = d
    result["s"] = s

    result = result.merge(
        integrator_config.as_dataframe(),
        left_index=True,
        right_index=True
    )

    return result


def run_benchmark(dimensions, sigmas, debug, cuda):
    if debug:
        logger = get_benchmark_logger_debug("benchmark_gaussian", zunis_integration_level=logging.DEBUG,
                                            zunis_training_level=logging.DEBUG, zunis_level=logging.DEBUG)
    else:
        logger = get_benchmark_logger("benchmark_gaussian")

    device = get_device(cuda_ID=cuda)

    results = pd.DataFrame()
    for d in dimensions:
        for s in sigmas:
            result = benchmark_gaussian(d, s, logger=logger, device=device)
            results = pd.concat([results, result], ignore_index=True)

    print(results)
    if not debug:
        sql_dtypes = conf.loaders.get_sql_types()
        append_dataframe_to_sqlite(results, dbname="benchmarks.db", tablename="gaussian", dtypes=sql_dtypes)


cli = click.Command("cli", callback=run_benchmark, params=[
    PythonLiteralOption(["--dimensions"], default=[2, 4, 6, 8, 10]),
    PythonLiteralOption(["--sigmas"], default=[0.5, 0.3, 0.1]),
    click.Option(["--debug/--no-debug"], default=True),
    click.Option(["--cuda"], default=0, type=int)
])

if __name__ == '__main__':
    cli()
