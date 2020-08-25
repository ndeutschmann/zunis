"""Comparing ZuNIS to VEGAS on camel integrals"""

import vegas
from functools import partial
import pandas as pd
import torch
from utils.integrands.gaussian import CamelIntegrand
from utils.logging import get_benchmark_logger, get_benchmark_logger_debug
from utils.torch_utils import get_device
from utils.integrator_integrals import evaluate_integral_integrator
from utils.vegas_integrals import evaluate_integral_vegas
from utils.flat_integrals import evaluate_integral_flat
from utils.integral_validation import compare_integral_result
from zunis.integration import Integrator

#############################################################
#       DEBUG FLAG: set to False to log and save to file
#############################################################
debug = True
#############################################################


if debug:
    logger = get_benchmark_logger_debug("benchmark_camel")
else:
    logger = get_benchmark_logger("benchmark_camel")

device = get_device(cuda_ID=0)


def benchmark_camel(d, s=0.3):
    logger.info(f"Benchmarking a camel with d={d} and s={s:.1f}")
    camel = CamelIntegrand(d=d, device=device, s1=s)

    @vegas.batchintegrand
    def vcamel(x):
        return camel(torch.tensor(x).to(device)).cpu()

    integrator = Integrator(d=d, f=camel, device=device, flow_options={"masking_options": {"repetitions": 2},
                                                                       "cell_params": {"n_bins": 50}})
    vintegrator = vegas.Integrator([[0, 1]] * d, max_nhcube=1)

    integrator_result = evaluate_integral_integrator(camel, integrator, n_batch=100000,
                                                     survey_args={"n_points": 100000,
                                                                  "n_epochs": 100})
    vegas_result = evaluate_integral_vegas(vcamel, vintegrator, n_batch=100000, n_batch_survey=100000)
    flat_result = evaluate_integral_flat(camel, d, n_batch=100000, device=device)
    result = compare_integral_result(integrator_result, vegas_result, sigma_cutoff=3).as_dataframe()
    result["flat_variance_ratio"] = (flat_result["value_std"] / result["value_std"]) ** 2

    result["d"] = d
    result["s"] = s

    return result


if __name__ == "__main__":
    results = pd.DataFrame()
    for d in [2, 4, 6, 8, 10]:
        for s in [0.5, 0.3, 0.1]:
            result = benchmark_camel(d, s)
            results = pd.concat([results, result], ignore_index=True)

    print(results)
    if not debug:
        results.to_pickle("benchmark_camel.bz2")
