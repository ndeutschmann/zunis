from functools import partial
import pandas as pd
import torch
from utils.integrands.volume import RegulatedHyperSphereIntegrand
from utils.benchmark import benchmark_known_integrand
from utils.logging import get_benchmark_logger, get_benchmark_logger_debug
from utils.torch_utils import get_device
from zunis.integration import Integrator

#############################################################
#       DEBUG FLAG: set to False to log and save to file
#############################################################
debug = True
#############################################################


if debug:
    logger = get_benchmark_logger_debug("benchmark_hypersphere")
else:
    logger = get_benchmark_logger("benchmark_hypersphere")

device = get_device(cuda_ID=0)


def benchmark_hypersphere(d, r=0.5, n_batch=100000, lr=1.e-3):
    """Run a benchmark on a hypersphere integrand"""
    logger.debug("=" * 72)
    logger.info(f"Benchmarking the hypersphere integral with d={d} and r={r:.2e}")
    integrand_params = {
        "r": r,
        "c": 0.5,
        "reg": 1.e-6
    }
    integrand = RegulatedHyperSphereIntegrand(d, device=device, **integrand_params)
    optim = partial(torch.optim.Adam, lr=lr)
    integrator = Integrator(f=integrand, d=d, device=device, trainer_options={"minibatch_size": 20000, "optim": optim})

    integrator_result = benchmark_known_integrand(d, integrand, integrator, n_batch=n_batch,
                                                  integrand_params=integrand_params, device=device)

    return integrator_result


if __name__ == "__main__":
    results = pd.DataFrame()
    n_batch = 100000
    for d in [2, 4, 6, 8, 10]:
        result = benchmark_hypersphere(d, 0.49, n_batch=n_batch)
        results = pd.concat([results, result.as_dataframe()], ignore_index=True)

    print(results)
    if not debug:
        results.to_pickle("benchmark_hypersphere.bz2")
