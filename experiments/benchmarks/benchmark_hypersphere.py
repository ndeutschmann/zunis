import sys
from functools import partial
import logging
import pandas as pd
import torch
from utils.integrands import RegulatedHyperSphereIntegrand
from utils.flat_integrals import validate_known_integrand_flat
from utils.integrator_integrals import validate_integral_integrator
from zunis.integration import Integrator

from zunis import logger_integration, logger_training

#############################################################
#       DEBUG FLAG: set to False to log and save to file
#############################################################
debug = True
#############################################################


if debug:
    logging.basicConfig(level=logging.DEBUG, stream=sys.stdout, format='%(levelname)s: %(message)s [%(name)s]')
else:
    logging.basicConfig(level=logging.INFO, filename="benchmark_hypersphere.log", filemode="w",
                        format='%(levelname)s: %(message)s [%(name)s]')
logger_integration.setLevel(logging.WARNING)
logger_training.setLevel(logging.WARNING)
logger = logging.getLogger("benchmark_hypersphere")


if torch.has_cuda:
    cuda_ID = 0
    device = torch.device(f"cuda:{cuda_ID}")
    logger.warning(f"Using CUDA:{cuda_ID}")
else:
    device = torch.device("cpu")
    logger.warning("Using CPU")


def benchmark_hyperrect(d, r=0.5, n_batch=100000, lr=1.e-3):
    logger.debug("=" * 72)
    logger.info(f"Benchmarking the hypersphere integral with d={d} and r={r:.2e}")
    logger.debug("=" * 72)
    sphere = RegulatedHyperSphereIntegrand(d=d, r=r, c=0.5)
    optim = partial(torch.optim.Adam, lr=lr)
    integrator = Integrator(f=sphere, d=d, device=device,
                            trainer_options={"n_epochs": 10, "minibatch_size": 20000, "optim": optim})
    logger.info("Running integrator")
    integrator_result = validate_integral_integrator(sphere, integrator, n_batch=n_batch, n_survey_steps=10)
    logger.debug("=" * 72)
    logger.info("Running flat sampler")
    flat_result = validate_known_integrand_flat(sphere, d=d, n_batch=n_batch)
    logger.debug("=" * 72)
    integrator_result["speedup"] = (flat_result["value_std"] / integrator_result["value_std"]) ** 2
    logger.info(f"speedup: {integrator_result['speedup']}")
    logger.debug("=" * 72)
    logger.debug(" " * 72)
    integrator_result["d"] = d
    integrator_result["r"] = r
    return integrator_result


if __name__ == "__main__":
    results = pd.DataFrame()
    n_batch = 100000
    for d in [2, 4, 6, 8, 10]:
        result = benchmark_hyperrect(d, 0.49, n_batch=n_batch)
        results = pd.concat([results, result.as_dataframe()], ignore_index=True)

    print(results)
    if not debug:
        results.to_csv("benchmark_hypersphere.csv", mode="w")
