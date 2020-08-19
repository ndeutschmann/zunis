import sys
import logging
import pandas as pd
import torch
from utils.integrands import HyperrectangleVolumeIntegrand
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
    logging.basicConfig(level=logging.INFO, filename="benchmark_hyperrect.log", filemode="w",
                        format='%(levelname)s: %(message)s [%(name)s]')
logger_integration.setLevel(logging.WARNING)
logger_training.setLevel(logging.WARNING)
logger = logging.getLogger("benchmark_hyperrect")

if torch.has_cuda:
    cuda_ID = 0
    device = torch.device(f"cuda:{cuda_ID}")
    logger.warning(f"Using CUDA:{cuda_ID}")

else:
    device = torch.device("cpu")
    logger.warning("Using CPU")


def benchmark_hyperrect(d, frac=0.5, n_batch=100000):
    logger.debug("=" * 72)
    logger.info(f"Benchmarking the hyperrectangle integral with d={d} and frac={frac:.2e}")
    logger.debug("=" * 72)
    hrect = HyperrectangleVolumeIntegrand(d=d, frac=frac)
    integrator = Integrator(f=hrect, d=d, device=device, trainer_options={"minibatch_size": 20000})
    logger.info("Running integrator")
    integrator_result = validate_integral_integrator(hrect, integrator, n_batch=n_batch)
    logger.debug("=" * 72)
    logger.info("Running flat sampler")
    flat_result = validate_known_integrand_flat(hrect, d=d, n_batch=n_batch)
    logger.debug("=" * 72)
    integrator_result["speedup"] = (flat_result["value_std"] / integrator_result["value_std"]) ** 2
    logger.info(f"speedup: {integrator_result['speedup']}")
    logger.debug("=" * 72)
    logger.debug(" " * 72)
    integrator_result["d"] = d
    integrator_result["frac"] = frac
    return integrator_result


if __name__ == "__main__":
    results = pd.DataFrame()
    for r in [0.3, 0.5, 0.8]:
        result = benchmark_hyperrect(2, r)
        results = pd.concat([results, result.as_dataframe()], ignore_index=True)

    print(results)
    if not debug:
        results.to_csv("benchmark_hyperrect.csv", mode="w")
