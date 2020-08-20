import logging
import sys
import torch
from functools import partial

from zunis.integration import Integrator
from utils.integrands.volume import HyperrectangleVolumeIntegrand, HypersphereVolumeIntegrand
from utils.integrator_integrals import validate_integral_integrator

from zunis import logger_integration, logger_training

logging.basicConfig(level=logging.INFO, stream=sys.stdout, format='%(levelname)s: %(message)s [%(name)s]')
logger_integration.setLevel(logging.WARNING)
logger_training.setLevel(logging.WARNING)
logger = logging.getLogger("integrator_validation")

if torch.has_cuda:
    cuda_ID = 0
    device = torch.device(f"cuda:{cuda_ID}")
    logger.warning(f"Using CUDA:{cuda_ID}")
else:
    device = torch.device("cpu")
    logger.warning("Using the CPU")


def run_hyperrect_test(d):
    return
    logger.info(f"Running hyperrectangle test in {d} dimensions")
    half = HyperrectangleVolumeIntegrand(d=d)
    integrator = Integrator(d=d, f=half, device=device)
    result = validate_integral_integrator(half, integrator, n_batch=100000)
    print("")


def run_hypersphere_test(d):
    logger.info(f"Running hypersphere test in {d} dimensions")
    sphere = HypersphereVolumeIntegrand(d=d, r=0.3, c=0.5, device=device)
    integrator = Integrator(d=d, f=sphere, device=device, trainer_options={"optim": partial(torch.optim.Adam, lr=1.e-4)})
    result = validate_integral_integrator(sphere, integrator, n_batch=100000)
    print("")
    return result


if __name__ == "__main__":
    for d in [2, 3, 4]:
        run_hyperrect_test(d)

    for d in [2, 3, 4]:
        run_hypersphere_test(d)
