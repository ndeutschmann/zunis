import sys
import logging
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from utils.integrands.volume import HypersphereVolumeIntegrand
from utils.flat_integrals import validate_known_integrand_flat


from zunis import logger_integration, logger_training

logging.basicConfig(level=logging.INFO, stream=sys.stdout, format='%(levelname)s: %(message)s [%(name)s]')
logger_integration.setLevel(logging.WARNING)
logger_training.setLevel(logging.WARNING)
logger = logging.getLogger("flat_hypersphere_validation")

results = pd.DataFrame()
n_batch = 10000000

dimensions = range(2,10)
radii = np.linspace(0.48,0.2,20)
centers = [0.5]


if torch.has_cuda:
    device = torch.device("cuda:7")
    logger.warning("Using CUDA:7")
else:
    device = torch.device("cpu")
    logger.warning("Using CPU")

for d in dimensions:
    for r in radii:
        for c in centers:
            logger.info(f"Validating flat hypersphere for d={d} and r={r:.2f}")
            f = HypersphereVolumeIntegrand(d, r, c, device=device)
            result = validate_known_integrand_flat(f, d, n_batch, device=device)
            result["dim"] = d
            result["radius"] = r
            result["center"] = c
            results = pd.concat([results, result.as_dataframe()], ignore_index=True)
            print("")

fig, ax = plt.subplots()
ax.scatter(results["dim"], results["percent_difference"])
ax.set_xlabel("Dimension")
ax.set_ylabel("Integral uncertainty")
plt.show()

fig, ax = plt.subplots()
for d in dimensions:
    filtered_result = results.loc[results["dim"] == d].sort_values("radius")
    ax.plot(filtered_result["radius"],filtered_result["percent_difference"])
ax.set_xlabel("Radius")
ax.set_ylabel("Integral uncertainty")
plt.show()


print(f"We found {100 * results['match'].mean()}% of matching tests")

