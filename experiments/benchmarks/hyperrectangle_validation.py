import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from utils.integrands.volume import HyperrectangleVolumeIntegrand
from utils.flat_integrals import validate_known_integrand_flat

dimensions = range(2, 10, 2)
volume_fractions = np.linspace(0.001, 0.999, 100).tolist()

results = pd.DataFrame()

n_batch = 10000

for d in dimensions:
    for volume_fraction in volume_fractions:
        f = HyperrectangleVolumeIntegrand(d, frac=volume_fraction)
        result = validate_known_integrand_flat(f, d, n_batch)
        result["volume_fraction"] = volume_fraction
        results = pd.concat([results, result.as_dataframe()], ignore_index=True)
fig, ax = plt.subplots()
ax.scatter(results["volume_fraction"], results["value_std"])
ax.set_xlabel("Volume fraction")
ax.set_ylabel("Integral uncertainty")
plt.show()

print(f"We found {100 * results['match'].mean()}% of matching tests")
