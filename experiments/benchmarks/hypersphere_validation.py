import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from utils.integrands import HypersphereVolumeIntegrand
from utils.flat_integrals import validate_known_integrand_flat

results = pd.DataFrame()
n_batch = 10000000

dimensions = range(2,10)
radii = np.linspace(0.48,0.2,20)
centers = [0.5]

for d in dimensions:
    for r in radii:
        for c in centers:
            f = HypersphereVolumeIntegrand(d, r, c)
            result = validate_known_integrand_flat(f, d, n_batch)
            result["dim"] = d
            result["radius"] = r
            result["center"] = c
            results = pd.concat([results, result.as_dataframe()], ignore_index=True)

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

