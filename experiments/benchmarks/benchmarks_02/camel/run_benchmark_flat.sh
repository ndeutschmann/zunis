#!/bin/bash

python benchmark_camel.py --config benchmarks_02/camel/camel_2d_grid_config_flat.yaml --cuda=0 --n_search=200&
python benchmark_camel.py --config benchmarks_02/camel/camel_4d_grid_config_flat.yaml --cuda=1 --n_search=200&
python benchmark_camel.py --config benchmarks_02/camel/camel_8d_grid_config_flat.yaml --cuda=2 --n_search=200&
python benchmark_camel.py --config benchmarks_02/camel/camel_16d_grid_config_flat.yaml --cuda=3 --n_search=200&
python benchmark_camel.py --config benchmarks_02/camel/camel_32d_grid_config_flat.yaml --cuda=4 --n_search=200&

# wait until all child processes terminate
wait
echo "All runs terminated"
