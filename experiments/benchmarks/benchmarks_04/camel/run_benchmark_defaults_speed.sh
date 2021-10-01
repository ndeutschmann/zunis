#!/bin/bash

python benchmark_camel.py --config benchmarks_04/camel/camel_2d_defaults.yaml --cuda=5 --n_repeat=1&
sleep 3; python benchmark_camel.py --config benchmarks_04/camel/camel_4d_defaults.yaml --cuda=1 --n_repeat=1&
sleep 3; python benchmark_camel.py --config benchmarks_04/camel/camel_8d_defaults.yaml --cuda=2 --n_repeat=1&
sleep 3; python benchmark_camel.py --config benchmarks_04/camel/camel_16d_defaults.yaml --cuda=3 --n_repeat=1&
sleep 3; python benchmark_camel.py --config benchmarks_04/camel/camel_32d_defaults.yaml --cuda=4 --n_repeat=1&

# wait until all child processes terminate
wait
echo "All runs terminated"
