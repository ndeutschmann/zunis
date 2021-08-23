#!/bin/bash

sleep 0; python benchmark_camel.py --config benchmarks_05/camel/camel_2d_defaults.yaml --cuda=0 --n_search=10 --stratified&
sleep 3; python benchmark_camel.py --config benchmarks_05/camel/camel_4d_defaults.yaml --cuda=1 --n_search=10 --stratified&
sleep 3; python benchmark_camel.py --config benchmarks_05/camel/camel_8d_defaults.yaml --cuda=2 --n_search=10 --stratified&
sleep 3; python benchmark_camel.py --config benchmarks_05/camel/camel_16d_defaults.yaml --cuda=3 --n_search=10 --stratified&
sleep 3; python benchmark_camel.py --config benchmarks_05/camel/camel_32d_defaults.yaml --cuda=4 --n_search=10 --stratified&

# wait until all child processes terminate
wait
echo "All runs terminated"