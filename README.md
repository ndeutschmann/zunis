ZüNIS: Normalizing flows for neural importance sampling
==============================

ZüNIS (Zürich Neural Importance Sampling) a work-in-progress Pytorch-based library for Monte-Carlo integration
 based on Neural imporance sampling [[1]](https://arxiv.org/abs/1808.03856), developed at ETH Zürich.
In simple terms, we use artificial intelligence to compute integrals faster.

The goal is to provide a flexible library to integrate black-box functions with a level of automation comparable to the
VEGAS Library [[2]](https://pypi.org/project/vegas/), while using state-of-the-art methods that go around
the limitations of existing tools.

## Installation

### Using `pip`

The library is available on PyPI:
```bash
 pip install zunis 
```

The latest version can be installed directly from GitHub:
```bash
    pip install 'git+https://github.com/ndeutschmann/zunis#egg=zunis&subdirectory=zunis_lib'
```

### Setting up a development environment

If you would like to contribute to the library, run the benchmarks or try the examples,
the easiest is to clone this repository directly and install the extended requirements:
````bash
# Clone the repository
git clone https://github.com/ndeutschmann/zunis.git ./zunis
# Create a virtual environment (recommended)
python3.7 -m venv  zunis_venv
source ./zunis_venv/bin/activate
pip install --upgrade pip
# Install the requirements
cd ./zunis
pip install -r requirements.txt
# Run one benchmark (GPU recommended)
cd ./experiments/benchmarks
python benchmark_hypersphere.py
````

## Library usage

For basic applications, the integrator is provided with default choices and can be created and used as follows:

```python
import torch
from zunis.integration import Integrator

device = torch.device("cuda")

d = 2

def f(x):
    return x[:,0]**2 + x[:,1]**2

integrator = Integrator(d=d,f=f,device=device)
result, uncertainty, history = integrator.integrate()
```

The function `f` is integrated over the `d`-dimensional unit hypercube and 

* takes `torch.Tensor` batched inputs with shape `(N,d)` for arbitrary batch size `N` on `device`
* returns `torch.Tensor` batched inputs with shape `(N,)` for arbitrary batch size `N` on `device`

A more systematic documentation is under construction [here](https://zunis.readthedocs.io).