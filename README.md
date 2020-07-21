Normalizing flows for neural importance sampling
==============================

This is a work-in-progress library to provide importance sampling Monte-Carlo integration tools based on
Neural imporance sampling [[1]](https://arxiv.org/abs/1808.03856). This method uses normalzing flows to optimally sample
an integrand function in order to evaluate its (multi-dimensional) integral.

The goal is to provide a flexible library to integrate black-box functions for which classical methods such as VEGAS do
not work well due to an unknown or complicated structure which prevents the typical variable change and multi-channelling
tricks.

## Usage

For basic uses, a RealNVP-based integrator is provided with default choices and can be created and used as follows:

```
import torch
from src.integration import DefaultIntegrator

device = torch.device("cuda")


d = 2

def f(x):
    return x[:,0]**2 + x[:,1]**2

integrator = DefaultIntegrator(d=d,f=f,device=device)
result, uncertainty, history = integrator.integrate()
```

The function `f` is integrated over the `d`-dimensional unit hypercube and 
* takes `torch.Tensor` batched inputs with shape `(N,d)` for arbitrary batch size `N`
* returns `torch.Tensor` batched inputs with shape `(N,)` for arbitrary batch size `N` 

A more systematic documentation is under construction [here](https://ndeutschmann.github.io/pytorch_flows).