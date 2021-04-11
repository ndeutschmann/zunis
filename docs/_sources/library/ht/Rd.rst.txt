How to integrate in R^d
########################

ZüNIS samples into the unit hypercube. In order to integrate functions over the
whole space of real numbers, it is necessary to perform a variable transformation.
The change of variables has to map [0,1] to R. This can be done by using any
bijective function with two divergences. Examples of such functions would be the
tangens or the arctanh.

Starting from the basic example, and integrating over a function in R^2 instead,
this could be done like the following:

.. code-block:: python

  import torch
  import numpy as np
  from zunis.integration import Integrator

  device = torch.device("cuda")

  d = 2

  def f(x):
      return torch.exp(-(x).square().sum(axis=1))

  def f_wrapped(x):
      return f(torch.tan(np.pi*(x-0.5)))*((np.pi/(torch.cos(np.pi*(x-0.5)).square()))).prod(axis=1)
      
  integrator = Integrator(d=d,f=f_wrapped,device=device)
  result, uncertainty, history = integrator.integrate()


Note that sampling uniformly over an arbitrary large interval instead of a
change of variable introduces infinite variance.
