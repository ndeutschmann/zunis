How to integrate in R^d
########################

ZüNIS integrates over the unit hypercube. In order to integrate functions over the
whole real line, it is necessary to perform a variable transformation.
The change of variables has to map [0,1] to R. This can be done by using any
bijective function with two divergences such as the tangent and the inverse hyperbolic tangent functions.

Here is for example how to compute a gaussian integral in two dimensions.

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
