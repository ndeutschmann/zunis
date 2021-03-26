
How to train without integrating
################################

ZüNIS allows to train a model without performing the integration. Starting
from the basic example, the settings are the following:

.. code-block:: python

  import torch
  from zunis.integration import Integrator

  device = torch.device("cuda")

  d = 2

  def f(x):
      return x[:,0]**2 + x[:,1]**2

  integrator = Integrator(d=d,f=f,device=device, use_survey=False)
  integrator.survey()
  integrator.n_iter_survey=0
  result, uncertainty, history = integrator.integrate()

`use_survey` determines if the survey points, which are used for training,
should be also used for integration. This is not the case per default. Calling
`survey()` performs only the training without calculating the integral. The
trained model can then be used for performing the integration. In order to not
attempt to train the model any further, `n_iter_survey` was now set to 0.
