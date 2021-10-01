How to train on a pre-evaluated sample
######################################

ZüNIS provides integrators which use pre-evaluated samples. This is especially
useful when fine-tuning integration parameters for a function that is very costly to evaluate.

The functionality for using pre-evaluated samples are provided by the
:doc:`Fixed Sample Integrator </api/zunis.integration.fixed_sample_integrator>`.
This integrator is accessible when using config files by choosing the survey strategy
`fixed_sample`.

Starting from the basic example, on can train on a sample defined as a
PyTorch tensor:

.. code-block:: python

  import torch
  from zunis.integration import Integrator

  device = torch.device("cuda")

  d = 2

  def f(x):
      return x[:,0]**2 + x[:,1]**2

  integrator =  Integrator(d=d, f=f, survey_strategy='fixed_sample', device=device, n_points_survey=1000)

  n_points = 1000
  # Uniformly sampled points
  x = torch.rand(n_points,d,device=device)
  # x.shape = (n_points,d)

  px = torch.ones(n_points, device=device)
  # px.shape = (n_points,)

  # Function values
  fx = f(x)


  sample = x, px, fx
  integrator.set_sample(sample)
  result, uncertainty, history = integrator.integrate()

The sample have to be PyTorch tensors present on the same device in a 3-tuple, with the first containing the sampled points,
the second containing the sampling distribution
PDF values, and the last entry containing the function
values, respectively of shapes `(sample_size, d)`, `(sample_size,)` and `(sample_size,)`


Fixed sample integrators can also directly import a pickle file, containing a sample
batch of the same structure:

.. code-block:: python

  import torch
  import pickle
  from zunis.integration import Integrator

  device = torch.device("cuda")

  d = 2

  def f(x):
    return x[:,0]**2 + x[:,1]**2

  integrator =  Integrator(d=d, f=f, survey_strategy='fixed_sample', device=device, n_points_survey=1000)

  data_x = torch.rand(1000,d,device=device)
  #[[0.2093, 0.9918],[0.3216, 0.6965],[0.0625, 0.5634],...]
  data_px = torch.ones(1000)
  #[1.0,1.0,1.0...]

  sample=(data_x.clone().detach(),data_px.clone().detach(),f(data_x.clone().detach()))
  pickle.dump(sample, open("sample.p","wb"))

  integrator.set_sample_pickle("sample.p",device=device)
  result, uncertainty, history = integrator.integrate()

Finally , it is also possible to provide samples as a `.csv` file. This
file has to have `d+2` columns, with the first `d` columns containing the sampled
points, the second the sampling distribution PDF values and the last the function
value.
For the above example, the `.csv` file would look like:

.. code-block:: python

  0.2093, 0.9918, 1, 1.0274
  0.3216, 0.6965, 1, 0.5885
  0.0625, 0.5634, 1, 0.3213
  ...

This could be imported as a pre-evaluated example and used for integration in the
following way:

.. code-block:: python

  import torch
  import numpy as np
  from zunis.integration import  Integrator

  device = torch.device("cuda")

  d = 2

  integrator =  Integrator(d=d, f=f, survey_strategy='fixed_sample', device=device, n_points_survey=1000)

  integrator.set_sample_csv("sample.csv",device="cuda",dtype=np.float32)
  result, uncertainty, history = integrator.integrate()
