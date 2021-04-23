How to train on a pre-evaluated sample
######################################

ZüNIS provides integrators which use pre-evaluated samples. This is especially
useful when fine-tuning integration parameters for a function that is very costly to evaluate.

The functionality for using pre-evaluated samples are provided by the
:doc:`Fixed Sample Integrator </api/zunis.integration.fixed_sample_integrator>`.

Starting from the basic example, on can train on a sample defined as a
PyTorch tensor:

.. code-block:: python

  import torch
  from zunis.integration.fixed_sample_integrator import  FixedSampleSurveyIntegrator
  from zunis.training.weighted_dataset.stateful_trainer import StatefulTrainer

  device = torch.device("cuda")

  d = 2

  def f(x):
      return x[:,0]**2 + x[:,1]**2

  trainer = StatefulTrainer(d=d, loss="variance", flow="pwquad", device=device)
  integrator =  FixedSampleSurveyIntegrator(f,trainer, device=device, n_points_survey=5)

  n_points = 100
  # Uniformly sampled points
  x = torch.rand(n_points,d,device=device)
  # x.shape = (n_points,d)

  px = torch.ones(n_points)
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
  from zunis.integration.fixed_sample_integrator import  FixedSampleSurveyIntegrator
  from zunis.training.weighted_dataset.stateful_trainer import StatefulTrainer

  device = torch.device("cuda")

  d = 2

  def f(x):
      return x[:,0]**2 + x[:,1]**2

  trainer = StatefulTrainer(d=d, loss="variance", flow="pwquad", device="cuda")
  integrator =  FixedSampleSurveyIntegrator(f,trainer, device=device, n_points_survey=5)

  data_x=[[0,4],[1,3],[2,2],[3,1],[4,0]]
  data_px=[1.0,1.0,1.0,1.0,1.0]

  sample=(torch.tensor(data_x, device="cuda"),torch.tensor(data_px, device="cuda"),f(torch.tensor(data_x, device="cuda")))
  pickle.dump(sample, open("sample.p","wb"))

  integrator.set_sample_pickle("sample.p",device="cuda")
  result, uncertainty, history = integrator.integrate()

Finally , it is also possible to provide samples as a `.csv` file. This
file has to have `d+2` columns, with the first `d` columns containing the sampled
points, the second the sampling distribution PDF values and the last the function
value.
For the above example, the `.csv` file would look like:

.. code-block:: python

  0, 4, 1, 16
  1, 3, 1, 10
  2, 2, 1, 8
  3, 1, 1, 10
  4, 0, 1, 16

This could be imported as a pre-evaluated example and used for integration in the
following way:

.. code-block:: python

  import torch
  import numpy as np
  from zunis.integration.fixed_sample_integrator import  FixedSampleSurveyIntegrator
  from zunis.training.weighted_dataset.stateful_trainer import StatefulTrainer

  device = torch.device("cuda")

  d = 2

  trainer = StatefulTrainer(d=d, loss="variance", flow="pwquad", device=device)
  integrator =  FixedSampleSurveyIntegrator(f,trainer, device=device, n_points_survey=5)


  integrator.set_sample_csv("sample.csv",device="cuda",dtype=np.float32)
  result, uncertainty, history = integrator.integrate()
