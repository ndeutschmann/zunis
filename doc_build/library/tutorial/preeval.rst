How to train on a pre-evaluated sample
######################################

ZüNIS provides integrators which use pre-evaluated samples. This is especially
useful when the to be integrated function is very costly to evaluate.

The functionality for using pre-evaluated samples are provided by the
:doc:`Fixed Sample Integrator </api/zunis.integration.fixed_sample_integrator>`.

Starting from the basic example, on can train on a sample already present as a
PyTorch tensor:

.. code-block:: python

  import torch
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
  integrator.set_sample(sample)
  result, uncertainty, history = integrator.integrate()

The sample have to be PyTorch tensors present on the same device in a 3-tuple, with
the first entry being of the shape `(sample_size, d)` containing the sampled points,
the second entry of shape `(sample_size,)` containing the sampling distribution
PDF values, and the last entry of the shape `(sample_size,)` containing the function
values.
Note that the number of evaluation points may not be greater than the sample size,
which is generally much bigger than in this example.

Another option to fed the sample to ZüNIS is via a pickle file, containing a sample
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

This saves the small example sample in a pickle file "sample.p" at the same path as the
Python script. This pickle file can be opened by the integrator and used as a sample.

As a last option, it is also possible to provide samples as a `.csv` file. This
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
