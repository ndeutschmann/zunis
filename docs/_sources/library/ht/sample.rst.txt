How to sample from a trained model
##################################
In the case that we have a pre-trained integrator object present, one can sample
from the trainer in a similar fashion as presented in the section :doc:`How to train
without integrating <nointeg>`:

.. code-block:: python

  import torch
  from zunis.integration import Integrator

  device = torch.device("cuda")

  d = 2

  def f(x):
      return x[:,0]**2 + x[:,1]**2

  integrator = Integrator(d=d,f=f,device=device, use_survey=False, n_points_survey=1000)
  integrator.survey()
  integrator.sample_refine(n_points=10, f=f)

After performing the survey step, the model is trained and can be used for sampling.
`sample_refine` returns a tensor of shape `(n_points,d)` with the sampled points,
as well as the Jacobian of the transformation for the sampled point and the function
value.


The model can also be saved on the disk for later use. In order to do so, one has
to save the PyTorch `state_dict`:

.. code-block:: python

  import torch
  from zunis.integration import Integrator

  device = torch.device("cuda")

  d = 2

  def f(x):
      return x[:,0]**2 + x[:,1]**2

  integrator = Integrator(d=d,f=f,device=device, use_survey=False)
  integrator.survey()
  torch.save(integrator.model_trainer.flow.state_dict(), "model_dict")

This saves the trained model to the same path as the execution file. Now, one can
at a later point initialise an untrained model with the same parameters and load
the trained state from the disk:

.. code-block:: python

  import torch
  from zunis.integration import Integrator

  device = torch.device("cuda")

  d = 2

  def f(x):
      return x[:,0]**2 + x[:,1]**2

  integrator.model_trainer.flow.load_state_dict(torch.load("model_dict"))
  integrator.sample_refine(n_points=10, f=f)
