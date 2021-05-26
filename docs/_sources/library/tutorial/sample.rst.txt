How to sample from a trained model
##################################
Provided a pre-trained model, one can sample
from the trainer in a similar fashion as presented in the section :doc:`How to train
without integrating <nointeg>`:

.. code-block:: python

  import torch
  from zunis.models.flows.sampling import UniformSampler
  from zunis.training.weighted_dataset.stateful_trainer import StatefulTrainer

  device = torch.device("cuda")

  d = 2

  def f(x):
    return x[:,0]**2 + x[:,1]**2

  trainer = StatefulTrainer(d=d, device=device)
  x, px, fx=trainer.generate_target_batch_from_posterior(10, f, UniformSampler(d=d, device=device))
  trainer.train_on_batch(x,px,fx)

  trainer.sample_forward(10)

After performing the a training step, the trainer can be used for sampling.
`sample_forward` returns a tensor of shape `(n_points,d+1)` with the sampled points,
as well as the Jacobian of the transformation for the sampled point.


The model can also be saved on the disk for later use. In order to do so, one has
to save the PyTorch `state_dict`:

.. code-block:: python

  import torch
  from zunis.models.flows.sampling import UniformSampler
  from zunis.training.weighted_dataset.stateful_trainer import StatefulTrainer

  device = torch.device("cuda")

  d = 2

  def f(x):
    return x[:,0]**2 + x[:,1]**2

  trainer = StatefulTrainer(d=d, device=device)
  x, px, fx=trainer.generate_target_batch_from_posterior(10, f, UniformSampler(d=d, device=device))
  trainer.train_on_batch(x,px,fx)

  torch.save(trainer.flow.state_dict(),"model_dict")

One can then reload the model weights from the disk:

.. code-block:: python

  import torch
  from zunis.models.flows.sampling import UniformSampler
  from zunis.training.weighted_dataset.stateful_trainer import StatefulTrainer

  device = torch.device("cuda")

  d = 2

  def f(x):
    return x[:,0]**2 + x[:,1]**2

  trainer = StatefulTrainer(d=d, device=device)
  trainer.flow.load_state_dict(torch.load("model_dict"))
  trainer.sample_forward(10)
