
How to train without integrating
################################

ZüNIS allows to train a model without performing the integration. For this, we can
use the trainer API:

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

In a first step, we initialize a trainer object. We generate a batch of 10 uniformly
sampled points in the 2D hypercube `x` with the probability distribution `px`
and the function value `fx`. Then, one training step is performed on this batch.
