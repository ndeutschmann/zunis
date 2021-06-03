How to invert a normalizing flow
##################################

For some applications, it is necessary to invert a trained model. As normalizing
flows are bijections, this is always possible, and due to the structure of coupling
cells, this can be done in an exact and efficient way. The forward and the backward
transformation access in runtime both the same parameters, so inverting a normalizing
flow in ZüNIS reduces to signaling the model which of both operations to perform.

The following example shows how this can be done:

.. code-block:: python

  import torch
  from zunis.models.flows.sampling import UniformSampler
  from zunis.training.weighted_dataset.stateful_trainer import StatefulTrainer

  device = torch.device("cpu")

  d = 2

  def f(x):
    return x[:,0]**2 + x[:,1]**2

  trainer = StatefulTrainer(d=d, loss="variance", flow="pwquad", device=device)

  n_points = 3
  # Uniformly sampled points
  x = torch.rand(n_points,d,device=device)
  print(x) #[[0.8797, 0.0277],[0.4615, 0.8289],[0.7171, 0.5085]]

  px = torch.ones(n_points)
  # px.shape = (n_points,)

  # Function values
  fx = f(x)

  sample = x, px, fx
  trainer.train_on_batch(x,px,fx)

  y=trainer.flow(torch.cat((x,px.unsqueeze(-1)),-1))
  print(y[:,:-1]) #[[0.8903, 0.0952],[0.2285, 0.6719],[0.5979, 0.3470]]

  trainer.flow.invert()
  q=trainer.flow(y)
  print(q[:,:-1]) #[[0.8797, 0.0277],[0.4615, 0.8289], [0.7171, 0.5085]]

From the sampled points, a training sample is created on which the trainer is optimized.
Then, we apply the trained model on the originally sampled points, which gives the mapped
points and the jacobian of the transformation. The trainer can be inverted by calling
`.invert()`, through which the underlying normalizing flow (`trainer.flow`) is operating
backwards instead of forward. This allows to again restore `x` from `y`.
