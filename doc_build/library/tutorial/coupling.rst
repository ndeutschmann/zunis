How to define a custom coupling cell
#####################################

ZüNIS provides popular choices for the coupling transforms, including affine transformations
as well as piecewise-linear and piecewise-quadratic transformations. In case the
user wishes to investigate the effecte of alternative choices of the coupling transform,
it easy to extend the classes provided by this package to do so. In the first step,
one needs to define an invertible coupling transform:

.. code-block:: python

  import torch
  from zunis.models.flows.sampling import FactorizedGaussianSampler
  from zunis.training.weighted_dataset.stateful_trainer import StatefulTrainer
  from zunis.models.flows.coupling_cells.general_coupling import InvertibleCouplingCell
  from zunis.models.flows.coupling_cells.transforms import InvertibleTransform
  from zunis.models.layers.trainable import ArbitraryShapeRectangularDNN

  class LinearTransform(InvertibleTransform):
    def forward(self,x,T):
        alpha = torch.exp(T)
        logj = T*x.shape[-1]
        return x*alpha, logj.squeeze()
    def backward(self,x,T):
        alpha = torch.exp(-T)
        logj = -T*x.shape[-1]
        return x*alpha, logj.squeeze()

Here, we chose a very simple linear mapping

.. math::

  y = Q(x):\;\left\{ \begin{array}{l} y^A = x^A\\ y^B = \exp\left(T(x^A)\right) \times x^B,\end{array} \right.

where the argument of the exponential is strictly positive and which can be
inverted in a straightforward way. Starting from this linear bijective transformation,
one can define a coupling cell by inheriting from ZüNIS' invertible coupling cell
class:

.. code-block:: python

  class LinearCouplingCell(InvertibleCouplingCell):
    def __init__(self, d, mask, nn_width, nn_depth):
        transform = LinearTransform()
        super(LinearCouplingCell, self).__init__(d=d, mask=mask,transform=transform)
        d_in = sum(mask)
        self.T = ArbitraryShapeRectangularDNN(d_in=d_in,out_shape=(1,),d_hidden=nn_width,n_hidden=nn_depth)
        self.inverse=False

This class is provided with the transformation we just defined, as well as with
the definition of neural network, for which we choose a generic rectangular dense
neural network as provided by ZüNIS. This coupling cell can now replace the predefined
coupling cells present in ZüNIS:

.. code-block:: python

  d = 2
  device = torch.device("cpu")

  mask=[True,False]
  nn_width=8
  nn_depth=256

  sampler=FactorizedGaussianSampler(d=d)
  linear_coupling=LinearCouplingCell(d,mask,nn_width,nn_depth)
  trainer = StatefulTrainer(d=d, loss="variance", flow_prior=sampler,flow=linear_coupling, device=device)

After defining the number of dimensions and the hardware we want to work on, we need
to provide a masking as well as the architecture of the neural network for creating
an instance of our coupling cell.
Additionally, the trainer needs to be supported with a sampling layer, which we
choose in this case to be a Gaussian sampler. Now, instead of providing a string
to the "flow" keyword fo the trainer, we can provide as an argument instead the
instance of our coupling cell, which will now be used for training.
