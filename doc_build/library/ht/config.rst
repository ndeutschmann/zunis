
How to use a config file
########################

Config files can be used to specify arguments for an integrator, including an
instantiated optimizer. This is included in the :doc:`config </api/zunis.utils.config>` subpackage.
A default config file, integrator_config.yaml, is given there:

.. code-block:: yaml

  flow: pwquad
  flow_options:
  cell_params:
    d_hidden: 256
    n_bins: 10
    n_hidden: 8
  masking: iflow
  masking_options:
    repetitions: 2
  loss: variance
  n_points_survey: 10000
  n_iter: 10
  trainer_options:
  minibatch_size: 1.0
  max_reloads: 0
  n_epochs: 50
  optim:
    optim_cls: Adam
    optim_config:
      betas: !!python/tuple
      - 0.9
      - 0.999
      eps: 1.0e-08
      lr: 0.001
  checkpoint: True
  checkpoint_on_cuda: True
  checkpoint_path: null

the settings specified in the config file are used for the setup of the trainer,
the integrator and the flow.
The flow option specifies which kind of flow to use in the coupling
cells (choices being "realnvp", "pwlinear" or "pwquad"), as well as the geometry
of the underlying neural network and, in case of piecewise-linear or -quadratic
flows, the number of bins. It is also possible to choose either a "checkerboard",
"maximal" or "iflow" masking strategy and define how many sets of coupling cells
should be used.
For the purpose of training, either a "variance" or "dkl" loss can be specified.
For the DKL loss, it is possible to also request the survey strategy "adaptive_dkl".
n_iter refers to the number of iterations, whereas n_points_survey defines the
number of points used per iteration for the survey stage; the same can be defined
for the refine stage too.
Apart from this, the trainer options itself can be also defined - the size of
minibatches, the maximum number of how often the trainer is allowed to restore
from a checkpoint if an exception happens as well as how many epochs are used
during an iteration. If "checkpoint" is set to True, checkpoints are saved
(on the GPU if "checkpoint_on_cuda" is true), alternative checkpoints can be
also taken from a file if a path is given. Lastly, the optimizer settings itself
are specified, defining which algorithm to use as well as its parameters.

TODO: add code
