
How to use a configuration file
###############################

Configuration files can be used to specify arguments for the :doc:`Integrator </library/integrator>` interface in YAML.
This is implemented in the :doc:`config </api/zunis.utils.config>` subpackage.
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

The settings specified in the configuration file are used for the setup of the trainer,
the integrator and the flow.

The flow option specifies which kind of flow to use in the coupling
cells (choices being `realnvp`, `pwlinear` or `pwquad`), as well as the geometry
of the underlying neural network and, in case of piecewise-linear or -quadratic
flows, the number of bins. It is also possible to choose either a `checkerboard`,
`maximal` or `iflow` masking strategy and define how many sets of coupling cells
should be used.

For the purpose of training, either a `variance` or `dkl` loss can be specified.
Next to the default `flat` survey strategy, there exists also the `forward` and
`forward_flat_int` survey strategy. For fixed samples, the `fixed_sample` survey
strategy creates a :doc:`Fixed Sample Integrator </api/zunis.integration.fixed_sample_integrator>`.
Specific for variance/DKL loss,
a survey strategy `adaptive_variance`/`adaptive_dkl` is provided.
`n_iter` refers to the number of iterations, whereas `n_points_survey` defines the
number of points used per iteration for the survey stage; the same can be defined
for the refine stage too.

Besides this, the trainer options itself can be also defined - the size of
minibatches, the maximum number of how often the trainer is allowed to restore
from a checkpoint if an exception happens as well as how many epochs are used
during an iteration. If `checkpoint` is set to True, checkpoints are saved
(on the GPU if `checkpoint_on_cuda` is true), alternative checkpoints can be
also taken from a file if a path is given. Lastly, the optimizer settings itself
are specified, defining which algorithm to use as well as its parameters.

In general, all keywords arguments specified for :func:`Integrators <zunis.integration.default_integrator.Integrator>` can be defined
in a configuration file.

Extending the basic example, this configuration file can be loaded to the integrator
in the following way:

.. code-block:: python

  import torch
  from zunis.integration import Integrator
  from zunis.utils.config.loaders import create_integrator_args

  device = torch.device("cpu")

  d = 2

  def f(x):
      return x[:,0]**2 + x[:,1]**2

  integrator = Integrator(f,d,**create_integrator_args(),device=device)
  result, uncertainty, history = integrator.integrate()

`create_integrator_args(None)` returns a dictionary with keyword arguments which
are given to the integrator. The values of the keyword arguments are specified by
the yaml file which is at the path specified by the argument. If the argument
is `None`, as it is in this case, the quoted default `config.yaml` is loaded.

The config files can be written by hand, or, alternatively, a generator is also
available at `zunis.utils.config.generators`


.. code-block:: python

  import torch
  from zunis.integration import Integrator
  from zunis.utils.config.loaders import create_integrator_args
  from zunis.utils.config.generators import create_integrator_config_file

  device = torch.device("cpu")

  d = 2

  def f(x):
      return x[:,0]**2 + x[:,1]**2

  create_integrator_config_file(filepath="integrator_config_new.yaml", base_config="integrator_config_old.yaml", n_points_survey=20000)
  integrator = Integrator(f,d,**create_integrator_args("integrator_config_new.yaml"),device=device)
  result, uncertainty, history = integrator.integrate()

This example loads an old, preexistent config file, changes the number of survey
points and provides the updated file to the integrator.
