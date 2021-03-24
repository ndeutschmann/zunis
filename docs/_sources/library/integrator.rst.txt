:todo:

###########
Integrators
###########

Integrators are intended as the main way for standard users to interact with ZüNIS.
They provide a high-level interface to the functionalities of the library and only optionally require you to know
to what lower levels of abstractions really entail and what their options correspond.
At the highest possible level, :func:`zunis.integration.Integrator <zunis.integration.default_integrator.Integrator>`
allows you to interface with the different types of integrators and comes with sane defaults for each of them.

******************
The Integrator API
******************

The main API to use ZüNIS integrators is
:func:`zunis.integration.Integrator <zunis.integration.default_integrator.Integrator>`,
which will instantiate the correct type of integrator and of subcomponents (trainer and flow).
Only two arguments are necessary to define an integrator with this API: a number
of dimensions and a :doc:`function <function>` mapping batches of pytorch `Tensors` into batches of values

.. code-block:: python

    from zunis.integration import Integrator

    def f(x):
        return x[:,0]**2 + x[:,1]**2

    integrator = Integrator(d=d,f=f)

Computing the integral is then a matter of calling the
:meth:`integrate <zunis.integration.base_integrator.BaseIntegrator.integrate>` method.

.. code-block:: python

    result, uncertainty, history = integrator.integrate()
    print(f"{result:.3e} +/- {uncertainty:.3}")
    # > 6.666e-01 +/- 4.69e-05

The main options of :func:`zunis.integration.Integrator <zunis.integration.default_integrator.Integrator>` control some
high-level choices:

* `loss` controls the loss function used during training. The options are `'variance'` (default) or `'dkl'`.
* `flow` controls which normalizing flow will be used. The options are `'pwquad'` (default), `'pwlin'` and `'realnvp'`. Without much surprise, this controls which flow class will be used


Furthermore, a few options are used to control administrative things:

* `device` controls where the integration is performed (*e.g.* `torch.device("cuda")`)
* `verbosity` controls the logging verbosity of the integration process
* `trainer_verbosity` controls the logging verbosity of the training process during the survey stage

Note that by default, the :obj:`ZüNIS logger <zunis.logger>` does not have a handler. Use
:func:`zunis.setup_std_stream_logger` to setup handlers to `stdout` and `stderr`.

Further customization requires one to set specific options for the lower level objects used by the integrator: either
the :doc:`Trainer <trainer>` or the :doc:`Flow <flow>`, which can be set through `trainer_options` and `flow_options`
respectively.

Configuration files
===================

An efficient way of defining specific options for an integrator is to use configuration files which encode the options
passed to the Integrator API. A good place to get started is the function
:func:`create_integrator_args <zunis.utils.config.loaders.create_integrator_args>` which can be called without arguments
to get a keyword dictionary with default options

.. code-block:: python

    from zunis.utils.config.loaders import create_integrator_args

    kwargs = create_integrator_args()
    integrator = integrator(d=2, f=f, **kwargs)
    print(kwargs)
    #{'flow': 'pwquad',
    #'flow_options': {'cell_params': {'d_hidden': 256, 'n_bins': 10, 'n_hidden': 8},
    #                  'masking': 'iflow',
    #                 'masking_options': {'repetitions': 2}},
    #'loss': 'variance',
    #'n_iter': 10,
    #'n_points_survey': 10000,
    #'trainer_options': {'checkpoint': True,
    #                    'checkpoint_on_cuda': True,
    #                    'checkpoint_path': None,
    #                    'max_reloads': 0,
    #                    'minibatch_size': 1.0,
    #                    'n_epochs': 50,
    #                    'optim': <class 'torch.optim.adam.Adam'>}}

This function actually reads a template configuration file `zunis/utils/config/integrator_config.yaml` by
calling the function :func:`get_default_integrator_config <zunis.utils.config.loaders.get_default_integrator_config>`.
A good way to experiment with the settings of Integrators and their subcomponents is to load this default and
adjust it:

.. code-block:: python

    from unis.utils.config.loaders import get_default_integrator_config
    from zunis.utils.config.loaders import create_integrator_args

    config = get_default_integrator_config()
    config['loss'] = 'dkl'
    config['lr'] = 1.e-4
    config['n_bins'] = 100

    kwargs = create_integrator_args(config)
    integrator = integrator(d=d, f=f, **kwargs)

Note that the :class:`Configuration <zunis.utils.config.configuration.Configuration>` object generated allows easy
edition despite its nested structure.

If you want to fully specify your configuration, you can define your own configuration file and make it a
:class:`Configuration <zunis.utils.config.configuration.Configuration>` by calling `Configuration.from_yaml`.


**********************
How Integrators work
**********************


Survey and Refine phases
========================

All integrators work by first performing a *survey phase*, in which it optimizes the way it samples points and then a
*refine phase*, in which it computes the integral by using its learned sampler. Each phase proceeds through a number
of steps, which can be set at instantiation or when integrating:

.. code-block:: python

    integrator = Integrator(d=d, f=f, n_iter_survey=3, n_iter_refine=5) # Default values
    integrator.integrate(n_survey=10, n_refine=10) # Override at integration time

For both the survey and the refine phases, using multiple steps is useful to monitor the stability of the training and of
the integration process: if one step is not within a few standard deviations of the next, either the sampling statistics
are too low, or something is wrong. For the refine stage, this is the main real advantage of using multiple steps. On the
other hand, at each new survey step, a new batch of points is re-sampled, which can be useful to mitigate overfitting.

By default, only the integral estimates obtained during the refine stage are combined to compute the final integral estimate,
and their combination is performed by taking their average. Indeed, because the model is trained during the survey step,
the points sampled during the refine stage are correlated in an uncontrolled way with the points used during training.
Ignoring the survey stage makes all estimates used in the combination independent
random variables, which permits us to build a formally correct estimator of the variance of the final result.
