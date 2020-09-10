"""Standard set of options to be logged


Typical process:

.. code-block:: python

    config = get_default_integrator_config()
    # ... override some config details
    args = create_integrator_args(config)
    Integrator(..., **args)
    flat_config = config.to_dict_flat()
    # ... log the flat config somewhere

"""

