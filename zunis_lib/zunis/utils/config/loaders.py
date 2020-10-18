import os
from functools import partial

import torch
from zunis.utils.config import Configuration

from .defaults import local_path


def get_optim_from_config(config=None):
    """Get an optimizer object from a config dictionary

    Parameters
    ----------
    config: dictionary-like, optional
        dictionary-like object with the following structure:
        {
        "optim_cls": <optim class name within torch.optim>,
        "optim_config": <dictionary defining the key-word argument parameters for instantiating that class>
        }
        If no argument is given, the default configuration is used (utils/config/optim_config.yaml)

    Returns
    -------
        :obj:`function`
            a pre-filled optimizer constructor that only needs to be fed the model parameters.

    Notes
    -----
    This function does not instantiate the optimize object, it wraps its constructor with functools.partial
    to pre-fill the optimizer settings with the provided value. Given a model, one then needs to call the output
    on model.parameters()
    """
    if config is None:
        config = Configuration.from_yaml(os.path.join(local_path, "optim_config.yaml"))

    optim = getattr(torch.optim, config["optim_cls"])
    return partial(optim, **config["optim_config"].as_dict())


def get_default_integrator_config():
    """Get a copy of the full default text-based integrator config

    The default configuration file is situated in utils/config/integrator_config.yaml
    """
    return Configuration.from_yaml(os.path.join(local_path, "integrator_config.yaml"))


def create_integrator_args(config=None):
    """Create the full hierarchy of arguments for an integrator, including an instantiated optimizer

    Parameters
    ----------
    config: dictionary-like, optional
        Full configuration dictionary for the Integrator function. If none is provided, the default configuration
        is used.

    Returns
    -------
        :obj:dict
            keyword dictionary ready to be provided as ``**kwargs`` to the
            :py:func:`Integrator <zunis.integration.default_integrator.Integrator>` function
    """
    if config is None:
        config = get_default_integrator_config()

    try:
        optim_config = config["optim"]
    except KeyError:
        optim_config = get_optim_from_config()
    args = config.as_dict()
    try:
        args["trainer_options"]["optim"] = get_optim_from_config(optim_config)
    except KeyError:
        args["trainer_options"] = {"optim": get_optim_from_config(optim_config)}
    return args