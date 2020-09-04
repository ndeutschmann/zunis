"""Tools for loading configurations"""
import os
from functools import partial

import torch

from utils.config import Configuration
from utils.config.configuration import Configuration
from utils.data_storage.record2sql import type_mapping

package_directory = os.path.dirname(os.path.abspath(__file__))


def get_optim_from_config(config=None):
    """Get an optimizer object from a config dictionary

    Parameters
    ----------
    config: dictionary-like, None
        dictionary-like object with the following structure:
        {
            "optim_cls": <optim class name within torch.optim>,
            "optim_config": <dictionary defining the key-word argument parameters for instantiating that class>
        }
        If no argument is given, the default configuration is used (utils/config/optim_config.yaml)

    Returns
    -------
        function
            a pre-filled optimizer constructor that only needs to be fed the model parameters.

    Notes
    -----
    This function does not instantiate the optimize object, it wraps its constructor with functools.partial
    to pre-fill the optimizer settings with the provided value. Given a model, one then needs to call the output
    on model.parameters()
    """
    if config is None:
        config = Configuration.from_yaml(os.path.join(package_directory, "optim_config.yaml"))

    optim = getattr(torch.optim, config["optim_cls"])
    return partial(optim, **config["optim_config"].as_dict())


def get_default_integrator_config():
    """Get a copy of the full default text-based integrator config

    The default configuration file is situated in utils/config/integrator_config.yaml
    """
    return Configuration.from_yaml(os.path.join(package_directory, "integrator_config.yaml"))


def create_integrator_args(config=None):
    """Create the full hierarchy of arguments for an integrator, including an instantiated optimizer

    Parameters
    ----------
    config: dictionary-like, None
    Full configuration dictionary for the Integrator function. If none is provided, the default configuration
    is used.

    Returns
    -------
        dict
            keyword dictionary ready to be provided as **kwargs to the Integator function
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


def get_sql_types(type_config=None):
    """Get the SQL types associated to configuration parameters from a string-based type-config

    Parameters
    ----------
    type_config: MutableMapping, None
        maps parameter names to strings describing data types. Allows MutableMappings (key-value mappings) and
        can accomodate Configuration objects. If no argument is given, the default integrator configuration is used.

    Returns
    -------
        dict
    """

    if type_config is None:
        type_config = Configuration.from_yaml("../config/integrator_config_types.yaml")

    if isinstance(type_config, Configuration):
        type_config = type_config.as_flat_dict()

    sql_types = dict([(key, type_mapping[value]) for key, value in type_config.items()])

    return sql_types