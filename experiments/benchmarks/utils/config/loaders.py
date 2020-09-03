""""""
import os
from functools import partial

import torch

from utils.config.configuration import Configuration

package_directory = os.path.dirname(os.path.abspath(__file__))


def get_optim_from_config(config=None):
    """Get an optimizer object from a config dictionary"""
    if config is None:
        config = Configuration.from_yaml(os.path.join(package_directory, "optim_config.yaml"))

    optim = getattr(torch.optim, config["optim_cls"])
    return partial(optim, **config["optim_config"].as_dict())


def get_default_integrator_config():
    """Get a copy of the full text-based integrator config"""
    return Configuration.from_yaml(os.path.join(package_directory, "integrator_config.yaml"))


def create_integrator_args(config=None):
    """Create the full hierarchy of arguments for an integrator, including an instantiated optimizer"""
    if config is None:
        config = get_default_integrator_config()

    optim_config = config["optim"]
    args = config.as_dict()
    try:
        args["trainer_options"]["optim"] = get_optim_from_config(optim_config)
    except KeyError:
        args["trainer_options"] = {"optim": get_optim_from_config(optim_config)}
    return args
