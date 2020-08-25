"""Standard set of options to be logged"""
import torch
from functools import partial
from copy import deepcopy

#TODO: these dictionaries should not be hardcoded here but read from file, it would be much nicer and
#TODO: allow reloading the default behavior (dangerous here as dicts are mutable)
#TODO: wrap in a class that recursively searches for child dictionaries to allow easy assignment

standard_pwlinear_params = {
    "n_bins": 10,
    "d_hidden": 256,
    "n_hidden": 8,
}

standard_checkerboard_options = {
    "repetitions": 2
}

standard_adam_config = {"lr": 1.e-3,
                        'betas': (0.9, 0.999),
                        'eps': 1e-08}

# The trainer config requires a torch.optim.Optimizer class object to instantiate on the the model parameters
# this is not suitable to text-based logging
# We therefore differentiate the loggable config from the options passed
standard_optim_config = {
    "optim": "Adam",
    "optim_config": standard_adam_config
}

def get_optim_from_config(config=None):
    """Get an optimizer object from a config dictionary"""
    if config is None:
        config = standard_optim_config

    optim = getattr(torch.optim, config["optim"])
    return partial(optim, **config["optim_config"])

#Text-based config at the trainer level
standard_trainer_config = {
    "n_epochs": 50,
    "optim": standard_optim_config,
    "minibatch_size": 1.
}

# Text-based config at the integrator level
standard_integrator_config = {
    "n_points_survey": 10000,
    "trainer_options": standard_trainer_config,
    "flow": "pwlinear",
    "loss": "variance",
    "flow_options": {
        "masking": "checkerboard",
        "masking_options": standard_checkerboard_options,
        "cell_params": standard_pwlinear_params}
}

# Typical process:
# config = get_default_integrator_config()
# ... override some config details
# args = create_integrator_args(config)
# Integrator(..., **args)
# flat_config = flatten_config(config)
# ... log the flat config somewhere

def get_default_integrator_config():
    """Get a copy of the full text-based integrator config"""
    return deepcopy(standard_integrator_config)


def create_integrator_args(config=None):
    """Create the full hierarchy of arguments for an integrator, including an instantiated optimizer"""
    if config is None:
        config = get_default_integrator_config()

    optim_config = config["trainer_options"]["optim"]
    args = deepcopy(config)
    try:
        args["trainer_options"]["optim"] = get_optim_from_config(optim_config)
    except KeyError:
        args["trainer_options"] = {"optim": get_optim_from_config(optim_config)}
    return args


def flatten_config(config):
    flat_config = dict()
    for key in config:
        if isinstance(config[key], dict):
            flat_lower_level = flatten_config(config[key])
            for lower_key in flat_lower_level:
                assert lower_key not in flat_config, f"cannot flatten config: key clash for {lower_key}"
                flat_config[lower_key] = flat_lower_level[lower_key]
        else:
            assert key not in flat_config, f"cannot flatten config: key clash for {key}"
            flat_config[key] = config[key]
    return flat_config
