"""Tools to create configuration files for ZuNIS objects"""
import ruamel.yaml as yaml

from .configuration import Configuration
from .loaders import get_default_integrator_config


def create_integrator_config_file(filepath="integator_config.yaml", base_config=None, force=False, **kwargs):
    """Create a config file for an integrator.

    Parameters
    ----------
    filepath: str
        path of the file to which a config file will be saved
    base_config: str or None
        path of the source config file on which to base the new config.
        If none, use the default provided with the library
    force: bool, False
        whether to overwrite an existing target file
    **kwargs:
        options that will be set in the new target file. If the option does not already exist in the base config file,
        the option is created at the top level. Otherwise, the existing option is updated even if it is nested
        (see :py:class:`zunis.utils.config.configuration.Configuration`).

    """
    if base_config is None:
        config = get_default_integrator_config()
    else:
        config = Configuration.from_yaml(base_config)

    for arg in kwargs:
        config[arg] = kwargs[arg]

    config = config.as_dict()

    mode = "w" if force else "x"

    yaml.dump(config, open(filepath, mode=mode))
