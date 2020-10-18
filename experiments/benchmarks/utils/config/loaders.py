"""Tools for loading configurations"""
import os

from zunis.utils.config import Configuration
from utils.data_storage.record2sql import type_mapping

package_directory = os.path.dirname(os.path.abspath(__file__))


def get_sql_types(type_config=None):
    """Get the SQL types associated to configuration parameters from a string-based type-config

    Parameters
    ----------
    type_config: dictionary-like, optional
        maps parameter names to strings describing data types. Allows MutableMappings (key-value mappings) and
        can accomodate Configuration objects. If no argument is given, the default integrator configuration is used.

    Returns
    -------
        :obj:`dict`
    """

    if type_config is None:
        type_config = Configuration.from_yaml(os.path.join(package_directory, "integrator_config_types.yaml"))

    if isinstance(type_config, Configuration):
        type_config = type_config.as_flat_dict()

    sql_types = dict([(key, type_mapping[value]) for key, value in type_config.items()])

    return sql_types
