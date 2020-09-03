"""Tools to convert configurations to SQL data"""

import os
import sqlalchemy as sql
from collections import defaultdict
from .configuration import Configuration

package_directory = os.path.dirname(os.path.abspath(__file__))

type_mapping = defaultdict(lambda: sql.types.PickleType,
                           {"int": sql.types.Integer,
                            "float": sql.types.Float,
                            "str": sql.types.String,
                            })


def get_sql_types(type_config=None):
    """Get the SQL types associated to configuration parameters from a string-based typeconfig"""
    if type_config is None:
        type_config = Configuration.from_yaml("integrator_config_types.yaml")

    if isinstance(type_config, Configuration):
        type_config = type_config.as_flat_dict()

    sql_types = dict([(key, type_mapping[value]) for key, value in type_config.items()])

    return sql_types


def infer_sql_types(config):
    """Get the SQL types qssociated to configuration parameters based on type inference"""
    if isinstance(config, Configuration):
        config = config.as_flat_dict()
    return dict([(key, type_mapping[type(value).__name__]) for key, value in config.items()])
