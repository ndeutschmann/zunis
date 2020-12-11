"""Tools to convert python data (Record) to SQL data"""

import os
import sqlalchemy as sql
from collections import defaultdict
from zunis.utils.config.configuration import Configuration
from utils.data_storage.dataframe2sql import append_dataframe_to_sqlite

package_directory = os.path.dirname(os.path.abspath(__file__))

type_mapping = defaultdict(lambda: sql.types.PickleType,
                           {"int": sql.types.Integer,
                            "float": sql.types.Float,
                            "str": sql.types.String,
                            })


def infer_sql_types(config):
    """Get the SQL types qssociated to configuration parameters based on type inference"""
    if isinstance(config, Configuration):
        config = config.as_flat_dict()
    return dict([(key, type_mapping[type(value).__name__]) for key, value in config.items()])


def append_record_to_sqlite(record, dbname="", tablename="results", dtypes=None, log_git_info=True):
    """Append record as a SQLite database row

    Parameters
    ----------
    record
    dbname
    tablename
    dtypes

    """

    if dtypes is None:
        dtypes = infer_sql_types(record)

    append_dataframe_to_sqlite(record.as_dataframe(), dbname, tablename, dtypes, log_git_info)
