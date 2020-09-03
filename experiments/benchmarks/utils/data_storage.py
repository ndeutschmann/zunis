"""Data storage facilities

We use SQL to store experiment results
"""

import pickle
import sqlalchemy as sql

from utils.config.config_sql import infer_sql_types


def append_dataframe_to_sqlite(dataframe, dbname="", tablename="results", dtypes=None):
    """Append dataframe to a SQLite

    Parameters
    ----------
    dataframe: pandas.Dataframe
    dbname
    tablename
    types

    """
    engine = sql.create_engine(f"sqlite:///{dbname}")

    if dtypes is None:
        dtypes = dict()

    dataframe.to_sql(tablename, con=engine, index=False, if_exists="append", dtype=dtypes)


def append_record_to_sqlite(record, dbname="", tablename="results", dtypes=None):
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

    append_dataframe_to_sqlite(record.as_dataframe(), dbname, tablename, dtypes)
