"""Facilities for interfacing dataframes to SQL databases"""
import logging
import pickle

import pandas as pd
import sqlalchemy as sql

logger = logging.getLogger(__name__)


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


def read_results(dbname="", tablename="results", type_config=None):
    """Read results from a SQLite database to a dataframe and reconstruct pickled objects

    Parameters
    ----------
    dbname: str
    tablename: str
    type_config: dict, None

    Returns
    -------

    """

    engine = sql.create_engine(f"sqlite:///{dbname}")

    df = pd.read_sql(tablename, con=engine)

    if type_config is not None:
        for key, value in type_config.items():
            if value == sql.PickleType:
                try:
                    df[key] = df[key].apply(pickle.loads)
                except pickle.UnpicklingError as e:
                    logger.error(f"Could not unpickle column {key}, leaving it as-is")
                    logger.error(e)
