"""Facilities for interfacing dataframes to SQL databases"""
import logging
import pickle

import pandas as pd
import sqlalchemy as sql
from utils.repo_info import get_git_summary

logger = logging.getLogger(__name__)


def append_dataframe_to_sqlite(dataframe, dbname="", tablename="results", dtypes=None, log_git_info=True):
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

    if log_git_info:
        git_info = get_git_summary()
        if "git_info" in dataframe.columns:
            dataframe["git_info"] = dataframe["git_info"].fillna(git_info)
        else:
            dataframe["git_info"] = git_info
        dtypes["git_info"] = sql.String

    if "extra_data" not in dataframe:
        dataframe["extra_data"] = [()] * len(dataframe)
    dtypes["extra_data"] = sql.types.PickleType

    if "value_history" in dataframe:
        dtypes["value_history"] = sql.types.PickleType
    if "target_history" in dataframe:
        dtypes["target_history"] = sql.types.PickleType

    dataframe.to_sql(tablename, con=engine, index=False, if_exists="append", dtype=dtypes)


def read_pkl_sql(dbname="", tablename="results", dtypes=None):
    """Read results from a SQLite database to a dataframe and reconstruct pickled objects

    Parameters
    ----------
    dbname: str
    tablename: str
    dtypes: dict, None

    Returns
    -------
        pd.DataFrame
    """

    engine = sql.create_engine(f"sqlite:///{dbname}")

    df = pd.read_sql(tablename, con=engine)

    if dtypes is not None:
        for key, value in dtypes.items():
            if value == sql.PickleType:
                try:
                    df[key] = df[key].apply(lambda x: None if x is None else pickle.loads(x))
                except (pickle.UnpicklingError, TypeError) as e:
                    logger.error(f"Could not unpickle column {key}, leaving it as-is")
                    logger.error(e)

    df.columns = df.columns.astype(str)
    return df
