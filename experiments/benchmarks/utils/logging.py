"""Tools for setting up standard logging"""

import sys
import logging
from zunis import logger as zunis_logger
from zunis import logger_training as zunis_logger_training
from zunis import logger_integration as zunis_logger_integration
from hashlib import sha1
from time import time


def set_benchmark_logger(name="benchmark_logger",
                         zunis_level=logging.INFO,
                         zunis_integration_level=logging.INFO,
                         zunis_training_level=logging.WARNING):
    """Setup an application level logger outputting to file and limit integration/training output

    Parameters
    ----------
    name: str
        name of the benchmark logger and the output file ({name}.log)
    zunis_level: int
        logging level of the zunis logger
    zunis_integration_level: int
        logging level of the zunis integration logger.
    zunis_training_level: int
        logging level of the zunis training logger.

    Returns
    -------
        logging.Logger
        desired logger for the benchmark
    """

    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    # Get a hash of the current time since the epoch as an identifier of the current process
    timestamp = sha1(str(time()).encode()).hexdigest()[:7]
    handler = logging.FileHandler(f"./{name}:{timestamp}.log", 'a', 'utf-8')
    print(f"Logging into ./{name}:{timestamp}.log")
    handler.setFormatter(logging.Formatter("%(asctime)s %(name)s:%(levelname)s:%(message)s"))
    root_logger.addHandler(handler)

    zunis_logger.setLevel(zunis_level)
    zunis_logger_integration.setLevel(zunis_integration_level)
    zunis_logger_training.setLevel(zunis_training_level)


def set_benchmark_logger_debug(zunis_level=logging.INFO,
                               zunis_integration_level=logging.INFO,
                               zunis_training_level=logging.INFO):
    """Setup an application level logger outputting to stdout for debugging purposes
     and limit integration/training output

    Parameters
    ----------
    zunis_level: int
        logging level of the zunis logger
    zunis_integration_level: int
        logging level of the zunis integration logger.
    zunis_training_level: int
        logging level of the zunis training logger.

    """

    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)
    handler = logging.StreamHandler(stream=sys.stdout, )
    handler.setFormatter(logging.Formatter("%(levelname)s: %(message)s [%(name)s]"))
    root_logger.addHandler(handler)

    zunis_logger.setLevel(zunis_level)
    zunis_logger_integration.setLevel(zunis_integration_level)
    zunis_logger_training.setLevel(zunis_training_level)
