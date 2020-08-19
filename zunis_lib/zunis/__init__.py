"""ZuNIS: Neural importance sampling library"""
import logging
import sys
from zunis.integration import integration_logger as ilogger
from zunis.training import training_logger as tlogger


logger = logging.getLogger(__name__)
"""Overall parent logger for the Zunis library"""
logger_integration = ilogger
"""Overall parent logger for all integration operations"""
logger_training = tlogger
"""Overall parent logger for all training operations"""

# Ensure that no submodule loggers outputs anything, unless explicitly setup by the user
logger.addHandler(logging.NullHandler())


def setup_std_stream_logger(min_level=None, debug=False, force=True):
    """Set the root logger of this module to split output between stdout and stderr
    Arguments:
        - min_level: minimum level output to stderr. Default is INFO
        - debug: overrides min_level to DEBUG

    Notes:
        stdout : [min_level, WARNING)
        stderr : [WARNING, +∞)
    """

    if min_level is None:
        min_level = logging.INFO

    if debug:
        min_level = logging.DEBUG

    class InfoFilter(logging.Filter):
        def filter(self, rec):
            return min_level <= rec.levelno < logging.WARNING

    logger.setLevel(min_level)

    h1 = logging.StreamHandler(sys.stdout)
    h1.setLevel(min_level)
    h1.addFilter(InfoFilter())
    h2 = logging.StreamHandler(sys.stderr)
    h2.setLevel(logging.WARNING)

    if len(logger.handlers) == 1 or force:
        logger.addHandler(h1)
        logger.addHandler(h2)
    else:
        logger.warning("The logger already has handlers setup none were added.")
        logger.warning("Use the `force` option to bypass this behavior.")
