"""Library of the Pytorch-Flows project"""
import logging
import sys

# Ensure that no submodule loggers outputs anything, unless explicitly setup by the user
logging.getLogger(__name__).addHandler(logging.NullHandler())


def setup_std_stream_rootlogger(min_level=None, debug=False):
    """Set the root logger to split output between stdout and stderr
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

    logger = logging.getLogger()
    logger.setLevel(min_level)

    h1 = logging.StreamHandler(sys.stdout)
    h1.setLevel(min_level)
    h1.addFilter(InfoFilter())
    h2 = logging.StreamHandler()
    h2.setLevel(logging.WARNING)

    logger.addHandler(h1)
    logger.addHandler(h2)
