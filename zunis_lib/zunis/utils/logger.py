import logging


verbosity_levels = {0: logging.ERROR,
                    1: logging.WARNING,
                    2: logging.INFO,
                    3: logging.DEBUG,
                    "ERROR": logging.ERROR,
                    "WARNING": logging.WARNING,
                    "INFO": logging.INFO,
                    "DEBUG": logging.DEBUG,
                    }


def set_verbosity(obj, level):
    if level is None:
        return
    level_key = level
    if isinstance(level, str):
        level_key = level.upper()
    try:
        obj.logger.setLevel(verbosity_levels[level_key])
    except KeyError:
        obj.logger.warning(f"Verbosity level {level} not recognized. Ignored.")
