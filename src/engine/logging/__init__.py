import logging

from pytorch_lightning.utilities import rank_zero_only
from rich.logging import RichHandler

_default_level = logging.INFO


def list_loggers():
    rootlogger = logging.getLogger()
    print(rootlogger)
    for h in rootlogger.handlers:
        print("     %s" % h)

    for nm, lgr in logging.Logger.manager.loggerDict.items():
        print("+ [%-20s] %s " % (nm, lgr))
        if not isinstance(lgr, logging.PlaceHolder):
            for h in lgr.handlers:
                print("     %s" % h)


def get_rich_handler():
    root_logger = logging.getLogger()
    for h in root_logger.handlers:
        if h.__class__.__name__.find("RichHandler") >= 0:
            return h
    return None


@rank_zero_only
def set_default_level(level):
    global _default_level
    if isinstance(level, str):
        level = getattr(logging, level)
    _default_level = level


def get_logger(name="PROJECT", level=None) -> logging.Logger:
    """Initializes multi-GPU-friendly python logger."""

    if level is None:
        level = _default_level
    if isinstance(level, str):
        level = getattr(logging, level)

    logger = logging.getLogger(name)
    logger.setLevel(level)

    # this ensures all logging levels get marked with the rank zero decorator
    # otherwise logs would get multiplied for each GPU process in multi-GPU setup
    for level in (
        "debug",
        "info",
        "warning",
        "error",
        "exception",
        "fatal",
        "critical",
    ):
        setattr(logger, level, rank_zero_only(getattr(logger, level)))

    return logger


class RichHandlerExitAtFatal(RichHandler):
    def emit(self, record):
        super().emit(record)
        if record.levelno in (logging.FATAL, logging.CRITICAL):
            raise SystemExit(-1)
