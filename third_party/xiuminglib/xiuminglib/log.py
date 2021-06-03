import logging
from platform import system


def get_logger(level=None):
    """Creates a logger for functions in the library.

    Args:
        level (str, optional): Logging level. Defaults to ``logging.INFO``.

    Returns:
        logging.Logger: Logger created.
    """
    if level is None:
        level = logging.INFO
    logging.basicConfig(level=level)
    logger = logging.getLogger()
    return logger


def _add_coloring_to_emit_ansi(fn):
    # Add methods we need to the class
    def new(*args):
        levelno = args[1].levelno
        if levelno >= 50:
            color = '\x1b[31m' # red
        elif levelno >= 40:
            color = '\x1b[31m' # red
        elif levelno >= 30:
            color = '\x1b[33m' # yellow
        elif levelno >= 20:
            color = '\x1b[32m' # green
        elif levelno >= 10:
            color = '\x1b[35m' # pink
        else:
            color = '\x1b[0m' # normal
        args[1].msg = color + args[1].msg + '\x1b[0m' # normal
        return fn(*args)
    return new


if system() == 'Windows':
    raise NotImplementedError(
        "This library has yet to be made Windows-compatible")

# All non-Windows platforms are supporting ANSI escapes so we use them
logging.StreamHandler.emit = _add_coloring_to_emit_ansi(
    logging.StreamHandler.emit)
