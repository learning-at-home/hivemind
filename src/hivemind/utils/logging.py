import logging
import os
import sys
import threading
from enum import Enum
from typing import Any, Optional, Union


def in_ipython() -> bool:
    """Check if the code is run in IPython, Jupyter, or Colab"""

    try:
        __IPYTHON__
        return True
    except NameError:
        return False


logging.addLevelName(logging.WARNING, "WARN")
loglevel = os.getenv("HIVEMIND_LOGLEVEL", "INFO")

TRUE_CONSTANTS = ["TRUE", "1"]

_env_colors = os.getenv("HIVEMIND_COLORS")
if _env_colors is not None:
    use_colors = _env_colors.upper() in TRUE_CONSTANTS
else:
    use_colors = sys.stderr.isatty() or in_ipython()

_env_log_caller = os.getenv("HIVEMIND_ALWAYS_LOG_CALLER", "0")
always_log_caller = _env_log_caller.upper() in TRUE_CONSTANTS


class HandlerMode(Enum):
    NOWHERE = 0
    IN_HIVEMIND = 1
    IN_ROOT_LOGGER = 2


_init_lock = threading.RLock()
_current_mode = HandlerMode.IN_HIVEMIND
_default_handler = None


class _DisableIfNoColors(type):
    def __getattribute__(self, name: str) -> Any:
        if name.isupper() and not use_colors:
            return ""
        return super().__getattribute__(name)


class TextStyle(metaclass=_DisableIfNoColors):
    """
    ANSI escape codes. Details: https://en.wikipedia.org/wiki/ANSI_escape_code#Colors
    """

    RESET = "\033[0m"
    BOLD = "\033[1m"
    RED = "\033[31m"
    BLUE = "\033[34m"
    PURPLE = "\033[35m"
    ORANGE = "\033[38;5;208m"  # From 8-bit palette


class CustomFormatter(logging.Formatter):
    """
    A formatter that allows a log time and caller info to be overridden via
    ``logger.log(level, message, extra={"origin_created": ..., "caller": ...})``.
    """

    # Details: https://en.wikipedia.org/wiki/ANSI_escape_code#Colors
    _LEVEL_TO_COLOR = {
        logging.DEBUG: TextStyle.PURPLE,
        logging.INFO: TextStyle.BLUE,
        logging.WARNING: TextStyle.ORANGE,
        logging.ERROR: TextStyle.RED,
        logging.CRITICAL: TextStyle.RED,
    }

    def format(self, record: logging.LogRecord) -> str:
        if hasattr(record, "origin_created"):
            record.created = record.origin_created
            record.msecs = (record.created - int(record.created)) * 1000

        if record.levelno != logging.INFO or always_log_caller:
            if not hasattr(record, "caller"):
                record.caller = f"{record.name}.{record.funcName}:{record.lineno}"
            record.caller_block = f" [{TextStyle.BOLD}{record.caller}{TextStyle.RESET}]"
        else:
            record.caller_block = ""

        # Aliases for the format argument
        record.levelcolor = self._LEVEL_TO_COLOR[record.levelno]
        record.bold = TextStyle.BOLD
        record.reset = TextStyle.RESET

        return super().format(record)


def _initialize_if_necessary():
    global _current_mode, _default_handler

    with _init_lock:
        if _default_handler is not None:
            return

        formatter = CustomFormatter(
            fmt="{asctime}.{msecs:03.0f} [{bold}{levelcolor}{levelname}{reset}]{caller_block} {message}",
            style="{",
            datefmt="%b %d %H:%M:%S",
        )
        _default_handler = logging.StreamHandler()
        _default_handler.setFormatter(formatter)

        _enable_default_handler("hivemind")


def get_logger(name: Optional[str] = None) -> logging.Logger:
    """
    Same as ``logging.getLogger()`` but ensures that the default hivemind log handler is initialized.

    :note: By default, the hivemind log handler (that reads the ``HIVEMIND_LOGLEVEL`` env variable and uses
           the colored log formatter) is only applied to messages logged inside the hivemind package.
           If you want to extend this handler to other loggers in your application, call
           ``use_hivemind_log_handler("in_root_logger")``.
    """

    _initialize_if_necessary()
    return logging.getLogger(name)


def _enable_default_handler(name: Optional[str]) -> None:
    logger = get_logger(name)

    # Remove the extra default handler in the Colab's default logger before adding a new one
    if isinstance(logger, logging.RootLogger):
        for handler in list(logger.handlers):
            if isinstance(handler, logging.StreamHandler) and handler.stream is sys.stderr:
                logger.removeHandler(handler)

    logger.addHandler(_default_handler)
    logger.propagate = False
    logger.setLevel(loglevel)


def _disable_default_handler(name: Optional[str]) -> None:
    logger = get_logger(name)
    logger.removeHandler(_default_handler)
    logger.propagate = True
    logger.setLevel(logging.NOTSET)


def use_hivemind_log_handler(where: Union[HandlerMode, str]) -> None:
    """
    Choose loggers where the default hivemind log handler is applied. Options for the ``where`` argument are:

    * "in_hivemind" (default): Use the hivemind log handler in the loggers of the ``hivemind`` package.
                               Don't propagate their messages to the root logger.
    * "nowhere": Don't use the hivemind log handler anywhere.
                 Propagate the ``hivemind`` messages to the root logger.
    * "in_root_logger": Use the hivemind log handler in the root logger
                        (that is, in all application loggers until they disable propagation to the root logger).
                        Propagate the ``hivemind`` messages to the root logger.

    The options may be defined as strings (case-insensitive) or values from the HandlerMode enum.
    """

    global _current_mode

    if isinstance(where, str):
        # We allow `where` to be a string, so a developer does not have to import the enum for one usage
        where = HandlerMode[where.upper()]

    _initialize_if_necessary()

    if where == _current_mode:
        return

    if _current_mode == HandlerMode.IN_HIVEMIND:
        _disable_default_handler("hivemind")
    elif _current_mode == HandlerMode.IN_ROOT_LOGGER:
        _disable_default_handler(None)

    _current_mode = where

    if _current_mode == HandlerMode.IN_HIVEMIND:
        _enable_default_handler("hivemind")
    elif _current_mode == HandlerMode.IN_ROOT_LOGGER:
        _enable_default_handler(None)


def golog_level_to_python(level: str) -> int:
    level = level.upper()
    if level in ["DPANIC", "PANIC", "FATAL"]:
        return logging.CRITICAL

    level = logging.getLevelName(level)
    if not isinstance(level, int):
        raise ValueError(f"Unknown go-log level: {level}")
    return level


def python_level_to_golog(level: str) -> str:
    if not isinstance(level, str):
        raise ValueError("`level` is expected to be a Python log level in the string form")

    if level == "CRITICAL":
        return "FATAL"
    if level == "WARNING":
        return "WARN"
    return level
