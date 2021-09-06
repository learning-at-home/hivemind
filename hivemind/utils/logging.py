import logging
import os
import sys
import threading
from enum import Enum
from typing import Optional, Union

logging.addLevelName(logging.WARNING, "WARN")

loglevel = os.getenv("LOGLEVEL", "INFO")

_env_colors = os.getenv("HIVEMIND_COLORS")
if _env_colors is not None:
    use_colors = _env_colors.lower() == "true"
else:
    use_colors = sys.stderr.isatty()


class TextStyle:
    """
    ANSI escape codes. Details: https://en.wikipedia.org/wiki/ANSI_escape_code#Colors
    """

    RESET = "\033[0m"
    BOLD = "\033[1m"
    RED = "\033[31m"
    BLUE = "\033[34m"
    PURPLE = "\033[35m"
    ORANGE = "\033[38;5;208m"  # From 8-bit palette

    if not use_colors:
        # Set the constants above to empty strings
        _codes = locals()
        _codes.update({_name: "" for _name in list(_codes) if _name.isupper()})


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

        if not hasattr(record, "caller"):
            module_path = record.name.split(".")
            if module_path[0] == "hivemind":
                module_path = module_path[1:]
            record.caller = f"{'.'.join(module_path)}.{record.funcName}:{record.lineno}"

        # Aliases for the format argument
        record.levelcolor = self._LEVEL_TO_COLOR[record.levelno]
        record.bold = TextStyle.BOLD
        record.reset = TextStyle.RESET

        return super().format(record)


_PACKAGE_NAME = __name__.split(".")[0]

_init_lock = threading.RLock()
_current_mode = HandlerMode.NOWHERE  # This is the initial state before module initialization but not an actual default
_default_handler = None


def _initialize_if_necessary():
    global _current_mode, _default_handler

    with _init_lock:
        if _default_handler is not None:
            return

        formatter = CustomFormatter(
            fmt="{asctime}.{msecs:03.0f} [{bold}{levelcolor}{levelname}{reset}] [{bold}{caller}{reset}] {message}",
            style="{",
            datefmt="%b %d %H:%M:%S",
        )
        _default_handler = logging.StreamHandler()
        _default_handler.setFormatter(formatter)

        use_hivemind_log_handler(HandlerMode.IN_HIVEMIND)  # Overriding it to the desired default


def get_logger(name: Optional[str] = None) -> logging.Logger:
    """
    Same as ``logging.getLogger()`` but ensures that the default log handler is initialized.
    """

    _initialize_if_necessary()
    return logging.getLogger(name)


def _enable_default_handler(name: str) -> None:
    logger = get_logger(name)
    logger.addHandler(_default_handler)
    logger.propagate = False
    logger.setLevel(loglevel)


def _disable_default_handler(name: str) -> None:
    logger = get_logger(name)
    logger.removeHandler(_default_handler)
    logger.propagate = True
    logger.setLevel(logging.NOTSET)


class HandlerMode(Enum):
    NOWHERE = 0
    IN_HIVEMIND = 1
    IN_ROOT_LOGGER = 2


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

    if _current_mode == HandlerMode.IN_HIVEMIND:
        _disable_default_handler(_PACKAGE_NAME)
    elif _current_mode == HandlerMode.IN_ROOT_LOGGER:
        _disable_default_handler(None)

    _current_mode = where

    if _current_mode == HandlerMode.IN_HIVEMIND:
        _enable_default_handler(_PACKAGE_NAME)
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
