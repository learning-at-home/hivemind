import logging
import os
import sys
from typing import Optional

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


def get_logger(name: Optional[str] = None) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(loglevel)
    logger.propagate = False

    if not logger.hasHandlers():
        formatter = CustomFormatter(
            fmt="{asctime}.{msecs:03.0f} [{bold}{levelcolor}{levelname}{reset}] [{bold}{caller}{reset}] {message}",
            style="{",
            datefmt="%b %d %H:%M:%S",
        )
        handler = logging.StreamHandler()
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    return logger


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
