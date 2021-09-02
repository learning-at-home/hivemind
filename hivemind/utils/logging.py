import logging
import os
import sys

loglevel = os.getenv("LOGLEVEL", "INFO")

_env_colors = os.getenv("HIVEMIND_COLORS")
if _env_colors is not None:
    use_colors = _env_colors.lower() == "true"
else:
    use_colors = sys.stderr.isatty()


class CustomFormatter(logging.Formatter):
    """
    A formatter that allows a log time and caller info to be overridden via
    ``logger.log(level, message, extra={"origin_created": ..., "caller": ...})``.
    """

    # Details: https://en.wikipedia.org/wiki/ANSI_escape_code#Colors
    _LEVEL_TO_COLOR = {
        logging.DEBUG: "35",  # Purple
        logging.INFO: "34",  # Blue
        logging.WARNING: "38;5;208",  # Orange (8-bit palette)
        logging.ERROR: "31",  # Red
        logging.CRITICAL: "31",  # Red
    }

    def format(self, record: logging.LogRecord) -> str:
        if hasattr(record, "origin_created"):
            record.created = record.origin_created
            record.msecs = (record.created - int(record.created)) * 1000

        if not hasattr(record, "caller"):
            record.caller = f"{record.name}.{record.funcName}:{record.lineno}"

        color_code = self._LEVEL_TO_COLOR[record.levelno]
        if use_colors:
            record.bold_color = f"\033[{color_code};1m"
            record.end_color = "\033[39m"
            record.end_bold = "\033[0m"
        else:
            record.bold_color = record.end_color = record.end_bold = ""

        return super().format(record)


def get_logger(module_name: str) -> logging.Logger:
    # trim package name
    name_without_prefix = ".".join(module_name.split(".")[1:])

    logging.addLevelName(logging.WARNING, "WARN")
    formatter = CustomFormatter(
        fmt="{asctime}.{msecs:03.0f}  {bold_color}{levelname}{end_color} {caller}{end_bold}  {message}",
        style="{",
        datefmt="%b %d %H:%M:%S",
    )
    handler = logging.StreamHandler()
    handler.setFormatter(formatter)
    logger = logging.getLogger(name_without_prefix)
    logger.setLevel(loglevel)
    logger.addHandler(handler)
    logger.propagate = False
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
