import logging
import os


loglevel = os.getenv("LOGLEVEL", "INFO")


class CustomFormatter(logging.Formatter):
    """
    A formatter that allows a log time and caller info to be overridden via
    ``logger.log(level, message, extra={"origin_created": ..., "caller": ...})``.
    """

    def format(self, record: logging.LogRecord) -> str:
        if hasattr(record, "origin_created"):
            record.created = record.origin_created
            record.msecs = (record.created - int(record.created)) * 1000

        if not hasattr(record, "caller"):
            record.caller = f"{record.name}.{record.funcName}:{record.lineno}"

        return super().format(record)


def get_logger(module_name: str) -> logging.Logger:
    # trim package name
    name_without_prefix = ".".join(module_name.split(".")[1:])

    logging.addLevelName(logging.WARNING, "WARN")
    formatter = CustomFormatter(
        fmt="[{asctime}.{msecs:03.0f}][{levelname}][{caller}] {message}",
        style="{",
        datefmt="%Y/%m/%d %H:%M:%S",
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
