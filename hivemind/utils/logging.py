import logging
import os


class OverridableLogger(logging.Logger):
    """
    A logger that permits to override LogRecord properties via the ``extra`` dictionary.
    Used when reporting log messages originated elsewhere (e.g. in p2pd).
    Example: ``logger.debug("message", extra={"funcName": "new_value"})``
    """

    def makeRecord(self, name, level, fn, lno, msg, args, exc_info,
                   func=None, extra=None, sinfo=None):
        factory = logging.getLogRecordFactory()
        rv = factory(name, level, fn, lno, msg, args, exc_info, func, sinfo)
        rv.caller = f'{rv.name}.{rv.funcName}:{rv.lineno}'
        if extra is not None:
            rv.__dict__.update(extra)
        return rv


logging.setLoggerClass(OverridableLogger)
logging.addLevelName(logging.WARNING, "WARN")
loglevel = os.getenv("LOGLEVEL", "INFO")


def get_logger(module_name: str) -> logging.Logger:
    formatter = logging.Formatter(
        fmt="[{asctime}.{msecs:03.0f}][{levelname}][{caller}] {message}",
        style="{",
        datefmt="%Y/%m/%d %H:%M:%S",
    )

    handler = logging.StreamHandler()
    handler.setFormatter(formatter)

    name_without_prefix = ".".join(module_name.split(".")[1:])
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
        raise ValueError('`level` is expected to be a Python log level in the string form')

    if level == 'CRITICAL':
        return 'FATAL'
    if level == 'WARNING':
        return 'WARN'
    return level
