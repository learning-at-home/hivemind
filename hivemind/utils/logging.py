import logging
import os


def get_logger(module_name: str) -> logging.Logger:
    # trim package name
    name_without_prefix = ".".join(module_name.split(".")[1:])
    loglevel = os.getenv("LOGLEVEL", "INFO")

    logging.addLevelName(logging.WARNING, "WARN")
    formatter = logging.Formatter(
        fmt="[{asctime}.{msecs:03.0f}][{levelname}][{name}.{funcName}:{lineno}] {message}",
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
