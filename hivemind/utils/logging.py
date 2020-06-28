import os

from aiologger import Logger
from aiologger.formatters.base import Formatter


def get_logger(module_name):
    name_without_prefix = '.'.join(module_name.split('.')[1:])
    loglevel = os.getenv('LOGLEVEL', 'INFO')

    formatter = Formatter(fmt='[{asctime}.{msecs:03.0f}][{levelname}][{name}.{funcName}:{lineno}] {message}', style='{',
                          datefmt='%Y/%m/%d %H:%M:%S')
    logger = Logger.with_default_handlers(name=name_without_prefix, level=loglevel, formatter=formatter)
    return logger
