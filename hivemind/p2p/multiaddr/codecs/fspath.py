import os

from . import LENGTH_PREFIXED_VAR_SIZE


SIZE = LENGTH_PREFIXED_VAR_SIZE
IS_PATH = True


def to_bytes(proto, string):
    return os.fsencode(string)


def to_string(proto, buf):
    return os.fsdecode(buf)
