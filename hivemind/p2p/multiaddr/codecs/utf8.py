from __future__ import absolute_import

from . import LENGTH_PREFIXED_VAR_SIZE


SIZE = LENGTH_PREFIXED_VAR_SIZE
IS_PATH = False


def to_bytes(proto, string):
    if len(string) == 0:
        raise ValueError("{0} value must not be empty".format(proto.name))
    return string.encode('utf-8')


def to_string(proto, buf):
    if len(buf) == 0:
        raise ValueError("invalid length (should be > 0)")
    return buf.decode('utf-8')
