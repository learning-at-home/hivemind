import idna

from . import LENGTH_PREFIXED_VAR_SIZE


SIZE = LENGTH_PREFIXED_VAR_SIZE
IS_PATH = False


def to_bytes(proto, string):
    return idna.uts46_remap(string).encode("utf-8")


def to_string(proto, buf):
    string = buf.decode("utf-8")
    for label in string.split("."):
        idna.check_label(label)
    return string
