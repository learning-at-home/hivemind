import importlib


# These are special sizes
LENGTH_PREFIXED_VAR_SIZE = -1


class NoneCodec:
    SIZE = 0
    IS_PATH = False


CODEC_CACHE = {}


def codec_by_name(name):
    if name is None:  # Special “do nothing – expect nothing” pseudo-codec
        return NoneCodec
    codec = CODEC_CACHE.get(name)
    if not codec:
        codec = CODEC_CACHE[name] = importlib.import_module(".{0}".format(name), __name__)
    return codec
