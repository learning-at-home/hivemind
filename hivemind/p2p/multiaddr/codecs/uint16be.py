import struct


SIZE = 16
IS_PATH = False


def to_bytes(proto, string):
    try:
        return struct.pack('>H', int(string, 10))
    except ValueError as exc:
        raise ValueError("Not a base 10 integer") from exc
    except struct.error as exc:
        raise ValueError("Integer not in range(65536)") from exc


def to_string(proto, buf):
    if len(buf) != 2:
        raise ValueError("Invalid integer length (must be 2 bytes / 16 bits)")
    return str(struct.unpack('>H', buf)[0])
