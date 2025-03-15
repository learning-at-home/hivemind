import base64
import struct

SIZE = 96
IS_PATH = False


def to_bytes(proto, string):
    addr = string.split(":")
    if len(addr) != 2:
        raise ValueError("Does not contain a port number")

    # onion address without the ".onion" substring
    if len(addr[0]) != 16:
        raise ValueError("Invalid onion host address length (must be 16 characters)")
    try:
        onion_host_bytes = base64.b32decode(addr[0].upper())
    except Exception as exc:
        raise ValueError("Cannot decode {0!r} as base32: {1}".format(addr[0], exc)) from exc

    # onion port number
    try:
        port = int(addr[1], 10)
    except ValueError as exc:
        raise ValueError("Port number is not a base 10 integer") from exc
    if port not in range(1, 65536):
        raise ValueError("Port number is not in range(1, 65536)")

    return b"".join((onion_host_bytes, struct.pack(">H", port)))


def to_string(proto, buf):
    addr_bytes, port_bytes = (buf[:-2], buf[-2:])
    addr = base64.b32encode(addr_bytes).decode("ascii").lower()
    port = str(struct.unpack(">H", port_bytes)[0])
    return ":".join([addr, port])
