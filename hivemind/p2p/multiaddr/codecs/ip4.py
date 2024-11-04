import netaddr


SIZE = 32
IS_PATH = False


def to_bytes(proto, string):
    return netaddr.IPAddress(string, version=4).packed


def to_string(proto, buf):
    return str(netaddr.IPAddress(int.from_bytes(buf, byteorder='big'), version=4))
