import socket
from contextlib import closing

Hostname, Port = str, int  # flavour types
Endpoint = str  # e.g. 1.2.3.4:1337 or [2a21:6с8:b192:2105]:8888, https://networkengineering.stackexchange.com/a/9435
LOCALHOST = '127.0.0.1'


def find_open_port(params=(socket.AF_INET, socket.SOCK_STREAM), opt=(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)):
    """ Finds a tcp port that can be occupied with a socket with *params and use *opt options """
    try:
        with closing(socket.socket(*params)) as sock:
            sock.bind(('', 0))
            sock.setsockopt(*opt)
            return sock.getsockname()[1]
    except Exception as e:
        raise e
