import socket
from contextlib import closing


def get_free_port(params=(socket.AF_INET, socket.SOCK_STREAM), opt=(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)):
    """
    Finds a tcp port that can be occupied with a socket with *params and use *opt options.

    :note: Using this function is discouraged since it often leads to a race condition
           with the "Address is already in use" error if the code is run in parallel.
    """
    try:
        with closing(socket.socket(*params)) as sock:
            sock.bind(("", 0))
            sock.setsockopt(*opt)
            return sock.getsockname()[1]
    except Exception as e:
        raise e
