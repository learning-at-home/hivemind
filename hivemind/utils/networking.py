import socket
from contextlib import closing
from ipaddress import ip_address
from typing import Optional, Sequence

from multiaddr import Multiaddr

Hostname, Port = str, int  # flavour types
Endpoint = str  # e.g. 1.2.3.4:1337 or [2a21:6с8:b192:2105]:8888, https://networkengineering.stackexchange.com/a/9435
LOCALHOST = "127.0.0.1"


def get_port(endpoint: Endpoint) -> Optional[Port]:
    """get port or None if port is undefined"""
    # TODO: find a standard way to get port, make sure it works in malformed ports
    try:
        return int(endpoint[endpoint.rindex(":") + 1 :], base=10)
    except ValueError:  # :* or not specified
        return None


def replace_port(endpoint: Endpoint, new_port: Port) -> Endpoint:
    assert endpoint.endswith(":*") or get_port(endpoint) is not None, endpoint
    return f"{endpoint[:endpoint.rindex(':')]}:{new_port}"


def strip_port(endpoint: Endpoint) -> Hostname:
    """Removes port from the end of endpoint. If port is not specified, does nothing"""
    maybe_port = endpoint[endpoint.rindex(":") + 1 :]
    return endpoint[: endpoint.rindex(":")] if maybe_port.isdigit() or maybe_port == "*" else endpoint


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


def choose_ip_address(
    maddrs: Sequence[Multiaddr], prefer_global: bool = True, protocol_priority: Sequence[str] = ("ip4", "ip6")
) -> Hostname:
    """
    Currently, some components of hivemind are not converted to work over libp2p and use classical networking.
    To allow other peers reach a server when needed, these components announce a machine's IP address.

    This function automatically selects the best IP address to announce among publicly visible multiaddrs
    of this machine identified by libp2p (typically, using the ``P2P.get_visible_maddrs()`` method),
    so a user does not need to define this address manually (unless the user wants to).

    The best IP address is chosen using the following logic:
      - Prefer IP addresses from global address blocks
        (in terms of https://docs.python.org/3/library/ipaddress.html#ipaddress.IPv4Address.is_global)
      - Among the IP addresses of the same globality status, prefer IPv4 addresses over IPv6

    If the default logic does not suit you, it is recommended to set the announced IP address manually.
    """

    for need_global in [prefer_global, not prefer_global]:
        for protocol in protocol_priority:
            for addr in maddrs:
                if protocol in addr.protocols():
                    value_for_protocol = addr[protocol]
                    if ip_address(value_for_protocol).is_global == need_global:
                        return value_for_protocol

    raise ValueError(f"No IP address found among given multiaddrs: {maddrs}")
