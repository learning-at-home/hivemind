from ipaddress import ip_address
from typing import List, Sequence

from multiaddr import Multiaddr

from hivemind.utils.logging import TextStyle, get_logger

LOCALHOST = "127.0.0.1"

logger = get_logger(__name__)


def choose_ip_address(
    maddrs: Sequence[Multiaddr], prefer_global: bool = True, protocol_priority: Sequence[str] = ("ip4", "ip6")
) -> str:
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


def log_visible_maddrs(visible_maddrs: List[Multiaddr], only_p2p: bool) -> None:
    if only_p2p:
        unique_addrs = {addr["p2p"] for addr in visible_maddrs}
        initial_peers = " ".join(f"/p2p/{addr}" for addr in unique_addrs)
    else:
        available_ips = [Multiaddr(addr) for addr in visible_maddrs if "ip4" in addr or "ip6" in addr]
        if available_ips:
            preferred_ip = choose_ip_address(available_ips)
            selected_maddrs = [addr for addr in visible_maddrs if preferred_ip in str(addr)]
        else:
            selected_maddrs = visible_maddrs
        initial_peers = " ".join(str(addr) for addr in selected_maddrs)

    logger.info(
        f"Running a DHT instance. To connect other peers to this one, use "
        f"{TextStyle.BOLD}{TextStyle.BLUE}--initial_peers {initial_peers}{TextStyle.RESET}"
    )
    logger.info(f"Full list of visible multiaddresses: {' '.join(str(addr) for addr in visible_maddrs)}")
