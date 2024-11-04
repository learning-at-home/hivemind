import varint

from . import exceptions
from .codecs import codec_by_name

__all__ = ("Protocol", "PROTOCOLS", "REGISTRY")


# source of protocols https://github.com/multiformats/multicodec/blob/master/table.csv#L382
# replicating table here to:
# 1. avoid parsing the csv
# 2. ensuring errors in the csv don't screw up code.
# 3. changing a number has to happen in two places.
P_IP4 = 0x04
P_IP6 = 0x29
P_IP6ZONE = 0x2A
P_TCP = 0x06
P_UDP = 0x0111
P_DCCP = 0x21
P_SCTP = 0x84
P_UDT = 0x012D
P_UTP = 0x012E
P_P2P = 0x01A5
P_HTTP = 0x01E0
P_HTTPS = 0x01BB
P_TLS = 0x01C0
P_QUIC = 0x01CC
P_QUIC1 = 0x01CD
P_WS = 0x01DD
P_WSS = 0x01DE
P_ONION = 0x01BC
P_ONION3 = 0x01BD
P_P2P_CIRCUIT = 0x0122
P_DNS = 0x35
P_DNS4 = 0x36
P_DNS6 = 0x37
P_DNSADDR = 0x38
P_P2P_WEBSOCKET_STAR = 0x01DF
P_P2P_WEBRTC_STAR = 0x0113
P_P2P_WEBRTC_DIRECT = 0x0114
P_UNIX = 0x0190


class Protocol:
    __slots__ = [
        "code",   # int
        "name",   # string
        "codec",  # string
    ]

    def __init__(self, code, name, codec):
        if not isinstance(code, int):
            raise TypeError("code must be an integer")
        if not isinstance(name, str):
            raise TypeError("name must be a string")
        if not isinstance(codec, str) and codec is not None:
            raise TypeError("codec must be a string or None")

        self.code = code
        self.name = name
        self.codec = codec

    @property
    def size(self):
        return codec_by_name(self.codec).SIZE

    @property
    def path(self):
        return codec_by_name(self.codec).IS_PATH

    @property
    def vcode(self):
        return varint.encode(self.code)

    def __eq__(self, other):
        if not isinstance(other, Protocol):
            return NotImplemented

        return all((self.code == other.code,
                    self.name == other.name,
                    self.codec == other.codec,
                    self.path == other.path))

    def __hash__(self):
        return self.code

    def __repr__(self):
        return "Protocol(code={code!r}, name={name!r}, codec={codec!r})".format(
            code=self.code,
            name=self.name,
            codec=self.codec,
        )


# List of multiaddr protocols supported by this module by default
PROTOCOLS = [
    Protocol(P_IP4, 'ip4', 'ip4'),
    Protocol(P_TCP, 'tcp', 'uint16be'),
    Protocol(P_UDP, 'udp', 'uint16be'),
    Protocol(P_DCCP, 'dccp', 'uint16be'),
    Protocol(P_IP6, 'ip6', 'ip6'),
    Protocol(P_IP6ZONE, 'ip6zone', 'utf8'),
    Protocol(P_DNS, 'dns', 'domain'),
    Protocol(P_DNS4, 'dns4', 'domain'),
    Protocol(P_DNS6, 'dns6', 'domain'),
    Protocol(P_DNSADDR, 'dnsaddr', 'domain'),
    Protocol(P_SCTP, 'sctp', 'uint16be'),
    Protocol(P_UDT, 'udt', None),
    Protocol(P_UTP, 'utp', None),
    Protocol(P_P2P, 'p2p', 'cid'),
    Protocol(P_ONION, 'onion', 'onion'),
    Protocol(P_ONION3, 'onion3', 'onion3'),
    Protocol(P_QUIC, 'quic', None),
    Protocol(P_QUIC1, 'quic-v1', None),
    Protocol(P_HTTP, 'http', None),
    Protocol(P_HTTPS, 'https', None),
    Protocol(P_TLS, 'tls', None),
    Protocol(P_WS, 'ws', None),
    Protocol(P_WSS, 'wss', None),
    Protocol(P_P2P_WEBSOCKET_STAR, 'p2p-websocket-star', None),
    Protocol(P_P2P_WEBRTC_STAR, 'p2p-webrtc-star', None),
    Protocol(P_P2P_WEBRTC_DIRECT, 'p2p-webrtc-direct', None),
    Protocol(P_P2P_CIRCUIT, 'p2p-circuit', None),
    Protocol(P_UNIX, 'unix', 'fspath'),
]


class ProtocolRegistry:
    """A collection of individual Multiaddr protocols indexed for fast lookup"""
    __slots__ = ("_codes_to_protocols", "_locked", "_names_to_protocols")

    def __init__(self, protocols=()):
        self._locked = False
        self._codes_to_protocols = {proto.code: proto for proto in protocols}
        self._names_to_protocols = {proto.name: proto for proto in protocols}

    def add(self, proto):
        """Add the given protocol description to this registry

        Raises
        ------
        ~multiaddr.exceptions.ProtocolRegistryLocked
            Protocol registry is locked and does not accept any new entries.

            You can use `.copy(unlock=True)` to copy an existing locked registry
            and unlock it.
        ~multiaddr.exceptions.ProtocolExistsError
            A protocol with the given name or code already exists.
        """
        if self._locked:
            raise exceptions.ProtocolRegistryLocked()

        if proto.name in self._names_to_protocols:
            raise exceptions.ProtocolExistsError(proto, "name")

        if proto.code in self._codes_to_protocols:
            raise exceptions.ProtocolExistsError(proto, "code")

        self._names_to_protocols[proto.name] = proto
        self._codes_to_protocols[proto.code] = proto
        return proto

    def add_alias_name(self, proto, alias_name):
        """Add an alternate name for an existing protocol description to the registry

        Raises
        ------
        ~multiaddr.exceptions.ProtocolRegistryLocked
            Protocol registry is locked and does not accept any new entries.

            You can use `.copy(unlock=True)` to copy an existing locked registry
            and unlock it.
        ~multiaddr.exceptions.ProtocolExistsError
            A protocol with the given name already exists.
        ~multiaddr.exceptions.ProtocolNotFoundError
            No protocol matching *proto* could be found.
        """
        if self._locked:
            raise exceptions.ProtocolRegistryLocked()

        proto = self.find(proto)
        assert self._names_to_protocols.get(proto.name) is proto, \
               "Protocol to alias must have already been added to the registry"

        if alias_name in self._names_to_protocols:
            raise exceptions.ProtocolExistsError(self._names_to_protocols[alias_name], "name")

        self._names_to_protocols[alias_name] = proto

    def add_alias_code(self, proto, alias_code):
        """Add an alternate code for an existing protocol description to the registry

        Raises
        ------
        ~multiaddr.exceptions.ProtocolRegistryLocked
            Protocol registry is locked and does not accept any new entries.

            You can use `.copy(unlock=True)` to copy an existing locked registry
            and unlock it.
        ~multiaddr.exceptions.ProtocolExistsError
            A protocol with the given code already exists.
        ~multiaddr.exceptions.ProtocolNotFoundError
            No protocol matching *proto* could be found.
        """
        if self._locked:
            raise exceptions.ProtocolRegistryLocked()

        proto = self.find(proto)
        assert self._codes_to_protocols.get(proto.code) is proto, \
               "Protocol to alias must have already been added to the registry"

        if alias_code in self._codes_to_protocols:
            raise exceptions.ProtocolExistsError(self._codes_to_protocols[alias_code], "name")

        self._codes_to_protocols[alias_code] = proto

    def lock(self):
        """Lock this registry instance to deny any further changes"""
        self._locked = True

    @property
    def locked(self):
        return self._locked

    def copy(self, *, unlock=False):
        """Create a copy of this protocol registry

        Arguments
        ---------
        unlock
            Create the copied registry unlocked even if the current one is locked?
        """
        registry = ProtocolRegistry()
        registry._locked = self._locked and not unlock
        registry._codes_to_protocols = self._codes_to_protocols.copy()
        registry._names_to_protocols = self._names_to_protocols.copy()
        return registry

    __copy__ = copy

    def find_by_name(self, name):
        """Look up a protocol by its human-readable name

        Raises
        ------
        ~multiaddr.exceptions.ProtocolNotFoundError
        """
        if name not in self._names_to_protocols:
            raise exceptions.ProtocolNotFoundError(name, "name")
        return self._names_to_protocols[name]

    def find_by_code(self, code):
        """Look up a protocol by its binary representation code

        Raises
        ------
        ~multiaddr.exceptions.ProtocolNotFoundError
        """
        if code not in self._codes_to_protocols:
            raise exceptions.ProtocolNotFoundError(code, "code")
        return self._codes_to_protocols[code]

    def find(self, proto):
        """Look up a protocol by its name or code, return existing protocol objects unchanged

        Raises
        ------
        ~multiaddr.exceptions.ProtocolNotFoundError
        """
        if isinstance(proto, Protocol):
            return proto
        elif isinstance(proto, str):
            return self.find_by_name(proto)
        elif isinstance(proto, int):
            return self.find_by_code(proto)
        else:
            raise TypeError("Protocol object, name or code expected, got {0!r}".format(proto))


REGISTRY = ProtocolRegistry(PROTOCOLS)
REGISTRY.add_alias_name("p2p", "ipfs")
REGISTRY.lock()


def protocol_with_name(name):
    return REGISTRY.find_by_name(name)


def protocol_with_code(code):
    return REGISTRY.find_by_code(code)


def protocol_with_any(proto):
    return REGISTRY.find(proto)


def protocols_with_string(string):
    """Return a list of protocols matching given string."""
    ret = []
    for name in string.split("/"):
        if len(name) > 0:
            ret.append(protocol_with_name(name))
    return ret
