import hashlib
from typing import Union, List, Sequence, Any

import base58
import multihash

from multiaddr import Multiaddr, protocols
from pb import p2pd_pb2

from keys import PublicKey

# NOTE: On inlining...
# See: https://github.com/libp2p/specs/issues/138
# NOTE: enabling to be interoperable w/ the Go implementation
ENABLE_INLINING = True
MAX_INLINE_KEY_LENGTH = 42

IDENTITY_MULTIHASH_CODE = 0x00

if ENABLE_INLINING:

    class IdentityHash:
        _digest: bytes

        def __init__(self) -> None:
            self._digest = bytearray()

        def update(self, input: bytes) -> None:
            self._digest += input

        def digest(self) -> bytes:
            return self._digest

    multihash.FuncReg.register(
        IDENTITY_MULTIHASH_CODE, "identity", hash_new=lambda: IdentityHash()
    )


class ID:
    _bytes: bytes
    _xor_id: int = None
    _b58_str: str = None

    def __init__(self, peer_id_bytes: bytes) -> None:
        self._bytes = peer_id_bytes

    @property
    def xor_id(self) -> int:
        if not self._xor_id:
            self._xor_id = int(sha256_digest(self._bytes).hex(), 16)
        return self._xor_id

    def to_bytes(self) -> bytes:
        return self._bytes

    def to_base58(self) -> str:
        if not self._b58_str:
            self._b58_str = base58.b58encode(self._bytes).decode()
        return self._b58_str

    def __repr__(self) -> str:
        return f"<libp2p.peer.id.ID ({self!s})>"

    __str__ = pretty = to_string = to_base58

    def __eq__(self, other: object) -> bool:
        if isinstance(other, str):
            return self.to_base58() == other
        elif isinstance(other, bytes):
            return self._bytes == other
        elif isinstance(other, ID):
            return self._bytes == other._bytes
        else:
            return NotImplemented

    def __hash__(self) -> int:
        return hash(self._bytes)

    @classmethod
    def from_base58(cls, b58_encoded_peer_id_str: str) -> "ID":
        peer_id_bytes = base58.b58decode(b58_encoded_peer_id_str)
        pid = ID(peer_id_bytes)
        return pid

    @classmethod
    def from_pubkey(cls, key: PublicKey) -> "ID":
        serialized_key = key.serialize()
        algo = multihash.Func.sha2_256
        if ENABLE_INLINING and len(serialized_key) <= MAX_INLINE_KEY_LENGTH:
            algo = IDENTITY_MULTIHASH_CODE
        mh_digest = multihash.digest(serialized_key, algo)
        return cls(mh_digest.encode())


def sha256_digest(data: Union[str, bytes]) -> bytes:
    if isinstance(data, str):
        data = data.encode("utf8")
    return hashlib.sha256(data).digest()


class StreamInfo:
    peer_id: ID
    addr: Multiaddr
    proto: str

    def __init__(self, peer_id: ID, addr: Multiaddr, proto: str) -> None:
        self.peer_id = peer_id
        self.addr = addr
        self.proto = proto

    def __repr__(self) -> str:
        return (
            f"<StreamInfo peer_id={self.peer_id} addr={self.addr} proto={self.proto}>"
        )

    def to_pb(self) -> p2pd_pb2.StreamInfo:
        pb_msg = p2pd_pb2.StreamInfo(
            peer=self.peer_id.to_bytes(), addr=self.addr.to_bytes(), proto=self.proto
        )
        return pb_msg

    @classmethod
    def from_pb(cls, pb_msg: p2pd_pb2.StreamInfo) -> "StreamInfo":
        stream_info = cls(
            peer_id=ID(pb_msg.peer), addr=Multiaddr(pb_msg.addr), proto=pb_msg.proto
        )
        return stream_info


class PeerInfoLibP2P:
    peer_id: ID
    addrs: List[Multiaddr]

    def __init__(self, peer_id: ID, addrs: Sequence[Multiaddr]) -> None:
        self.peer_id = peer_id
        self.addrs = list(addrs)

    def __eq__(self, other: Any) -> bool:
        return (
            isinstance(other, PeerInfo)
            and self.peer_id == other.peer_id
            and self.addrs == other.addrs
        )


def info_from_p2p_addr(addr: Multiaddr) -> PeerInfoLibP2P:
    if not addr:
        raise InvalidAddrError("`addr` should not be `None`")

    parts = addr.split()
    if not parts:
        raise InvalidAddrError(
            f"`parts`={parts} should at least have a protocol `P_P2P`"
        )

    p2p_part = parts[-1]
    last_protocol_code = p2p_part.protocols()[0].code
    if last_protocol_code != protocols.P_P2P:
        raise InvalidAddrError(
            f"The last protocol should be `P_P2P` instead of `{last_protocol_code}`"
        )

    # make sure the /p2p value parses as a peer.ID
    peer_id_str: str = p2p_part.value_for_protocol(protocols.P_P2P)
    peer_id: ID = ID.from_base58(peer_id_str)

    # we might have received just an / p2p part, which means there's no addr.
    if len(parts) > 1:
        addr = Multiaddr.join(*parts[:-1])

    return PeerInfo(peer_id, [addr])


class InvalidAddrError(ValueError):
    pass


class PeerInfo(PeerInfoLibP2P):
    @classmethod
    def from_pb(cls, peer_info_pb: p2pd_pb2.PeerInfo) -> PeerInfoLibP2P:
        peer_id = ID(peer_info_pb.id)
        addrs = [Multiaddr(addr) for addr in peer_info_pb.addrs]
        return PeerInfo(peer_id, addrs)

    def __str__(self):
        return self.peer_id.pretty() + " " + ",".join(str(a) for a in self.addrs)
