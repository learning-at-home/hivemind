"""
Originally taken from: https://github.com/mhchia/py-libp2p-daemon-bindings
Licence: MIT
Author: Kevin Mai-Husan Chia
"""

import hashlib
from typing import Any, Sequence, Union

import base58
import multihash
from cryptography.hazmat.primitives import serialization
from multiaddr import Multiaddr

from hivemind.proto import crypto_pb2, p2pd_pb2


class PeerID:
    def __init__(self, peer_id_bytes: bytes) -> None:
        self._bytes = peer_id_bytes
        self._b58_str = base58.b58encode(self._bytes).decode()

    def to_bytes(self) -> bytes:
        return self._bytes

    def to_base58(self) -> str:
        return self._b58_str

    def __repr__(self) -> str:
        return f"<libp2p.peer.id.ID ({self.to_base58()})>"

    def __str__(self):
        return self.to_base58()

    def pretty(self):
        return self.to_base58()

    def to_string(self):
        return self.to_base58()

    def __eq__(self, other: object) -> bool:
        if isinstance(other, str):
            return self.to_base58() == other
        elif isinstance(other, bytes):
            return self._bytes == other
        elif isinstance(other, PeerID):
            return self._bytes == other._bytes
        else:
            return False

    def __lt__(self, other: object) -> bool:
        if not isinstance(other, PeerID):
            raise TypeError(f"'<' not supported between instances of 'PeerID' and '{type(other)}'")

        return self.to_base58() < other.to_base58()

    def __hash__(self) -> int:
        return hash(self._bytes)

    @classmethod
    def from_base58(cls, base58_id: str) -> "PeerID":
        peer_id_bytes = base58.b58decode(base58_id)
        return cls(peer_id_bytes)

    @classmethod
    def from_identity(cls, data: bytes) -> "PeerID":
        """
        See [1] for the specification of how this conversion should happen.

        [1] https://github.com/libp2p/specs/blob/master/peer-ids/peer-ids.md#peer-ids
        """
        key_data = crypto_pb2.PrivateKey.FromString(data).data
        private_key = serialization.load_der_private_key(key_data, password=None)

        encoded_public_key = private_key.public_key().public_bytes(
            encoding=serialization.Encoding.DER,
            format=serialization.PublicFormat.SubjectPublicKeyInfo,
        )
        encoded_public_key = crypto_pb2.PublicKey(
            key_type=crypto_pb2.RSA,
            data=encoded_public_key,
        ).SerializeToString()

        encoded_digest = multihash.encode(
            hashlib.sha256(encoded_public_key).digest(),
            multihash.coerce_code("sha2-256"),
        )
        return cls(encoded_digest)


def sha256_digest(data: Union[str, bytes]) -> bytes:
    if isinstance(data, str):
        data = data.encode("utf8")
    return hashlib.sha256(data).digest()


class StreamInfo:
    def __init__(self, peer_id: PeerID, addr: Multiaddr, proto: str) -> None:
        self.peer_id = peer_id
        self.addr = addr
        self.proto = proto

    def __repr__(self) -> str:
        return f"<StreamInfo peer_id={self.peer_id} addr={self.addr} proto={self.proto}>"

    def to_protobuf(self) -> p2pd_pb2.StreamInfo:
        pb_msg = p2pd_pb2.StreamInfo(peer=self.peer_id.to_bytes(), addr=self.addr.to_bytes(), proto=self.proto)
        return pb_msg

    @classmethod
    def from_protobuf(cls, pb_msg: p2pd_pb2.StreamInfo) -> "StreamInfo":
        stream_info = cls(peer_id=PeerID(pb_msg.peer), addr=Multiaddr(pb_msg.addr), proto=pb_msg.proto)
        return stream_info


class PeerInfo:
    def __init__(self, peer_id: PeerID, addrs: Sequence[Multiaddr]) -> None:
        self.peer_id = peer_id
        self.addrs = list(addrs)

    def __eq__(self, other: Any) -> bool:
        return isinstance(other, PeerInfo) and self.peer_id == other.peer_id and self.addrs == other.addrs

    @classmethod
    def from_protobuf(cls, peer_info_pb: p2pd_pb2.PeerInfo) -> "PeerInfo":
        peer_id = PeerID(peer_info_pb.id)
        addrs = [Multiaddr(addr) for addr in peer_info_pb.addrs]
        return PeerInfo(peer_id, addrs)

    def __str__(self):
        return f"{self.peer_id.pretty()} {','.join(str(a) for a in self.addrs)}"

    def __repr__(self):
        return f"PeerInfo(peer_id={repr(self.peer_id)}, addrs={repr(self.addrs)})"
