import collections.abc

import varint

from . import exceptions, protocols

from .transforms import bytes_iter
from .transforms import string_to_bytes
from .transforms import bytes_to_string


__all__ = ("Multiaddr",)


class MultiAddrKeys(collections.abc.KeysView, collections.abc.Sequence):
    def __contains__(self, proto):
        proto = self._mapping.registry.find(proto)
        return collections.abc.Sequence.__contains__(self, proto)

    def __getitem__(self, idx):
        if idx < 0:
            idx = len(self)+idx
        for idx2, proto in enumerate(self):
            if idx2 == idx:
                return proto
        raise IndexError("Protocol list index out of range")

    __hash__ = collections.abc.KeysView._hash

    def __iter__(self):
        for _, proto, _, _ in bytes_iter(self._mapping.to_bytes()):
            yield proto


class MultiAddrItems(collections.abc.ItemsView, collections.abc.Sequence):
    def __contains__(self, item):
        proto, value = item
        proto = self._mapping.registry.find(proto)
        return collections.abc.Sequence.__contains__(self, (proto, value))

    def __getitem__(self, idx):
        if idx < 0:
            idx = len(self)+idx
        for idx2, item in enumerate(self):
            if idx2 == idx:
                return item
        raise IndexError("Protocol item list index out of range")

    def __iter__(self):
        for _, proto, codec, part in bytes_iter(self._mapping.to_bytes()):
            if codec.SIZE != 0:
                try:
                    # If we have an address, return it
                    yield proto, codec.to_string(proto, part)
                except Exception as exc:
                    raise exceptions.BinaryParseError(
                            str(exc),
                            self._mapping.to_bytes(),
                            proto.name,
                            exc,
                        ) from exc
            else:
                # We were given something like '/utp', which doesn't have
                # an address, so return None
                yield proto, None


class MultiAddrValues(collections.abc.ValuesView, collections.abc.Sequence):
    __contains__ = collections.abc.Sequence.__contains__

    def __getitem__(self, idx):
        if idx < 0:
            idx = len(self)+idx
        for idx2, proto in enumerate(self):
            if idx2 == idx:
                return proto
        raise IndexError("Protocol value list index out of range")

    def __iter__(self):
        for _, value in MultiAddrItems(self._mapping):
            yield value


class Multiaddr(collections.abc.Mapping):
    """Multiaddr is a representation of multiple nested internet addresses.

    Multiaddr is a cross-protocol, cross-platform format for representing
    internet addresses. It emphasizes explicitness and self-description.

    Learn more here: https://multiformats.io/multiaddr/

    Multiaddrs have both a binary and string representation.

        >>> from multiaddr import Multiaddr
        >>> addr = Multiaddr("/ip4/1.2.3.4/tcp/80")

    Multiaddr objects are immutable, so `encapsulate` and `decapsulate`
    return new objects rather than modify internal state.
    """

    __slots__ = ("_bytes", "registry")

    def __init__(self, addr, *, registry=protocols.REGISTRY):
        """Instantiate a new Multiaddr.

        Args:
            addr : A string-encoded or a byte-encoded Multiaddr

        """
        self.registry = registry
        if isinstance(addr, str):
            self._bytes = string_to_bytes(addr)
        elif isinstance(addr, bytes):
            self._bytes = addr
        elif isinstance(addr, Multiaddr):
            self._bytes = addr.to_bytes()
        else:
            raise TypeError("MultiAddr must be bytes, str or another MultiAddr instance")

    @classmethod
    def join(cls, *addrs):
        """Concatenate the values of the given MultiAddr strings or objects,
        encapsulating each successive MultiAddr value with the previous ones."""
        return cls(b"".join(map(lambda a: cls(a).to_bytes(), addrs)))

    def __eq__(self, other):
        """Checks if two Multiaddr objects are exactly equal."""
        return self._bytes == other._bytes

    def __str__(self):
        """Return the string representation of this Multiaddr.

        May raise a :class:`~multiaddr.exceptions.BinaryParseError` if the
        stored MultiAddr binary representation is invalid."""
        return bytes_to_string(self._bytes)

    def __contains__(self, proto):
        return proto in MultiAddrKeys(self)

    def __iter__(self):
        return iter(MultiAddrKeys(self))

    def __len__(self):
        return sum(1 for _ in bytes_iter(self.to_bytes()))

    def __repr__(self):
        return "<Multiaddr %s>" % str(self)

    def __hash__(self):
        return self._bytes.__hash__()

    def to_bytes(self):
        """Returns the byte array representation of this Multiaddr."""
        return self._bytes

    __bytes__ = to_bytes

    def protocols(self):
        """Returns a list of Protocols this Multiaddr includes."""
        return MultiAddrKeys(self)

    def split(self, maxsplit=-1):
        """Returns the list of individual path components this MultiAddr is made
        up of."""
        final_split_offset = -1
        results = []
        for idx, (offset, proto, codec, part_value) in enumerate(bytes_iter(self._bytes)):
            # Split at most `maxplit` times
            if idx == maxsplit:
                final_split_offset = offset
                break

            # Re-assemble binary MultiAddr representation
            part_size = varint.encode(len(part_value)) if codec.SIZE < 0 else b""
            part = b"".join((proto.vcode, part_size, part_value))

            # Add MultiAddr with the given value
            results.append(self.__class__(part))
        # Add final item with remainder of MultiAddr if there is anything left
        if final_split_offset >= 0:
            results.append(self.__class__(self._bytes[final_split_offset:]))

        return results

    keys = protocols

    def items(self):
        return MultiAddrItems(self)

    def values(self):
        return MultiAddrValues(self)

    def encapsulate(self, other):
        """Wrap this Multiaddr around another.

        For example:
            /ip4/1.2.3.4 encapsulate /tcp/80 = /ip4/1.2.3.4/tcp/80
        """
        return self.join(self, other)

    def decapsulate(self, other):
        """Remove a Multiaddr wrapping.

        For example:
            /ip4/1.2.3.4/tcp/80 decapsulate /ip4/1.2.3.4 = /tcp/80
        """
        s1 = self.to_bytes()
        s2 = Multiaddr(other).to_bytes()
        try:
            idx = s1.rindex(s2)
        except ValueError:
            # if multiaddr not contained, returns a copy
            return Multiaddr(self)
        return Multiaddr(s1[:idx])

    def value_for_protocol(self, proto):
        """Return the value (if any) following the specified protocol

        Returns
        -------
        Union[object, NoneType]
            The parsed protocol value for the given protocol code or ``None``
            if the given protocol does not require any value

        Raises
        ------
        ~multiaddr.exceptions.BinaryParseError
            The stored MultiAddr binary representation is invalid
        ~multiaddr.exceptions.ProtocolLookupError
            MultiAddr does not contain any instance of this protocol
        """
        proto = self.registry.find(proto)
        for proto2, value in self.items():
            if proto2 is proto or proto2 == proto:
                return value
        raise exceptions.ProtocolLookupError(proto, str(self))

    __getitem__ = value_for_protocol
