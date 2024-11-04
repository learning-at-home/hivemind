import base58
import cid

from . import LENGTH_PREFIXED_VAR_SIZE


SIZE = LENGTH_PREFIXED_VAR_SIZE
IS_PATH = False


# Spec: https://github.com/libp2p/specs/blob/master/peer-ids/peer-ids.md#string-representation
CIDv0_PREFIX_TO_LENGTH = {
    # base58btc prefixes for valid lengths 1 – 42 with the identity “hash” function
    '12': [5, 12, 19, 23, 30, 41, 52, 56],
    '13': [9, 16, 34, 45],
    '14': [27, 38, 49, 60],
    '15': [3, 6, 20],
    '16': [3, 6, 13, 20, 31, 42, 53],
    '17': [3, 13, 42],
    '18': [3],
    '19': [3, 24, 57],
    '1A': [24, 35, 46],
    '1B': [35],
    '1D': [17],
    '1E': [10, 17],
    '1F': [10],
    '1G': [10, 28, 50],
    '1H': [28, 39],
    '1P': [21],
    '1Q': [21],
    '1R': [21, 54],
    '1S': [54],
    '1T': [7, 32, 43],
    '1U': [7, 32, 43],
    '1V': [7],
    '1W': [7, 14],
    '1X': [7, 14],
    '1Y': [7, 14],
    '1Z': [7, 14],
    '1f': [4],
    '1g': [4, 58],
    '1h': [4, 25, 58],
    '1i': [4, 25],
    '1j': [4, 25],
    '1k': [4, 25, 47],
    '1m': [4, 36, 47],
    '1n': [4, 36],
    '1o': [4, 36],
    '1p': [4],
    '1q': [4],
    '1r': [4],
    '1s': [4],
    '1t': [4],
    '1u': [4],
    '1v': [4],
    '1w': [4],
    '1x': [4],
    '1y': [4],
    '1z': [4, 18],

    # base58btc prefix for length 42 with the sha256 hash function
    'Qm': [46],
}

PROTO_NAME_TO_CIDv1_CODEC = {
    # The “p2p” multiaddr protocol requires all keys to use the “libp2p-key” multicodec
    "p2p": "libp2p-key",
}


def to_bytes(proto, string):
    expected_codec = PROTO_NAME_TO_CIDv1_CODEC.get(proto.name)

    if len(string) in CIDv0_PREFIX_TO_LENGTH.get(string[0:2], ()):  # CIDv0
        # Upgrade the wire (binary) representation of any received CIDv0 string
        # to CIDv1 if we can determine which multicodec value to use
        if expected_codec:
            return cid.make_cid(1, expected_codec, base58.b58decode(string)).buffer

        return base58.b58decode(string)
    else:  # CIDv1+
        parsed = cid.from_string(string)

        # Ensure CID has correct codec for protocol
        if expected_codec and parsed.codec != expected_codec:
            raise ValueError("“{0}” multiaddr CIDs must use the “{1}” multicodec"
                             .format(proto.name, expected_codec))

        return parsed.buffer


def _is_binary_cidv0_multihash(buf):
    if buf.startswith(b"\x12\x20") and len(buf) == 34:  # SHA2-256
        return True

    if (buf[0] == 0x00 and buf[1] in range(43)) and len(buf) == (buf[1] + 2):  # Identity hash
        return True

    return False


def to_string(proto, buf):
    expected_codec = PROTO_NAME_TO_CIDv1_CODEC.get(proto.name)

    if _is_binary_cidv0_multihash(buf):  # CIDv0
        if not expected_codec:
            # Simply encode as base58btc as there is nothing better to do
            return base58.b58encode(buf).decode('ascii')

        # “Implementations SHOULD display peer IDs using the first (raw
        #  base58btc encoded multihash) format until the second format is
        #  widely supported.”
        #
        # In the future the following line should instead convert the multihash
        # to CIDv1 and with the `expected_codec` and wrap it in base32:
        #   return cid.make_cid(1, expected_codec, buf).encode("base32").decode("ascii")
        return base58.b58encode(buf).decode("ascii")
    else:  # CIDv1+
        parsed = cid.from_bytes(buf)

        # Ensure CID has correct codec for protocol
        if expected_codec and parsed.codec != expected_codec:
            raise ValueError("“{0}” multiaddr CIDs must use the “{1}” multicodec"
                             .format(proto.name, expected_codec))

        # “Implementations SHOULD display peer IDs using the first (raw
        #  base58btc encoded multihash) format until the second format is
        #  widely supported.”
        if expected_codec and _is_binary_cidv0_multihash(parsed.multihash):
            return base58.b58encode(parsed.multihash).decode("ascii")

        return parsed.encode("base32").decode("ascii")
