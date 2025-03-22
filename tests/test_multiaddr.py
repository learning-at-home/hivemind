# This code is originally taken from https://github.com/multiformats/py-multiaddr
#
# The MIT License (MIT)
#
# Copyright (c) 2014-2015 Steven Buss
# Copyright (c) 2019-2020 Alexander Schlarb
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
import pytest

from hivemind.utils.multiaddr.exceptions import (
    BinaryParseError,
    ProtocolLookupError,
    ProtocolNotFoundError,
    StringParseError,
)
from hivemind.utils.multiaddr.multiaddr import Multiaddr
from hivemind.utils.multiaddr.protocols import (
    P_DNS,
    P_IP4,
    P_IP6,
    P_P2P,
    P_TCP,
    P_UDP,
    P_UNIX,
    P_UTP,
    protocol_with_name,
    protocols_with_string,
)


@pytest.mark.parametrize(
    "addr_str",
    [
        "/ip4",
        "/ip4/::1",
        "/ip4/fdpsofodsajfdoisa",
        "/ip6",
        "/ip6zone",
        "/ip6zone/",
        "/ip6zone//ip6/fe80::1",
        "/udp",
        "/tcp",
        "/sctp",
        "/udp/65536",
        "/tcp/65536",
        "/onion/9imaq4ygg2iegci7:80",
        "/onion/aaimaq4ygg2iegci7:80",
        "/onion/timaq4ygg2iegci7:0",
        "/onion/timaq4ygg2iegci7:-1",
        "/onion/timaq4ygg2iegci7",
        "/onion/timaq4ygg2iegci@:666",
        "/onion3/9ww6ybal4bd7szmgncyruucpgfkqahzddi37ktceo3ah7ngmcopnpyyd:80",
        "/onion3/vww6ybal4bd7szmgncyruucpgfkqahzddi37ktceo3ah7ngmcopnpyyd7:80",
        "/onion3/vww6ybal4bd7szmgncyruucpgfkqahzddi37ktceo3ah7ngmcopnpyyd:0",
        "/onion3/vww6ybal4bd7szmgncyruucpgfkqahzddi37ktceo3ah7ngmcopnpyyd:a",
        "/onion3/vww6ybal4bd7szmgncyruucpgfkqahzddi37ktceo3ah7ngmcopnpyyd:-1",
        "/onion3/vww6ybal4bd7szmgncyruucpgfkqahzddi37ktceo3ah7ngmcopnpyyd",
        "/onion3/vww6ybal4bd7szmgncyruucpgfkqahzddi37ktceo3ah7ngmcopnpyy@:666",
        "/udp/1234/sctp",
        "/udp/1234/udt/1234",
        "/udp/1234/utp/1234",
        "/ip4/127.0.0.1/udp/jfodsajfidosajfoidsa",
        "/ip4/127.0.0.1/udp",
        "/ip4/127.0.0.1/tcp/jfodsajfidosajfoidsa",
        "/ip4/127.0.0.1/tcp",
        "/ip4/127.0.0.1/p2p",
        "/ip4/127.0.0.1/p2p/tcp",
        "/unix",
        "/ip4/1.2.3.4/tcp/80/unix",
        "/ip4/127.0.0.1/tcp/9090/http/p2p-webcrt-direct",
        "/dns",
        "/dns4",
        "/dns6",
        "/cancer",
    ],
)
def test_invalid(addr_str):
    with pytest.raises(StringParseError):
        Multiaddr(addr_str)


@pytest.mark.parametrize(
    "addr_str",
    [
        "/ip4/1.2.3.4",
        "/ip4/0.0.0.0",
        "/ip6/::1",
        "/ip6/2601:9:4f81:9700:803e:ca65:66e8:c21",
        "/ip6zone/x/ip6/fe80::1",
        "/ip6zone/x%y/ip6/fe80::1",
        "/ip6zone/x%y/ip6/::",
        "/onion/timaq4ygg2iegci7:1234",
        "/onion/timaq4ygg2iegci7:80/http",
        "/onion3/vww6ybal4bd7szmgncyruucpgfkqahzddi37ktceo3ah7ngmcopnpyyd:1234",
        "/onion3/vww6ybal4bd7szmgncyruucpgfkqahzddi37ktceo3ah7ngmcopnpyyd:80/http",
        "/udp/0",
        "/tcp/0",
        "/sctp/0",
        "/udp/1234",
        "/tcp/1234",
        "/sctp/1234",
        "/utp",
        "/udp/65535",
        "/tcp/65535",
        "/p2p/QmcgpsyWgH8Y8ajJz1Cu72KnS5uo2Aa2LpzU7kinSupNKC",
        "/udp/1234/sctp/1234",
        "/udp/1234/udt",
        "/udp/1234/utp",
        "/tcp/1234/http",
        "/tcp/1234/https",
        "/p2p/QmcgpsyWgH8Y8ajJz1Cu72KnS5uo2Aa2LpzU7kinSupNKC/tcp/1234",
        "/ip4/127.0.0.1/udp/1234",
        "/ip4/127.0.0.1/p2p/QmcgpsyWgH8Y8ajJz1Cu72KnS5uo2Aa2LpzU7kinSupNKC/tcp/1234",
        "/unix/a/b/c/d/e",
        "/unix/Überrschung!/大柱",
        "/unix/stdio",
        "/ip4/1.2.3.4/tcp/80/unix/a/b/c/d/e/f",
        "/ip4/127.0.0.1/p2p/QmcgpsyWgH8Y8ajJz1Cu72KnS5uo2Aa2LpzU7kinSupNKC/tcp/1234/unix/stdio",
        "/ip4/127.0.0.1/tcp/9090/http/p2p-webrtc-direct",
        "/dns/example.com",
        "/dns4/موقع.وزارة-الاتصالات.مصر",
    ],
)  # nopep8
def test_valid(addr_str):
    ma = Multiaddr(addr_str)
    assert str(ma) == addr_str.rstrip("/")


def test_eq():
    m1 = Multiaddr("/ip4/127.0.0.1/udp/1234")
    m2 = Multiaddr("/ip4/127.0.0.1/tcp/1234")
    m3 = Multiaddr("/ip4/127.0.0.1/tcp/1234")
    m4 = Multiaddr("/ip4/127.0.0.1/tcp/1234/")

    assert m1 != m2
    assert m2 != m1

    assert m2 == m3
    assert m3 == m2

    assert m1 == m1

    assert m2 == m4
    assert m4 == m2
    assert m3 == m4
    assert m4 == m3


def test_protocols():
    ma = Multiaddr("/ip4/127.0.0.1/udp/1234")
    protos = ma.protocols()
    assert protos[0].code == protocol_with_name("ip4").code
    assert protos[1].code == protocol_with_name("udp").code


@pytest.mark.parametrize(
    "proto_string,expected",
    [
        ("/ip4", [protocol_with_name("ip4")]),
        ("/ip4/tcp", [protocol_with_name("ip4"), protocol_with_name("tcp")]),
        (
            "ip4/tcp/udp/ip6",
            [
                protocol_with_name("ip4"),
                protocol_with_name("tcp"),
                protocol_with_name("udp"),
                protocol_with_name("ip6"),
            ],
        ),
        ("////////ip4/tcp", [protocol_with_name("ip4"), protocol_with_name("tcp")]),
        ("ip4/udp/////////", [protocol_with_name("ip4"), protocol_with_name("udp")]),
        ("////////ip4/tcp////////", [protocol_with_name("ip4"), protocol_with_name("tcp")]),
        ("////////ip4/////////tcp////////", [protocol_with_name("ip4"), protocol_with_name("tcp")]),
    ],
)
def test_protocols_with_string(proto_string, expected):
    protos = protocols_with_string(proto_string)
    assert protos == expected


@pytest.mark.parametrize("proto_string", ["dsijafd", "/ip4/tcp/fidosafoidsa", "////////ip4/tcp/21432141/////////"])
def test_invalid_protocols_with_string(proto_string):
    with pytest.raises(ProtocolNotFoundError):
        protocols_with_string(proto_string)


@pytest.mark.parametrize(
    "proto_string,maxsplit,expected",
    [
        ("/ip4/1.2.3.4", -1, ("/ip4/1.2.3.4",)),
        ("/ip4/0.0.0.0", 0, ("/ip4/0.0.0.0",)),
        ("/ip6/::1", 1, ("/ip6/::1",)),
        ("/onion/timaq4ygg2iegci7:80/http", 0, ("/onion/timaq4ygg2iegci7:80/http",)),
        (
            "/ip4/127.0.0.1/p2p/bafzbeigvf25ytwc3akrijfecaotc74udrhcxzh2cx3we5qqnw5vgrei4bm/tcp/1234",
            1,
            ("/ip4/127.0.0.1", "/p2p/QmcgpsyWgH8Y8ajJz1Cu72KnS5uo2Aa2LpzU7kinSupNKC/tcp/1234"),
        ),
        ("/ip4/1.2.3.4/tcp/80/unix/a/b/c/d/e/f", -1, ("/ip4/1.2.3.4", "/tcp/80", "/unix/a/b/c/d/e/f")),
    ],
)
def test_split(proto_string, maxsplit, expected):
    assert tuple(map(str, Multiaddr(proto_string).split(maxsplit))) == expected


@pytest.mark.parametrize(
    "proto_parts,expected",
    [
        (("/ip4/1.2.3.4",), "/ip4/1.2.3.4"),
        ((b"\x04\x00\x00\x00\x00",), "/ip4/0.0.0.0"),
        (("/ip6/::1",), "/ip6/::1"),
        (("/onion/timaq4ygg2iegci7:80/http",), "/onion/timaq4ygg2iegci7:80/http"),
        (
            (
                b"\x04\x7f\x00\x00\x01",
                "/p2p/bafzbeigvf25ytwc3akrijfecaotc74udrhcxzh2cx3we5qqnw5vgrei4bm/tcp/1234",
            ),
            "/ip4/127.0.0.1/p2p/QmcgpsyWgH8Y8ajJz1Cu72KnS5uo2Aa2LpzU7kinSupNKC/tcp/1234",
        ),
        (("/ip4/1.2.3.4", "/tcp/80", "/unix/a/b/c/d/e/f"), "/ip4/1.2.3.4/tcp/80/unix/a/b/c/d/e/f"),
    ],
)
def test_join(proto_parts, expected):
    assert str(Multiaddr.join(*proto_parts)) == expected


def test_encapsulate():
    m1 = Multiaddr("/ip4/127.0.0.1/udp/1234")
    m2 = Multiaddr("/udp/5678")

    encapsulated = m1.encapsulate(m2)
    assert str(encapsulated) == "/ip4/127.0.0.1/udp/1234/udp/5678"

    m3 = Multiaddr("/udp/5678")
    decapsulated = encapsulated.decapsulate(m3)
    assert str(decapsulated) == "/ip4/127.0.0.1/udp/1234"

    m4 = Multiaddr("/ip4/127.0.0.1")
    decapsulated_2 = decapsulated.decapsulate(m4)
    assert str(decapsulated_2) == ""

    m5 = Multiaddr("/ip6/::1")
    decapsulated_3 = decapsulated.decapsulate(m5)

    assert str(decapsulated_3) == "/ip4/127.0.0.1/udp/1234"


def assert_value_for_proto(multi, proto, expected):
    assert multi.value_for_protocol(proto) == expected


def test_get_value():
    ma = Multiaddr(
        "/ip4/127.0.0.1/utp/tcp/5555/udp/1234/utp/p2p/bafzbeigalb34xlqdtvyklzqa5ibmn6pssqsdskc4ty2e4jxy2kamquh22y"
    )

    assert_value_for_proto(ma, P_IP4, "127.0.0.1")
    assert_value_for_proto(ma, P_UTP, None)
    assert_value_for_proto(ma, P_TCP, "5555")
    assert_value_for_proto(ma, P_UDP, "1234")
    assert_value_for_proto(ma, P_P2P, "QmbHVEEepCi7rn7VL7Exxpd2Ci9NNB6ifvqwhsrbRMgQFP")
    assert_value_for_proto(ma, "ip4", "127.0.0.1")
    assert_value_for_proto(ma, "utp", None)
    assert_value_for_proto(ma, "tcp", "5555")
    assert_value_for_proto(ma, "udp", "1234")
    assert_value_for_proto(ma, protocol_with_name("ip4"), "127.0.0.1")
    assert_value_for_proto(ma, protocol_with_name("utp"), None)
    assert_value_for_proto(ma, protocol_with_name("tcp"), "5555")
    assert_value_for_proto(ma, protocol_with_name("udp"), "1234")

    with pytest.raises(ProtocolLookupError):
        ma.value_for_protocol(P_IP6)
    with pytest.raises(ProtocolLookupError):
        ma.value_for_protocol("ip6")
    with pytest.raises(ProtocolLookupError):
        ma.value_for_protocol(protocol_with_name("ip6"))

    a = Multiaddr(b"\x35\x03a:b")  # invalid protocol value
    with pytest.raises(BinaryParseError):
        a.value_for_protocol(P_DNS)

    a = Multiaddr("/ip4/0.0.0.0")  # only one addr
    assert_value_for_proto(a, P_IP4, "0.0.0.0")

    a = Multiaddr("/ip4/0.0.0.0/ip4/0.0.0.0/ip4/0.0.0.0")  # same sub-addr
    assert_value_for_proto(a, P_IP4, "0.0.0.0")

    a = Multiaddr("/ip4/0.0.0.0/udp/12345/utp")  # ending in a no-value one.
    assert_value_for_proto(a, P_IP4, "0.0.0.0")
    assert_value_for_proto(a, P_UDP, "12345")
    assert_value_for_proto(a, P_UTP, None)

    a = Multiaddr("/ip4/0.0.0.0/unix/a/b/c/d")  # ending in a path one.
    assert_value_for_proto(a, P_IP4, "0.0.0.0")
    assert_value_for_proto(a, P_UNIX, "/a/b/c/d")

    a = Multiaddr("/unix/studio")
    assert_value_for_proto(a, P_UNIX, "/studio")  # only a path.


def test_views():
    ma = Multiaddr(
        "/ip4/127.0.0.1/utp/tcp/5555/udp/1234/utp/p2p/bafzbeigalb34xlqdtvyklzqa5ibmn6pssqsdskc4ty2e4jxy2kamquh22y"
    )

    for idx, (proto1, proto2, item, value) in enumerate(zip(ma, ma.keys(), ma.items(), ma.values())):  # noqa: E501
        assert (proto1, value) == (proto2, value) == item
        assert proto1 in ma
        assert proto2 in ma.keys()
        assert item in ma.items()
        assert value in ma.values()
        assert ma.keys()[idx] == ma.keys()[idx - len(ma)] == proto1 == proto2
        assert ma.items()[idx] == ma.items()[idx - len(ma)] == item
        assert ma.values()[idx] == ma.values()[idx - len(ma)] == ma[proto1] == value

    assert len(ma.keys()) == len(ma.items()) == len(ma.values()) == len(ma)
    assert len(list(ma.keys())) == len(ma.keys())
    assert len(list(ma.items())) == len(ma.items())
    assert len(list(ma.values())) == len(ma.values())

    with pytest.raises(IndexError):
        ma.keys()[len(ma)]
    with pytest.raises(IndexError):
        ma.items()[len(ma)]
    with pytest.raises(IndexError):
        ma.values()[len(ma)]


def test_bad_initialization_no_params():
    with pytest.raises(TypeError):
        Multiaddr()


def test_bad_initialization_too_many_params():
    with pytest.raises(TypeError):
        Multiaddr("/ip4/0.0.0.0", "")


def test_bad_initialization_wrong_type():
    with pytest.raises(TypeError):
        Multiaddr(42)


def test_value_for_protocol_argument_wrong_type():
    a = Multiaddr("/ip4/127.0.0.1/udp/1234")
    with pytest.raises(ProtocolNotFoundError):
        a.value_for_protocol("str123")

    with pytest.raises(TypeError):
        a.value_for_protocol(None)


def test_multi_addr_str_corruption():
    a = Multiaddr("/ip4/127.0.0.1/udp/1234")
    a._bytes = b"047047047"

    with pytest.raises(BinaryParseError):
        str(a)


def test_decapsulate():
    a = Multiaddr("/ip4/127.0.0.1/udp/1234")
    u = Multiaddr("/udp/1234")
    assert a.decapsulate(u) == Multiaddr("/ip4/127.0.0.1")


def test__repr():
    a = Multiaddr("/ip4/127.0.0.1/udp/1234")
    assert repr(a) == "<Multiaddr %s>" % str(a)


def test_zone():
    ip6_string = "/ip6zone/eth0/ip6/::1"
    ip6_bytes = b"\x2a\x04eth0\x29\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x01"

    maddr_from_str = Multiaddr(ip6_string)
    assert maddr_from_str.to_bytes() == ip6_bytes

    maddr_from_bytes = Multiaddr(ip6_bytes)
    assert str(maddr_from_bytes) == ip6_string


def test_hash():
    addr_bytes = Multiaddr("/ip4/127.0.0.1/udp/1234").to_bytes()

    assert hash(Multiaddr(addr_bytes)) == hash(addr_bytes)
