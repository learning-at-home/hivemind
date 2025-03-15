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
import varint

from hivemind.utils.multiaddr import exceptions, protocols


def test_code_to_varint():
    vi = varint.encode(5)
    assert vi == b"\x05"
    vi = varint.encode(150)
    assert vi == b"\x96\x01"


def test_varint_to_code():
    cc = varint.decode_bytes(b"\x05")
    assert cc == 5
    cc = varint.decode_bytes(b"\x96\x01")
    assert cc == 150


@pytest.fixture
def valid_params():
    return {"code": protocols.P_IP4, "name": "ipb4", "codec": "ipb"}


def test_valid(valid_params):
    proto = protocols.Protocol(**valid_params)
    for key in valid_params:
        assert getattr(proto, key) == valid_params[key]


@pytest.mark.parametrize("invalid_code", ["abc"])
def test_invalid_code(valid_params, invalid_code):
    valid_params["code"] = invalid_code
    with pytest.raises(TypeError):
        protocols.Protocol(**valid_params)


@pytest.mark.parametrize("invalid_name", [123, 1.0])
def test_invalid_name(valid_params, invalid_name):
    valid_params["name"] = invalid_name
    with pytest.raises(TypeError):
        protocols.Protocol(**valid_params)


@pytest.mark.parametrize("invalid_codec", [b"ip4", 123, 0.123])
def test_invalid_codec(valid_params, invalid_codec):
    valid_params["codec"] = invalid_codec
    with pytest.raises(TypeError):
        protocols.Protocol(**valid_params)


@pytest.mark.parametrize("name", ["foo-str", "foo-u"])
def test_valid_names(valid_params, name):
    valid_params["name"] = name
    test_valid(valid_params)


@pytest.mark.parametrize("codec", ["ip4", "ip6"])
def test_valid_codecs(valid_params, codec):
    valid_params["codec"] = codec
    test_valid(valid_params)


def test_protocol_with_name():
    proto = protocols.protocol_with_name("ip4")
    assert proto.name == "ip4"
    assert proto.code == protocols.P_IP4
    assert proto.size == 32
    assert proto.vcode == varint.encode(protocols.P_IP4)
    assert hash(proto) == protocols.P_IP4
    assert protocols.protocol_with_any("ip4") == proto
    assert protocols.protocol_with_any(proto) == proto

    with pytest.raises(exceptions.ProtocolNotFoundError):
        proto = protocols.protocol_with_name("foo")


def test_protocol_with_code():
    proto = protocols.protocol_with_code(protocols.P_IP4)
    assert proto.name == "ip4"
    assert proto.code == protocols.P_IP4
    assert proto.size == 32
    assert proto.vcode == varint.encode(protocols.P_IP4)
    assert hash(proto) == protocols.P_IP4
    assert protocols.protocol_with_any(protocols.P_IP4) == proto
    assert protocols.protocol_with_any(proto) == proto

    with pytest.raises(exceptions.ProtocolNotFoundError):
        proto = protocols.protocol_with_code(1234)


def test_protocol_equality():
    proto1 = protocols.protocol_with_name("ip4")
    proto2 = protocols.protocol_with_code(protocols.P_IP4)
    proto3 = protocols.protocol_with_name("onion")
    proto4 = protocols.protocol_with_name("onion3")

    assert proto1 == proto2
    assert proto1 != proto3
    assert proto3 != proto4
    assert proto1 is not None
    assert proto2 != str(proto2)


@pytest.mark.parametrize("names", [["ip4"], ["ip4", "tcp"], ["ip4", "tcp", "udp"]])
def test_protocols_with_string(names):
    expected = [protocols.protocol_with_name(name) for name in names]
    ins = "/".join(names)
    assert protocols.protocols_with_string(ins) == expected
    assert protocols.protocols_with_string("/" + ins) == expected
    assert protocols.protocols_with_string("/" + ins + "/") == expected


@pytest.mark.parametrize("invalid_name", ["", "/", "//"])
def test_protocols_with_string_invalid(invalid_name):
    assert protocols.protocols_with_string(invalid_name) == []


def test_protocols_with_string_mixed():
    names = ["ip4"]
    ins = "/".join(names)
    test_protocols_with_string(names)
    with pytest.raises(exceptions.ProtocolNotFoundError):
        names.append("foo")
        ins = "/".join(names)
        protocols.protocols_with_string(ins)


def test_add_protocol(valid_params):
    registry = protocols.ProtocolRegistry()
    proto = protocols.Protocol(**valid_params)
    registry.add(proto)
    assert proto.name in registry._names_to_protocols
    assert proto.code in registry._codes_to_protocols
    assert registry.find(proto.name) is registry.find(proto.code) is proto


def test_add_protocol_twice(valid_params):
    registry = protocols.ProtocolRegistry()
    proto = registry.add(protocols.Protocol(**valid_params))

    with pytest.raises(exceptions.ProtocolExistsError):
        registry.add(proto)
    del registry._names_to_protocols[proto.name]
    with pytest.raises(exceptions.ProtocolExistsError):
        registry.add(proto)
    del registry._codes_to_protocols[proto.code]
    registry.add(proto)


def test_add_protocol_alias():
    registry = protocols.REGISTRY.copy(unlock=True)
    registry.add_alias_name("tcp", "abcd")
    registry.add_alias_code("tcp", 123456)

    with pytest.raises(exceptions.ProtocolExistsError):
        registry.add_alias_name("tcp", "abcd")
    with pytest.raises(exceptions.ProtocolExistsError):
        registry.add_alias_code("tcp", 123456)

    assert registry.find("tcp") is registry.find("abcd")
    assert registry.find("tcp") is registry.find(123456)


def test_add_protocol_lock(valid_params):
    registry = protocols.REGISTRY.copy(unlock=True)
    assert not registry.locked
    registry.lock()
    assert registry.locked

    with pytest.raises(exceptions.ProtocolRegistryLocked):
        registry.add(protocols.Protocol(**valid_params))
    with pytest.raises(exceptions.ProtocolRegistryLocked):
        registry.add_alias_name("tcp", "abcdef")
    with pytest.raises(exceptions.ProtocolRegistryLocked):
        registry.add_alias_code(0x4, 0x123456)


def test_protocol_repr():
    proto = protocols.protocol_with_name("ip4")
    assert "Protocol(code=4, name='ip4', codec='ip4')" == repr(proto)
