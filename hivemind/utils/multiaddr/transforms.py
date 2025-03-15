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
import io

import varint

from . import exceptions
from .codecs import LENGTH_PREFIXED_VAR_SIZE, codec_by_name
from .protocols import protocol_with_code, protocol_with_name


def string_to_bytes(string):
    bs = []
    for proto, codec, value in string_iter(string):
        bs.append(varint.encode(proto.code))
        if value is not None:
            try:
                buf = codec.to_bytes(proto, value)
            except Exception as exc:
                raise exceptions.StringParseError(str(exc), string, proto.name, exc) from exc
            if codec.SIZE == LENGTH_PREFIXED_VAR_SIZE:
                bs.append(varint.encode(len(buf)))
            bs.append(buf)
    return b"".join(bs)


def bytes_to_string(buf):
    st = [""]  # start with empty string so we get a leading slash on join()
    for _, proto, codec, part in bytes_iter(buf):
        st.append(proto.name)
        if codec.SIZE != 0:
            try:
                value = codec.to_string(proto, part)
            except Exception as exc:
                raise exceptions.BinaryParseError(str(exc), buf, proto.name, exc) from exc
            if codec.IS_PATH and value[0] == "/":
                st.append(value[1:])
            else:
                st.append(value)
    return "/".join(st)


def size_for_addr(codec, buf_io):
    if codec.SIZE >= 0:
        return codec.SIZE // 8
    else:
        return varint.decode_stream(buf_io)


def string_iter(string):
    if not string.startswith("/"):
        raise exceptions.StringParseError("Must begin with /", string)
    # consume trailing slashes
    string = string.rstrip("/")
    sp = string.split("/")

    # skip the first element, since it starts with /
    sp.pop(0)
    while sp:
        element = sp.pop(0)
        try:
            proto = protocol_with_name(element)
            codec = codec_by_name(proto.codec)
        except (ImportError, exceptions.ProtocolNotFoundError) as exc:
            raise exceptions.StringParseError("Unknown Protocol", string, element) from exc
        value = None
        if codec.SIZE != 0:
            if len(sp) < 1:
                raise exceptions.StringParseError("Protocol requires address", string, proto.name)
            if codec.IS_PATH:
                value = "/" + "/".join(sp)
                sp.clear()
            else:
                value = sp.pop(0)
        yield proto, codec, value


def bytes_iter(buf):
    buf_io = io.BytesIO(buf)
    while buf_io.tell() < len(buf):
        offset = buf_io.tell()
        code = varint.decode_stream(buf_io)
        proto = None
        try:
            proto = protocol_with_code(code)
            codec = codec_by_name(proto.codec)
        except (ImportError, exceptions.ProtocolNotFoundError) as exc:
            raise exceptions.BinaryParseError(
                "Unknown Protocol",
                buf,
                proto.name if proto else code,
            ) from exc

        size = size_for_addr(codec, buf_io)
        yield offset, proto, codec, buf_io.read(size)
