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
import struct

SIZE = 16
IS_PATH = False


def to_bytes(proto, string):
    try:
        return struct.pack(">H", int(string, 10))
    except ValueError as exc:
        raise ValueError("Not a base 10 integer") from exc
    except struct.error as exc:
        raise ValueError("Integer not in range(65536)") from exc


def to_string(proto, buf):
    if len(buf) != 2:
        raise ValueError("Invalid integer length (must be 2 bytes / 16 bits)")
    return str(struct.unpack(">H", buf)[0])
