from contextlib import closing
import socket
import asyncio

from google.protobuf.message import Message as PBMessage

from pb import p2pd_pb2 as p2pd_pb


DEFAULT_MAX_BITS: int = 64


class ControlFailure(Exception):
    pass


class DispatchFailure(Exception):
    pass


async def write_unsigned_varint(
    stream: asyncio.StreamWriter, integer: int, max_bits: int = DEFAULT_MAX_BITS
) -> None:
    max_int: int = 1 << max_bits
    if integer < 0:
        raise ValueError(f"negative integer: {integer}")
    if integer >= max_int:
        raise ValueError(f"integer too large: {integer}")
    while True:
        value: int = integer & 0x7F
        integer >>= 7
        if integer != 0:
            value |= 0x80
        byte = value.to_bytes(1, "big")
        stream.write(byte)
        if integer == 0:
            break


async def read_unsigned_varint(
    stream: asyncio.StreamReader, max_bits: int = DEFAULT_MAX_BITS
) -> int:
    max_int: int = 1 << max_bits
    iteration: int = 0
    result: int = 0
    has_next: bool = True
    while has_next:
        data = await stream.readexactly(1)
        c = data[0]
        value = c & 0x7F
        result |= value << (iteration * 7)
        has_next = (c & 0x80) != 0
        iteration += 1
        if result >= max_int:
            raise ValueError(f"varint overflowed: {result}")
    return result


def raise_if_failed(response: p2pd_pb.Response) -> None:
    if response.type == p2pd_pb.Response.ERROR:
        raise ControlFailure(f"connect failed. msg={response.error.msg}")


async def write_pbmsg(stream: asyncio.StreamWriter, pbmsg: PBMessage) -> None:
    size = pbmsg.ByteSize()
    await write_unsigned_varint(stream, size)
    msg_bytes: bytes = pbmsg.SerializeToString()
    stream.write(msg_bytes)


async def read_pbmsg_safe(stream: asyncio.StreamReader, pbmsg: PBMessage) -> None:
    len_msg_bytes = await read_unsigned_varint(stream)
    msg_bytes = await stream.readexactly(len_msg_bytes)
    pbmsg.ParseFromString(msg_bytes)


def get_unused_tcp_port() -> int:
    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
        s.bind(("localhost", 0))
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return s.getsockname()[1]
