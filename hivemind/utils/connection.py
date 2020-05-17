import socket
from contextlib import AbstractContextManager, closing
from typing import Tuple
import asyncio

Hostname, Port = str, int  # flavour types
Endpoint = Tuple[Hostname, Port]  # https://networkengineering.stackexchange.com/a/9435
LOCALHOST = '127.0.0.1'


class Connection(AbstractContextManager):
    header_size = 4  # number of characters in all headers
    payload_length_size = 8  # number of bytes used to encode payload length

    __slots__ = ('conn', 'addr')

    def __init__(self, conn: socket, addr: Endpoint):
        self.conn, self.addr = conn, addr

    @staticmethod
    def create(host: str, port: int):
        sock = socket.socket()
        addr = (host, port)
        sock.connect(addr)
        return Connection(sock, addr)

    def send_raw(self, header: str, content: bytes):
        self.conn.send(header.encode())
        self.conn.send(len(content).to_bytes(self.payload_length_size, byteorder='big'))

        self.conn.sendall(content)

    def recv_header(self) -> str:
        return self.conn.recv(self.header_size).decode()

    def recv_raw(self, max_package: int = 2048) -> bytes:
        length = int.from_bytes(self.conn.recv(self.payload_length_size), byteorder='big')
        chunks = []
        bytes_recd = 0
        while bytes_recd < length:
            chunk = self.conn.recv(min(length - bytes_recd, max_package))
            if chunk == b'':
                raise RuntimeError("socket connection broken")
            chunks.append(chunk)
            bytes_recd = bytes_recd + len(chunk)
        ret = b''.join(chunks)
        assert len(ret) == length
        return ret

    def recv_message(self) -> Tuple[str, bytes]:
        return self.recv_header(), self.recv_raw()

    def __exit__(self, *exc_info):
        self.conn.close()


class AsyncConnection(Connection):
    async def recv_message(self):
        header = await self.recv_header()
        message = await self.recv_raw()
        return header, message

    async def recv_header(self):
        loop = asyncio.get_running_loop()
        data = await loop.sock_recv(self.conn, self.header_size)
        return data.decode()

    async def recv_raw(self, max_package: int = 2048):
        loop = asyncio.get_running_loop()
        raw_length = await loop.sock_recv(self.conn, self.payload_length_size)
        length = int.from_bytes(raw_length, byteorder='big')
        chunks = []
        bytes_recd = 0
        while bytes_recd < length:
            chunk = await loop.sock_recv(self.conn, min(length - bytes_recd, max_package))
            if chunk == b'':
                raise RuntimeError("socket connection broken")
            chunks.append(chunk)
            bytes_recd = bytes_recd + len(chunk)
        ret = b''.join(chunks)
        assert len(ret) == length
        return ret

    async def send_raw(self, header: str, content: bytes):
        loop = asyncio.get_running_loop()
        await loop.sock_sendall(self.conn, header.encode())
        await loop.sock_sendall(self.conn, len(content).to_bytes(self.payload_length_size, byteorder='big'))
        await loop.sock_sendall(self.conn, content)


def find_open_port(params=(socket.AF_INET, socket.SOCK_STREAM), opt=(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)):
    """ Finds a tcp port that can be occupied with a socket with *params and use *opt options """
    try:
        with closing(socket.socket(*params)) as sock:
            sock.bind(('', 0))
            sock.setsockopt(*opt)
            return sock.getsockname()[1]
    except Exception as e:
        raise e
