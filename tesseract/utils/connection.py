from contextlib import AbstractContextManager
from socket import socket
from typing import Tuple


class Connection(AbstractContextManager):
    header_size = 4  # number of characters in all headers
    payload_length_size = 8  # number of bytes used to encode payload length

    __slots__ = ('conn', 'addr')

    def __init__(self, conn: socket, addr: Tuple[str, int]):
        self.conn, self.addr = conn, addr

    @staticmethod
    def create(host: str, port: int):
        sock = socket()
        addr = (host, port)
        sock.connect(addr)
        return Connection(sock, addr)

    def send_raw(self, header: str, content: bytes):
        self.conn.send(header.encode())
        self.conn.send(len(content).to_bytes(self.payload_length_size, byteorder='big'))

        total_sent = 0
        while total_sent < len(content):
            sent = self.conn.send(content[total_sent:])
            if sent == 0:
                raise RuntimeError("socket connection broken")
            total_sent = total_sent + sent

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
