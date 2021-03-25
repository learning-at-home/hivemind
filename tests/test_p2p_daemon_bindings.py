import asyncio
import functools
import io
import os
import subprocess
import time
import uuid
from contextlib import asynccontextmanager, AsyncExitStack
from typing import NamedTuple

from google.protobuf.message import EncodeError
from multiaddr import Multiaddr, protocols

import pytest

from hivemind.p2p.p2p_daemon_bindings import config
from hivemind.p2p.p2p_daemon_bindings.control import parse_conn_protocol, DaemonConnector, ControlClient
from hivemind.p2p.p2p_daemon_bindings.p2pclient import Client
from hivemind.p2p.p2p_daemon_bindings.utils import ControlFailure, raise_if_failed, write_unsigned_varint, \
    read_unsigned_varint, read_pbmsg_safe, write_pbmsg, get_unused_tcp_port
from hivemind.proto import p2pd_pb2 as p2pd_pb
from hivemind.p2p.p2p_daemon_bindings.datastructures import ID, StreamInfo, PeerInfo


def test_raise_if_failed_raises():
    resp = p2pd_pb.Response()
    resp.type = p2pd_pb.Response.ERROR
    with pytest.raises(ControlFailure):
        raise_if_failed(resp)


def test_raise_if_failed_not_raises():
    resp = p2pd_pb.Response()
    resp.type = p2pd_pb.Response.OK
    raise_if_failed(resp)


pairs_int_varint_valid = (
    (0, b"\x00"),
    (1, b"\x01"),
    (128, b"\x80\x01"),
    (2 ** 32, b"\x80\x80\x80\x80\x10"),
    (2 ** 64 - 1, b"\xff\xff\xff\xff\xff\xff\xff\xff\xff\x01"),
)

pairs_int_varint_overflow = (
    (2 ** 64, b"\x80\x80\x80\x80\x80\x80\x80\x80\x80\x02"),
    (2 ** 64 + 1, b"\x81\x80\x80\x80\x80\x80\x80\x80\x80\x02"),
    (
        2 ** 128,
        b"\x80\x80\x80\x80\x80\x80\x80\x80\x80\x80\x80\x80\x80\x80\x80\x80\x80\x80\x04",
    ),
)


class MockReader(io.BytesIO):
    async def readexactly(self, n):
        await asyncio.sleep(0)
        return self.read(n)


class MockWriter(io.BytesIO):
    pass


class MockReaderWriter(MockReader, MockWriter):
    pass


@pytest.mark.parametrize("integer, var_integer", pairs_int_varint_valid)
@pytest.mark.asyncio
async def test_write_unsigned_varint(integer, var_integer):
    s = MockWriter()
    await write_unsigned_varint(s, integer)
    assert s.getvalue() == var_integer


@pytest.mark.parametrize("integer", tuple(i[0] for i in pairs_int_varint_overflow))
@pytest.mark.asyncio
async def test_write_unsigned_varint_overflow(integer):
    s = MockWriter()
    with pytest.raises(ValueError):
        await write_unsigned_varint(s, integer)


@pytest.mark.parametrize("integer", (-1, -(2 ** 32), -(2 ** 64), -(2 ** 128)))
@pytest.mark.asyncio
async def test_write_unsigned_varint_negative(integer):
    s = MockWriter()
    with pytest.raises(ValueError):
        await write_unsigned_varint(s, integer)


@pytest.mark.parametrize("integer, var_integer", pairs_int_varint_valid)
@pytest.mark.asyncio
async def test_read_unsigned_varint(integer, var_integer):
    s = MockReader(var_integer)
    result = await read_unsigned_varint(s)
    assert result == integer


@pytest.mark.parametrize("var_integer", tuple(i[1] for i in pairs_int_varint_overflow))
@pytest.mark.asyncio
async def test_read_unsigned_varint_overflow(var_integer):
    s = MockReader(var_integer)
    with pytest.raises(ValueError):
        await read_unsigned_varint(s)


@pytest.mark.parametrize("max_bits", (2, 31, 32, 63, 64, 127, 128))
@pytest.mark.asyncio
async def test_read_write_unsigned_varint_max_bits_edge(max_bits):
    """
    Test the edge with different `max_bits`
    """
    for i in range(-3, 0):
        integer = i + (2 ** max_bits)
        s = MockReaderWriter()
        await write_unsigned_varint(s, integer, max_bits=max_bits)
        s.seek(0, 0)
        result = await read_unsigned_varint(s, max_bits=max_bits)
        assert integer == result


@pytest.fixture(scope="module")
def peer_id_string():
    return "QmS5QmciTXXnCUCyxud5eWFenUMAmvAWSDa1c7dvdXRMZ7"


@pytest.fixture(scope="module")
def peer_id_bytes():
    return b'\x12 7\x87F.[\xb5\xb1o\xe5*\xc7\xb9\xbb\x11:"Z|j2\x8ad\x1b\xa6\xe5<Ip\xfe\xb4\xf5v'


@pytest.fixture(scope="module")
def peer_id(peer_id_bytes):
    return ID(peer_id_bytes)


@pytest.fixture(scope="module")
def maddr():
    return Multiaddr("/unix/123")


def test_peer_id(peer_id_string, peer_id_bytes, peer_id):
    # test initialized with bytes
    assert peer_id.to_bytes() == peer_id_bytes
    assert peer_id.to_string() == peer_id_string
    # test initialized with string
    peer_id_2 = ID.from_base58(peer_id_string)
    assert peer_id_2.to_bytes() == peer_id_bytes
    assert peer_id_2.to_string() == peer_id_string
    # test equal
    assert peer_id == peer_id_2
    # test not equal
    peer_id_3 = ID.from_base58("QmbmfNDEth7Ucvjuxiw3SP3E4PoJzbk7g4Ge6ZDigbCsNp")
    assert peer_id != peer_id_3


def test_stream_info(peer_id, maddr):
    proto = "123"
    # test case: `StreamInfo.__init__`
    si = StreamInfo(peer_id, maddr, proto)
    assert si.peer_id == peer_id
    assert si.addr == maddr
    assert si.proto == proto
    # test case: `StreamInfo.to_pb`
    pb_si = si.to_pb()
    assert pb_si.peer == peer_id.to_bytes()
    assert pb_si.addr == maddr.to_bytes()
    assert pb_si.proto == si.proto
    # test case: `StreamInfo.from_pb`
    si_1 = StreamInfo.from_pb(pb_si)
    assert si_1.peer_id == peer_id
    assert si_1.addr == maddr
    assert si_1.proto == proto


def test_peer_info(peer_id, maddr):
    pi = PeerInfo(peer_id, [maddr])
    # test case: `PeerInfo.__init__`
    assert pi.peer_id == peer_id
    assert pi.addrs == [maddr]
    # test case: `PeerInfo.from_pb`
    pi_pb = p2pd_pb.PeerInfo(id=peer_id.to_bytes(), addrs=[maddr.to_bytes()])
    pi_1 = PeerInfo.from_pb(pi_pb)
    assert pi.peer_id == pi_1.peer_id
    assert pi.addrs == pi_1.addrs


@pytest.mark.parametrize(
    "maddr_str, expected_proto",
    (("/unix/123", protocols.P_UNIX), ("/ip4/127.0.0.1/tcp/7777", protocols.P_IP4)),
)
def test_parse_conn_protocol_valid(maddr_str, expected_proto):
    assert parse_conn_protocol(Multiaddr(maddr_str)) == expected_proto


@pytest.mark.parametrize(
    "maddr_str",
    (
        "/p2p/QmbHVEEepCi7rn7VL7Exxpd2Ci9NNB6ifvqwhsrbRMgQFP",
        "/onion/timaq4ygg2iegci7:1234",
    ),
)
def test_parse_conn_protocol_invalid(maddr_str):
    maddr = Multiaddr(maddr_str)
    with pytest.raises(ValueError):
        parse_conn_protocol(maddr)


@pytest.mark.parametrize("control_maddr_str", ("/unix/123", "/ip4/127.0.0.1/tcp/6666"))
def test_client_ctor_control_maddr(control_maddr_str):
    c = DaemonConnector(Multiaddr(control_maddr_str))
    assert c.control_maddr == Multiaddr(control_maddr_str)


def test_client_ctor_default_control_maddr():
    c = DaemonConnector()
    assert c.control_maddr == Multiaddr(config.control_maddr_str)


@pytest.mark.parametrize("listen_maddr_str", ("/unix/123", "/ip4/127.0.0.1/tcp/6666"))
def test_control_client_ctor_listen_maddr(listen_maddr_str):
    c = ControlClient(
        daemon_connector=DaemonConnector(), listen_maddr=Multiaddr(listen_maddr_str)
    )
    assert c.listen_maddr == Multiaddr(listen_maddr_str)


def test_control_client_ctor_default_listen_maddr():
    c = ControlClient(daemon_connector=DaemonConnector())
    assert c.listen_maddr == Multiaddr(config.listen_maddr_str)


@pytest.mark.parametrize(
    "msg_bytes",
    (
        b'\x08\x00"R\n"\x12 F\xec\xd3p0X\xbeT\x95p^\xc8{\xc8\x13\xa3\x9c\x84d\x0b\x1b\xbb\xa0P\x98w\xc1\xb3\x981i\x16\x12\x02\xa2\x02\x12\x08\x04\x7f\x00\x00\x01\x06\xc7\xb6\x12\x08\x04\xc0\xa8\n\x87\x06\xc7\xb6\x12\x14)\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x01\x06\xc7\xb7',  # noqa: E501
        b'\x08\x00"R\n"\x12 \xd0\xf0 \x9a\xc6v\xa6\xd3;\xcac|\x95\x94\xa0\xe6:\nM\xc53T\x0e\xf0\x89\x8e(\x0c\xb9\xf7\\\xa5\x12\x02\xa2\x02\x12\x08\x04\x7f\x00\x00\x01\x06\xc9%\x12\x08\x04\xc0\xa8\n\x87\x06\xc9%\x12\x14)\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x01\x06\xc9&',  # noqa: E501
        b'\x08\x00"R\n"\x12 \xc3\xc3\xee\x18i\x8a\xde\x13\xa9y\x905\xeb\xcb\xa4\xd07\x14\xbe\xf4\xf8\x1b\xe8[g94\x94\xe3f\x18\xa9\x12\x02\xa2\x02\x12\x08\x04\x7f\x00\x00\x01\x06\xc9`\x12\x08\x04\xc0\xa8\n\x87\x06\xc9`\x12\x14)\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x01\x06\xc9a',  # noqa: E501
    ),
    # give test cases ids to prevent bytes from ruining the terminal
    ids=("pb example Response 0", "pb example Response 1", "pb example Response 2"),
)
@pytest.mark.asyncio
async def test_read_pbmsg_safe_valid(msg_bytes):
    s = MockReaderWriter()
    await write_unsigned_varint(s, len(msg_bytes))
    s.write(msg_bytes)
    # reset the offset back to the beginning
    s.seek(0, 0)
    pb_msg = p2pd_pb.Response()
    await read_pbmsg_safe(s, pb_msg)
    assert pb_msg.SerializeToString() == msg_bytes


@pytest.mark.parametrize(
    "pb_msg, msg_bytes",
    (
        (
            p2pd_pb.Response(),
            b'Z\x08\x00*V\x08\x01\x12R\n"\x12 \x03\x8d\xf5\xd4(/#\xd6\xed\xa5\x1bU\xb8s\x8c\xfa\xad\xfc{\x04\xe3\xecw\xdeK\xc9,\xfe\x9c\x00:\xc8\x12\x02\xa2\x02\x12\x08\x04\x7f\x00\x00\x01\x06\xdea\x12\x08\x04\xc0\xa8\n\x87\x06\xdea\x12\x14)\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x01\x06\xdeb',  # noqa: E501
        ),
        (p2pd_pb.Request(), b"\x02\x08\x05"),
        (
            p2pd_pb.DHTRequest(),
            b'&\x08\x00\x12"\x12 \xd5\x0b\x18/\x9e\xa5G\x06.\xdd\xebW\xf0N\xf5\x0eW\xd3\xec\xdf\x06\x02\xe2\x89\x1e\xf0\xbb.\xc0\xbdE\xb8',  # noqa: E501
        ),
        (
            p2pd_pb.DHTResponse(),
            b'V\x08\x01\x12R\n"\x12 wy\xe2\xfa\x11\x9e\xe2\x84X]\x84\xf8\x98\xba\x8c\x8cQ\xd7,\xb59\x1e!G\x92\x86G{\x141\xe9\x1b\x12\x02\xa2\x02\x12\x08\x04\x7f\x00\x00\x01\x06\xdeA\x12\x08\x04\xc0\xa8\n\x87\x06\xdeA\x12\x14)\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x01\x06\xdeB',  # noqa: E501
        ),
        (
            p2pd_pb.StreamInfo(),
            b';\n"\x12 \xf6\x9e=\x9f\xc1J\xfe\x02\x93k!S\x80\xa0\xcc(s\xea&\xbe\xed\x9274qTI\xc1\xf7\xb6\xbd7\x12\x08\x04\x7f\x00\x00\x01\x06\xde\xc5\x1a\x0bprotocol123',  # noqa: E501
        ),
    ),
    ids=(
        "pb example Response",
        "pb example Request",
        "pb example DHTRequest",
        "pb example DHTResponse",
        "pb example StreamInfo",
    ),
)
@pytest.mark.asyncio
async def test_write_pbmsg(pb_msg, msg_bytes):
    s_read = MockReaderWriter(msg_bytes)
    await read_pbmsg_safe(s_read, pb_msg)
    s_write = MockReaderWriter()
    await write_pbmsg(s_write, pb_msg)
    assert msg_bytes == s_write.getvalue()


@pytest.mark.parametrize(
    "pb_msg",
    (
        p2pd_pb.Response(),
        p2pd_pb.Request(),
        p2pd_pb.DHTRequest(),
        p2pd_pb.DHTResponse(),
        p2pd_pb.StreamInfo(),
    ),
)
@pytest.mark.asyncio
async def test_write_pbmsg_missing_fields(pb_msg):
    with pytest.raises(EncodeError):
        await write_pbmsg(MockReaderWriter(), pb_msg)

TIMEOUT_DURATION = 30  # seconds

@pytest.fixture
def num_p2pds():
    return 4


@pytest.fixture(scope="module")
def peer_id_random():
    return ID.from_base58("QmcgpsyWgH8Y8ajJz1Cu72KnS5uo2Aa2LpzU7kinSupNK1")


@pytest.fixture
def enable_control():
    return True


@pytest.fixture
def enable_connmgr():
    return False


@pytest.fixture
def enable_dht():
    return False


@pytest.fixture
def enable_pubsub():
    return False


@pytest.fixture
def func_make_p2pd_pair():
    return make_p2pd_pair_ip4


async def try_until_success(coro_func, timeout=TIMEOUT_DURATION):
    """
    Keep running ``coro_func`` until the time is out.
    All arguments of ``coro_func`` should be filled, i.e. it should be called without arguments.
    """
    t_start = time.monotonic()
    while True:
        result = await coro_func()
        if result:
            break
        if (time.monotonic() - t_start) >= timeout:
            # timeout
            assert False, f"{coro_func} still failed after `{timeout}` seconds"
        await asyncio.sleep(0.01)


class Daemon:
    control_maddr = None
    proc_daemon = None
    log_filename = ""
    f_log = None
    closed = None

    def __init__(
        self, control_maddr, enable_control, enable_connmgr, enable_dht, enable_pubsub
    ):
        self.control_maddr = control_maddr
        self.enable_control = enable_control
        self.enable_connmgr = enable_connmgr
        self.enable_dht = enable_dht
        self.enable_pubsub = enable_pubsub
        self.is_closed = False
        self._start_logging()
        self._run()

    def _start_logging(self):
        name_control_maddr = str(self.control_maddr).replace("/", "_").replace(".", "_")
        self.log_filename = f"/tmp/log_p2pd{name_control_maddr}.txt"
        self.f_log = open(self.log_filename, "wb")

    def _run(self):
        cmd_list = ["hivemind/hivemind_cli/p2pd", f"-listen={str(self.control_maddr)}"]
        cmd_list += [f"-hostAddrs=/ip4/127.0.0.1/tcp/{get_unused_tcp_port()}"]
        if self.enable_connmgr:
            cmd_list += ["-connManager=true", "-connLo=1", "-connHi=2", "-connGrace=0"]
        if self.enable_dht:
            cmd_list += ["-dht=true"]
        if self.enable_pubsub:
            cmd_list += ["-pubsub=true", "-pubsubRouter=gossipsub"]
        self.proc_daemon = subprocess.Popen(
            cmd_list, stdout=self.f_log, stderr=self.f_log, bufsize=0
        )

    async def wait_until_ready(self):
        lines_head_pattern = (b"Control socket:", b"Peer ID:", b"Peer Addrs:")
        lines_head_occurred = {line: False for line in lines_head_pattern}

        with open(self.log_filename, "rb") as f_log_read:

            async def read_from_daemon_and_check():
                line = f_log_read.readline()
                for head_pattern in lines_head_occurred:
                    if line.startswith(head_pattern):
                        lines_head_occurred[head_pattern] = True
                return all([value for _, value in lines_head_occurred.items()])

            await try_until_success(read_from_daemon_and_check)

        # sleep for a while in case that the daemon haven't been ready after emitting these lines
        await asyncio.sleep(0.1)

    def close(self):
        if self.is_closed:
            return
        self.proc_daemon.terminate()
        self.proc_daemon.wait()
        self.f_log.close()
        self.is_closed = True


class DaemonTuple(NamedTuple):
    daemon: Daemon
    client: Client


class ConnectionFailure(Exception):
    pass


@asynccontextmanager
async def make_p2pd_pair_unix(
    enable_control, enable_connmgr, enable_dht, enable_pubsub
):
    name = str(uuid.uuid4())[:8]
    control_maddr = Multiaddr(f"/unix/tmp/test_p2pd_control_{name}.sock")
    listen_maddr = Multiaddr(f"/unix/tmp/test_p2pd_listen_{name}.sock")
    # Remove the existing unix socket files if they are existing
    try:
        os.unlink(control_maddr.value_for_protocol(protocols.P_UNIX))
    except FileNotFoundError:
        pass
    try:
        os.unlink(listen_maddr.value_for_protocol(protocols.P_UNIX))
    except FileNotFoundError:
        pass
    async with _make_p2pd_pair(
        control_maddr=control_maddr,
        listen_maddr=listen_maddr,
        enable_control=enable_control,
        enable_connmgr=enable_connmgr,
        enable_dht=enable_dht,
        enable_pubsub=enable_pubsub,
    ) as pair:
        yield pair


@asynccontextmanager
async def make_p2pd_pair_ip4(enable_control, enable_connmgr, enable_dht, enable_pubsub):
    control_maddr = Multiaddr(f"/ip4/127.0.0.1/tcp/{get_unused_tcp_port()}")
    listen_maddr = Multiaddr(f"/ip4/127.0.0.1/tcp/{get_unused_tcp_port()}")
    async with _make_p2pd_pair(
        control_maddr=control_maddr,
        listen_maddr=listen_maddr,
        enable_control=enable_control,
        enable_connmgr=enable_connmgr,
        enable_dht=enable_dht,
        enable_pubsub=enable_pubsub,
    ) as pair:
        yield pair


@asynccontextmanager
async def _make_p2pd_pair(
    control_maddr,
    listen_maddr,
    enable_control,
    enable_connmgr,
    enable_dht,
    enable_pubsub,
):
    p2pd = Daemon(
        control_maddr=control_maddr,
        enable_control=enable_control,
        enable_connmgr=enable_connmgr,
        enable_dht=enable_dht,
        enable_pubsub=enable_pubsub,
    )
    # wait for daemon ready
    await p2pd.wait_until_ready()
    client = Client(control_maddr=control_maddr, listen_maddr=listen_maddr)
    try:
        async with client.listen():
            yield DaemonTuple(daemon=p2pd, client=client)
    finally:
        if not p2pd.is_closed:
            p2pd.close()


@pytest.fixture
async def p2pcs(
    num_p2pds,
    enable_control,
    enable_connmgr,
    enable_dht,
    enable_pubsub,
    func_make_p2pd_pair,
):
    # TODO: Change back to gather style
    async with AsyncExitStack() as stack:
        p2pd_tuples = [
            await stack.enter_async_context(
                func_make_p2pd_pair(
                    enable_control=enable_control,
                    enable_connmgr=enable_connmgr,
                    enable_dht=enable_dht,
                    enable_pubsub=enable_pubsub,
                )
            )
            for _ in range(num_p2pds)
        ]
        yield tuple(p2pd_tuple.client for p2pd_tuple in p2pd_tuples)


@pytest.mark.parametrize(
    "enable_control, func_make_p2pd_pair", ((True, make_p2pd_pair_unix),)
)
@pytest.mark.asyncio
async def test_client_identify_unix_socket(p2pcs):
    await p2pcs[0].identify()


@pytest.mark.parametrize("enable_control", (True,))
@pytest.mark.asyncio
async def test_client_identify(p2pcs):
    await p2pcs[0].identify()


@pytest.mark.parametrize("enable_control", (True,))
@pytest.mark.asyncio
async def test_client_connect_success(p2pcs):
    peer_id_0, maddrs_0 = await p2pcs[0].identify()
    peer_id_1, maddrs_1 = await p2pcs[1].identify()
    await p2pcs[0].connect(peer_id_1, maddrs_1)
    # test case: repeated connections
    await p2pcs[1].connect(peer_id_0, maddrs_0)


@pytest.mark.parametrize("enable_control", (True,))
@pytest.mark.asyncio
async def test_client_connect_failure(peer_id_random, p2pcs):
    peer_id_1, maddrs_1 = await p2pcs[1].identify()
    await p2pcs[0].identify()
    # test case: `peer_id` mismatches
    with pytest.raises(ControlFailure):
        await p2pcs[0].connect(peer_id_random, maddrs_1)
    # test case: empty maddrs
    with pytest.raises(ControlFailure):
        await p2pcs[0].connect(peer_id_1, [])
    # test case: wrong maddrs
    with pytest.raises(ControlFailure):
        await p2pcs[0].connect(peer_id_1, [Multiaddr("/ip4/127.0.0.1/udp/0")])


async def _check_connection(p2pd_tuple_0, p2pd_tuple_1):
    peer_id_0, _ = await p2pd_tuple_0.identify()
    peer_id_1, _ = await p2pd_tuple_1.identify()
    peers_0 = [pinfo.peer_id for pinfo in await p2pd_tuple_0.list_peers()]
    peers_1 = [pinfo.peer_id for pinfo in await p2pd_tuple_1.list_peers()]
    return (peer_id_0 in peers_1) and (peer_id_1 in peers_0)


async def connect_safe(p2pd_tuple_0, p2pd_tuple_1):
    peer_id_1, maddrs_1 = await p2pd_tuple_1.identify()
    await p2pd_tuple_0.connect(peer_id_1, maddrs_1)
    await try_until_success(
        functools.partial(
            _check_connection, p2pd_tuple_0=p2pd_tuple_0, p2pd_tuple_1=p2pd_tuple_1
        )
    )


@pytest.mark.parametrize("enable_control", (True,))
@pytest.mark.asyncio
async def test_connect_safe(p2pcs):
    await connect_safe(p2pcs[0], p2pcs[1])


@pytest.mark.parametrize("enable_control", (True,))
@pytest.mark.asyncio
async def test_client_list_peers(p2pcs):
    # test case: no peers
    assert len(await p2pcs[0].list_peers()) == 0
    # test case: 1 peer
    await connect_safe(p2pcs[0], p2pcs[1])
    assert len(await p2pcs[0].list_peers()) == 1
    assert len(await p2pcs[1].list_peers()) == 1
    # test case: one more peer
    await connect_safe(p2pcs[0], p2pcs[2])
    assert len(await p2pcs[0].list_peers()) == 2
    assert len(await p2pcs[1].list_peers()) == 1
    assert len(await p2pcs[2].list_peers()) == 1


@pytest.mark.parametrize("enable_control", (True,))
@pytest.mark.asyncio
async def test_client_disconnect(peer_id_random, p2pcs):
    # test case: disconnect a peer without connections
    await p2pcs[1].disconnect(peer_id_random)
    # test case: disconnect
    peer_id_0, _ = await p2pcs[0].identify()
    await connect_safe(p2pcs[0], p2pcs[1])
    assert len(await p2pcs[0].list_peers()) == 1
    assert len(await p2pcs[1].list_peers()) == 1
    await p2pcs[1].disconnect(peer_id_0)
    assert len(await p2pcs[0].list_peers()) == 0
    assert len(await p2pcs[1].list_peers()) == 0
    # test case: disconnect twice
    await p2pcs[1].disconnect(peer_id_0)
    assert len(await p2pcs[0].list_peers()) == 0
    assert len(await p2pcs[1].list_peers()) == 0


@pytest.mark.parametrize("enable_control", (True,))
@pytest.mark.asyncio
async def test_client_stream_open_success(p2pcs):
    peer_id_1, maddrs_1 = await p2pcs[1].identify()
    await connect_safe(p2pcs[0], p2pcs[1])

    proto = "123"

    async def handle_proto(stream_info, reader, writer):
        await reader.readexactly(1)

    await p2pcs[1].stream_handler(proto, handle_proto)

    # test case: normal
    stream_info, reader, writer = await p2pcs[0].stream_open(peer_id_1, (proto,))
    assert stream_info.peer_id == peer_id_1
    assert stream_info.addr in maddrs_1
    assert stream_info.proto == "123"
    writer.close()

    # test case: open with multiple protocols
    stream_info, reader, writer = await p2pcs[0].stream_open(
        peer_id_1, (proto, "another_protocol")
    )
    assert stream_info.peer_id == peer_id_1
    assert stream_info.addr in maddrs_1
    assert stream_info.proto == "123"
    writer.close()


@pytest.mark.parametrize("enable_control", (True,))
@pytest.mark.asyncio
async def test_client_stream_open_failure(p2pcs):
    peer_id_1, _ = await p2pcs[1].identify()
    await connect_safe(p2pcs[0], p2pcs[1])

    proto = "123"

    # test case: `stream_open` to a peer who didn't register the protocol
    with pytest.raises(ControlFailure):
        await p2pcs[0].stream_open(peer_id_1, (proto,))

    # test case: `stream_open` to a peer for a non-registered protocol
    async def handle_proto(stream_info, reader, writer):
        pass

    await p2pcs[1].stream_handler(proto, handle_proto)
    with pytest.raises(ControlFailure):
        await p2pcs[0].stream_open(peer_id_1, ("another_protocol",))


@pytest.mark.parametrize("enable_control", (True,))
@pytest.mark.asyncio
async def test_client_stream_handler_success(p2pcs):
    peer_id_1, _ = await p2pcs[1].identify()
    await connect_safe(p2pcs[0], p2pcs[1])

    proto = "protocol123"
    bytes_to_send = b"yoyoyoyoyog"
    # event for this test function to wait until the handler function receiving the incoming data
    event_handler_finished = asyncio.Event()

    async def handle_proto(stream_info, reader, writer):
        nonlocal event_handler_finished
        bytes_received = await reader.readexactly(len(bytes_to_send))
        assert bytes_received == bytes_to_send
        event_handler_finished.set()

    await p2pcs[1].stream_handler(proto, handle_proto)
    assert proto in p2pcs[1].control.handlers
    assert handle_proto == p2pcs[1].control.handlers[proto]

    # test case: test the stream handler `handle_proto`

    _, reader, writer = await p2pcs[0].stream_open(peer_id_1, (proto,))

    # wait until the handler function starts blocking waiting for the data
    # because we haven't sent the data, we know the handler function must still blocking waiting.
    # get the task of the protocol handler
    writer.write(bytes_to_send)

    # wait for the handler to finish
    writer.close()

    await event_handler_finished.wait()

    # test case: two streams to different handlers respectively
    another_proto = "another_protocol123"
    another_bytes_to_send = b"456"
    event_another_proto = asyncio.Event()

    async def handle_another_proto(stream_info, reader, writer):
        event_another_proto.set()
        bytes_received = await reader.readexactly(len(another_bytes_to_send))
        assert bytes_received == another_bytes_to_send

    await p2pcs[1].stream_handler(another_proto, handle_another_proto)
    assert another_proto in p2pcs[1].control.handlers
    assert handle_another_proto == p2pcs[1].control.handlers[another_proto]

    _, reader, writer = await p2pcs[0].stream_open(peer_id_1, (another_proto,))
    await event_another_proto.wait()

    # we know at this moment the handler must still blocking wait

    writer.write(another_bytes_to_send)

    writer.close()

    # test case: registering twice can override the previous registration
    event_third = asyncio.Event()

    async def handler_third(stream_info, reader, writer):
        event_third.set()

    await p2pcs[1].stream_handler(another_proto, handler_third)
    assert another_proto in p2pcs[1].control.handlers
    # ensure the handler is override
    assert handler_third == p2pcs[1].control.handlers[another_proto]

    await p2pcs[0].stream_open(peer_id_1, (another_proto,))
    # ensure the overriding handler is called when the protocol is opened a stream
    await event_third.wait()


@pytest.mark.parametrize("enable_control", (True,))
@pytest.mark.asyncio
async def test_client_stream_handler_failure(p2pcs):
    peer_id_1, _ = await p2pcs[1].identify()
    await connect_safe(p2pcs[0], p2pcs[1])

    proto = "123"

    # test case: registered a wrong protocol name
    async def handle_proto_correct_params(stream_info, stream):
        pass

    await p2pcs[1].stream_handler("another_protocol", handle_proto_correct_params)
    with pytest.raises(ControlFailure):
        await p2pcs[0].stream_open(peer_id_1, (proto,))
