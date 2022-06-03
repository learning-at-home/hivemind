import asyncio
import io
from contextlib import AsyncExitStack

import pytest
from google.protobuf.message import EncodeError
from multiaddr import Multiaddr, protocols

from hivemind.p2p.p2p_daemon_bindings.control import ControlClient, DaemonConnector, parse_conn_protocol
from hivemind.p2p.p2p_daemon_bindings.datastructures import PeerID, PeerInfo, StreamInfo
from hivemind.p2p.p2p_daemon_bindings.utils import (
    ControlFailure,
    raise_if_failed,
    read_pbmsg_safe,
    read_unsigned_varint,
    write_pbmsg,
    write_unsigned_varint,
)
from hivemind.proto import p2pd_pb2 as p2pd_pb

from test_utils.p2p_daemon import connect_safe, make_p2pd_pair_unix


def test_raise_if_failed_raises():
    resp = p2pd_pb.Response()
    resp.type = p2pd_pb.Response.ERROR
    with pytest.raises(ControlFailure):
        raise_if_failed(resp)


def test_raise_if_failed_not_raises():
    resp = p2pd_pb.Response()
    resp.type = p2pd_pb.Response.OK
    raise_if_failed(resp)


PAIRS_INT_SERIALIZED_VALID = (
    (0, b"\x00"),
    (1, b"\x01"),
    (128, b"\x80\x01"),
    (2**32, b"\x80\x80\x80\x80\x10"),
    (2**64 - 1, b"\xff\xff\xff\xff\xff\xff\xff\xff\xff\x01"),
)

PAIRS_INT_SERIALIZED_OVERFLOW = (
    (2**64, b"\x80\x80\x80\x80\x80\x80\x80\x80\x80\x02"),
    (2**64 + 1, b"\x81\x80\x80\x80\x80\x80\x80\x80\x80\x02"),
    (
        2**128,
        b"\x80\x80\x80\x80\x80\x80\x80\x80\x80\x80\x80\x80\x80\x80\x80\x80\x80\x80\x04",
    ),
)

PEER_ID_STRING = "QmS5QmciTXXnCUCyxud5eWFenUMAmvAWSDa1c7dvdXRMZ7"
PEER_ID_BYTES = b'\x12 7\x87F.[\xb5\xb1o\xe5*\xc7\xb9\xbb\x11:"Z|j2\x8ad\x1b\xa6\xe5<Ip\xfe\xb4\xf5v'
PEER_ID = PeerID(PEER_ID_BYTES)
MADDR = Multiaddr("/unix/123")
NUM_P2PDS = 4
PEER_ID_RANDOM = PeerID.from_base58("QmcgpsyWgH8Y8ajJz1Cu72KnS5uo2Aa2LpzU7kinSupNK1")
ENABLE_CONTROL = True
ENABLE_CONNMGR = False
ENABLE_DHT = False
ENABLE_PUBSUB = False
FUNC_MAKE_P2PD_PAIR = make_p2pd_pair_unix


class MockReader(io.BytesIO):
    async def readexactly(self, n):
        await asyncio.sleep(0)
        return self.read(n)


class MockWriter(io.BytesIO):
    pass


class MockReaderWriter(MockReader, MockWriter):
    pass


@pytest.mark.parametrize("integer, serialized_integer", PAIRS_INT_SERIALIZED_VALID)
@pytest.mark.asyncio
async def test_write_unsigned_varint(integer, serialized_integer):
    s = MockWriter()
    await write_unsigned_varint(s, integer)
    assert s.getvalue() == serialized_integer


@pytest.mark.parametrize("integer", tuple(i[0] for i in PAIRS_INT_SERIALIZED_OVERFLOW))
@pytest.mark.asyncio
async def test_write_unsigned_varint_overflow(integer):
    s = MockWriter()
    with pytest.raises(ValueError):
        await write_unsigned_varint(s, integer)


@pytest.mark.parametrize("integer", (-1, -(2**32), -(2**64), -(2**128)))
@pytest.mark.asyncio
async def test_write_unsigned_varint_negative(integer):
    s = MockWriter()
    with pytest.raises(ValueError):
        await write_unsigned_varint(s, integer)


@pytest.mark.parametrize("integer, serialized_integer", PAIRS_INT_SERIALIZED_VALID)
@pytest.mark.asyncio
async def test_read_unsigned_varint(integer, serialized_integer):
    s = MockReader(serialized_integer)
    result = await read_unsigned_varint(s)
    assert result == integer


@pytest.mark.parametrize("serialized_integer", tuple(i[1] for i in PAIRS_INT_SERIALIZED_OVERFLOW))
@pytest.mark.asyncio
async def test_read_unsigned_varint_overflow(serialized_integer):
    s = MockReader(serialized_integer)
    with pytest.raises(ValueError):
        await read_unsigned_varint(s)


@pytest.mark.parametrize("max_bits", (2, 31, 32, 63, 64, 127, 128))
@pytest.mark.asyncio
async def test_read_write_unsigned_varint_max_bits_edge(max_bits):
    """
    Test edge cases with different `max_bits`
    """
    for i in range(-3, 0):
        integer = i + 2**max_bits
        s = MockReaderWriter()
        await write_unsigned_varint(s, integer, max_bits=max_bits)
        s.seek(0, 0)
        result = await read_unsigned_varint(s, max_bits=max_bits)
        assert integer == result


def test_peer_id():
    assert PEER_ID.to_bytes() == PEER_ID_BYTES
    assert PEER_ID.to_string() == PEER_ID_STRING

    peer_id_2 = PeerID.from_base58(PEER_ID_STRING)
    assert peer_id_2.to_bytes() == PEER_ID_BYTES
    assert peer_id_2.to_string() == PEER_ID_STRING
    assert PEER_ID == peer_id_2
    peer_id_3 = PeerID.from_base58("QmbmfNDEth7Ucvjuxiw3SP3E4PoJzbk7g4Ge6ZDigbCsNp")
    assert PEER_ID != peer_id_3

    a = PeerID.from_base58("bob")
    b = PeerID.from_base58("eve")
    assert a < b and b > a and not (b < a) and not (a > b)
    with pytest.raises(TypeError):
        assert a < object()


def test_stream_info():
    proto = "123"
    si = StreamInfo(PEER_ID, MADDR, proto)
    assert si.peer_id == PEER_ID
    assert si.addr == MADDR
    assert si.proto == proto
    pb_si = si.to_protobuf()
    assert pb_si.peer == PEER_ID.to_bytes()
    assert pb_si.addr == MADDR.to_bytes()
    assert pb_si.proto == si.proto
    si_1 = StreamInfo.from_protobuf(pb_si)
    assert si_1.peer_id == PEER_ID
    assert si_1.addr == MADDR
    assert si_1.proto == proto


def test_peer_info():
    pi = PeerInfo(PEER_ID, [MADDR])
    assert pi.peer_id == PEER_ID
    assert pi.addrs == [MADDR]
    pi_pb = p2pd_pb.PeerInfo(id=PEER_ID.to_bytes(), addrs=[MADDR.to_bytes()])
    pi_1 = PeerInfo.from_protobuf(pi_pb)
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
@pytest.mark.asyncio
async def test_client_create_control_maddr(control_maddr_str):
    c = DaemonConnector(Multiaddr(control_maddr_str))
    assert c.control_maddr == Multiaddr(control_maddr_str)


def test_client_create_default_control_maddr():
    c = DaemonConnector()
    assert c.control_maddr == Multiaddr(DaemonConnector.DEFAULT_CONTROL_MADDR)


@pytest.mark.parametrize("listen_maddr_str", ("/unix/123", "/ip4/127.0.0.1/tcp/6666"))
@pytest.mark.asyncio
async def test_control_client_create_listen_maddr(listen_maddr_str):
    c = await ControlClient.create(
        daemon_connector=DaemonConnector(),
        listen_maddr=Multiaddr(listen_maddr_str),
        use_persistent_conn=False,
    )
    assert c.listen_maddr == Multiaddr(listen_maddr_str)


@pytest.mark.asyncio
async def test_control_client_create_default_listen_maddr():
    c = await ControlClient.create(daemon_connector=DaemonConnector(), use_persistent_conn=False)
    assert c.listen_maddr == Multiaddr(ControlClient.DEFAULT_LISTEN_MADDR)


@pytest.mark.parametrize(
    "msg_bytes",
    (
        p2pd_pb.Response(
            type=p2pd_pb.Response.Type.OK,
            identify=p2pd_pb.IdentifyResponse(
                id=PeerID.from_base58("QmT7WhTne9zBLfAgAJt9aiZ8jZ5BxJGowRubxsHYmnyzUd").to_bytes(),
                addrs=[
                    Multiaddr("/p2p-circuit").to_bytes(),
                    Multiaddr("/ip4/127.0.0.1/tcp/51126").to_bytes(),
                    Multiaddr("/ip4/192.168.10.135/tcp/51126").to_bytes(),
                    Multiaddr("/ip6/::1/tcp/51127").to_bytes(),
                ],
            ),
        ).SerializeToString(),
        p2pd_pb.Response(
            type=p2pd_pb.Response.Type.OK,
            identify=p2pd_pb.IdentifyResponse(
                id=PeerID.from_base58("QmcQFt2MFfCZ9AxzUCNrk4k7TtMdZZvAAteaA6tHpBKdrk").to_bytes(),
                addrs=[
                    Multiaddr("/p2p-circuit").to_bytes(),
                    Multiaddr("/ip4/127.0.0.1/tcp/51493").to_bytes(),
                    Multiaddr("/ip4/192.168.10.135/tcp/51493").to_bytes(),
                    Multiaddr("/ip6/::1/tcp/51494").to_bytes(),
                ],
            ),
        ).SerializeToString(),
        p2pd_pb.Response(
            type=p2pd_pb.Response.Type.OK,
            identify=p2pd_pb.IdentifyResponse(
                id=PeerID.from_base58("QmbWqVVoz7v9LS9ZUQAhyyfdFJY3iU8ZrUY3XQozoTA5cc").to_bytes(),
                addrs=[
                    Multiaddr("/p2p-circuit").to_bytes(),
                    Multiaddr("/ip4/127.0.0.1/tcp/51552").to_bytes(),
                    Multiaddr("/ip4/192.168.10.135/tcp/51552").to_bytes(),
                    Multiaddr("/ip6/::1/tcp/51553").to_bytes(),
                ],
            ),
        ).SerializeToString(),
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
    "pb_type, pb_msg",
    (
        (
            p2pd_pb.Response,
            p2pd_pb.Response(
                type=p2pd_pb.Response.Type.OK,
                dht=p2pd_pb.DHTResponse(
                    type=p2pd_pb.DHTResponse.Type.VALUE,
                    peer=p2pd_pb.PeerInfo(
                        id=PeerID.from_base58("QmNaXUy78W9moQ9APCoKaTtPjLcEJPN9hRBCqErY7o2fQs").to_bytes(),
                        addrs=[
                            Multiaddr("/p2p-circuit").to_bytes(),
                            Multiaddr("/ip4/127.0.0.1/tcp/56929").to_bytes(),
                            Multiaddr("/ip4/192.168.10.135/tcp/56929").to_bytes(),
                            Multiaddr("/ip6/::1/tcp/56930").to_bytes(),
                        ],
                    ),
                ),
            ),
        ),
        (p2pd_pb.Request, p2pd_pb.Request(type=p2pd_pb.Request.Type.LIST_PEERS)),
        (
            p2pd_pb.DHTRequest,
            p2pd_pb.DHTRequest(
                type=p2pd_pb.DHTRequest.Type.FIND_PEER,
                peer=PeerID.from_base58("QmcgHMuEhqdLHDVeNjiCGU7Ds6E7xK3f4amgiwHNPKKn7R").to_bytes(),
            ),
        ),
        (
            p2pd_pb.DHTResponse,
            p2pd_pb.DHTResponse(
                type=p2pd_pb.DHTResponse.Type.VALUE,
                peer=p2pd_pb.PeerInfo(
                    id=PeerID.from_base58("QmWP32GhEyXVQsLXFvV81eadDC8zQRZxZvJK359rXxLquk").to_bytes(),
                    addrs=[
                        Multiaddr("/p2p-circuit").to_bytes(),
                        Multiaddr("/ip4/127.0.0.1/tcp/56897").to_bytes(),
                        Multiaddr("/ip4/192.168.10.135/tcp/56897").to_bytes(),
                        Multiaddr("/ip6/::1/tcp/56898").to_bytes(),
                    ],
                ),
            ),
        ),
        (
            p2pd_pb.StreamInfo,
            p2pd_pb.StreamInfo(
                peer=PeerID.from_base58("QmewLxB46MftfxQiunRgJo2W8nW4Lh5NLEkRohkHhJ4wW6").to_bytes(),
                addr=Multiaddr("/ip4/127.0.0.1/tcp/57029").to_bytes(),
                proto=b"protocol123",
            ),
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
async def test_write_pbmsg(pb_type, pb_msg):
    msg_bytes = bytes(chr(pb_msg.ByteSize()), "utf-8") + pb_msg.SerializeToString()
    pb_obj = pb_type()

    s_read = MockReaderWriter(msg_bytes)
    await read_pbmsg_safe(s_read, pb_obj)
    s_write = MockReaderWriter()
    await write_pbmsg(s_write, pb_obj)
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


@pytest.fixture
async def p2pcs():
    # TODO: Change back to gather style
    async with AsyncExitStack() as stack:
        p2pd_tuples = [
            await stack.enter_async_context(
                FUNC_MAKE_P2PD_PAIR(
                    enable_control=ENABLE_CONTROL,
                    enable_connmgr=ENABLE_CONNMGR,
                    enable_dht=ENABLE_DHT,
                    enable_pubsub=ENABLE_PUBSUB,
                )
            )
            for _ in range(NUM_P2PDS)
        ]
        yield tuple(p2pd_tuple.client for p2pd_tuple in p2pd_tuples)


@pytest.mark.asyncio
async def test_client_identify(p2pcs):
    await p2pcs[0].identify()


@pytest.mark.asyncio
async def test_client_connect_success(p2pcs):
    peer_id_0, maddrs_0 = await p2pcs[0].identify()
    peer_id_1, maddrs_1 = await p2pcs[1].identify()
    await p2pcs[0].connect(peer_id_1, maddrs_1)
    # test case: repeated connections
    await p2pcs[1].connect(peer_id_0, maddrs_0)


@pytest.mark.asyncio
async def test_client_connect_failure(p2pcs):
    peer_id_1, maddrs_1 = await p2pcs[1].identify()
    await p2pcs[0].identify()
    # test case: `peer_id` mismatches
    with pytest.raises(ControlFailure):
        await p2pcs[0].connect(PEER_ID_RANDOM, maddrs_1)
    # test case: empty maddrs
    with pytest.raises(ControlFailure):
        await p2pcs[0].connect(peer_id_1, [])
    # test case: wrong maddrs
    with pytest.raises(ControlFailure):
        await p2pcs[0].connect(peer_id_1, [Multiaddr("/ip4/127.0.0.1/udp/0")])


@pytest.mark.asyncio
async def test_connect_safe(p2pcs):
    await connect_safe(p2pcs[0], p2pcs[1])


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


@pytest.mark.asyncio
async def test_client_disconnect(p2pcs):
    # test case: disconnect a peer without connections
    await p2pcs[1].disconnect(PEER_ID_RANDOM)
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
    stream_info, reader, writer = await p2pcs[0].stream_open(peer_id_1, (proto, "another_protocol"))
    assert stream_info.peer_id == peer_id_1
    assert stream_info.addr in maddrs_1
    assert stream_info.proto == "123"
    writer.close()


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

    # test case: registering twice can't override the previous registration without balanced flag
    event_third = asyncio.Event()

    async def handler_third(stream_info, reader, writer):
        event_third.set()

    # p2p raises now for doubled stream handlers
    with pytest.raises(ControlFailure):
        await p2pcs[1].stream_handler(another_proto, handler_third)

    # add in balanced mode: handler should be placed in round robin queue
    # and become the next to be called
    await p2pcs[1].stream_handler(another_proto, handler_third, balanced=True)
    assert another_proto in p2pcs[1].control.handlers
    # ensure the handler is override
    assert handler_third == p2pcs[1].control.handlers[another_proto]

    await p2pcs[0].stream_open(peer_id_1, (another_proto,))
    # ensure the overriding handler is called when the protocol is opened a stream
    await event_third.wait()


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
