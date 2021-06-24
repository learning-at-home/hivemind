import asyncio
import functools
import os
import subprocess
import time
import uuid
from contextlib import asynccontextmanager
from typing import NamedTuple
from pkg_resources import resource_filename

from multiaddr import Multiaddr, protocols

from hivemind import find_open_port
from hivemind.p2p.p2p_daemon_bindings.p2pclient import Client


TIMEOUT_DURATION = 30  # seconds
P2PD_PATH = resource_filename("hivemind", "hivemind_cli/p2pd")


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
        cmd_list = [P2PD_PATH, f"-listen={str(self.control_maddr)}"]
        cmd_list += [f"-hostAddrs=/ip4/127.0.0.1/tcp/{find_open_port()}"]
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
    control_maddr = Multiaddr(f"/ip4/127.0.0.1/tcp/{find_open_port()}")
    listen_maddr = Multiaddr(f"/ip4/127.0.0.1/tcp/{find_open_port()}")
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
