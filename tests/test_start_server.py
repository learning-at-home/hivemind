import os
import re
from functools import partial
from subprocess import PIPE, Popen
from tempfile import TemporaryDirectory

from hivemind.moe.server import background_server


def test_background_server_identity_path():
    with TemporaryDirectory() as tempdir:
        id_path = os.path.join(tempdir, "id")

        server_runner = partial(background_server, num_experts=1, device="cpu", hidden_dim=1)

        with server_runner(identity_path=id_path) as server_info_1, server_runner(
            identity_path=id_path
        ) as server_info_2, server_runner(identity_path=None) as server_info_3:
            assert server_info_1.peer_id == server_info_2.peer_id
            assert server_info_1.peer_id != server_info_3.peer_id
            assert server_info_3.peer_id == server_info_3.peer_id


def test_cli_run_server_identity_path():
    pattern = r"Running DHT node on \[(.+)\],"

    with TemporaryDirectory() as tempdir:
        id_path = os.path.join(tempdir, "id")

        server_1_proc = Popen(
            ["hivemind-server", "--num_experts", "1", "--identity_path", id_path],
            stderr=PIPE,
            text=True,
            encoding="utf-8",
        )

        line = server_1_proc.stderr.readline()
        assert "Generating new identity" in line

        line = server_1_proc.stderr.readline()
        addrs_pattern_result = re.search(pattern, line)
        assert addrs_pattern_result is not None, line
        addrs_1 = set(addrs_pattern_result.group(1).split(", "))
        ids_1 = set(a.split("/")[-1] for a in addrs_1)

        assert len(ids_1) == 1

        server_2_proc = Popen(
            ["hivemind-server", "--num_experts", "1", "--identity_path", id_path],
            stderr=PIPE,
            text=True,
            encoding="utf-8",
        )

        line = server_2_proc.stderr.readline()
        addrs_pattern_result = re.search(pattern, line)
        assert addrs_pattern_result is not None, line
        addrs_2 = set(addrs_pattern_result.group(1).split(", "))
        ids_2 = set(a.split("/")[-1] for a in addrs_2)

        assert len(ids_2) == 1

        server_3_proc = Popen(
            ["hivemind-server", "--num_experts", "1"],
            stderr=PIPE,
            text=True,
            encoding="utf-8",
        )

        line = server_3_proc.stderr.readline()
        addrs_pattern_result = re.search(pattern, line)
        assert addrs_pattern_result is not None, line
        addrs_3 = set(addrs_pattern_result.group(1).split(", "))
        ids_3 = set(a.split("/")[-1] for a in addrs_3)

        assert len(ids_3) == 1

        assert ids_1 == ids_2
        assert ids_1 != ids_3
        assert ids_2 != ids_3

        assert addrs_1 != addrs_2
        assert addrs_1 != addrs_3
        assert addrs_2 != addrs_3

        server_1_proc.terminate()
        server_2_proc.terminate()
        server_3_proc.terminate()

        server_1_proc.wait()
        server_2_proc.wait()
        server_3_proc.wait()
