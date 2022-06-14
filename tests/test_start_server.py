import os
import re
from subprocess import PIPE, Popen
from tempfile import TemporaryDirectory

from hivemind.moe.server import background_server


def test_background_server_identity_path():
    with TemporaryDirectory() as tempdir:
        id_path = os.path.join(tempdir, "id")

        with background_server(
            num_experts=2,
            device="cpu",
            hidden_dim=16,
            num_handlers=2,
            identity_path=id_path,
        ) as server_info_1, background_server(
            num_experts=2,
            device="cpu",
            hidden_dim=16,
            num_handlers=2,
            identity_path=id_path,
        ) as server_info_2, background_server(
            num_experts=2,
            device="cpu",
            hidden_dim=16,
            num_handlers=2,
            identity_path=None,
        ) as server_info_3:

            assert server_info_1.peer_id == server_info_2.peer_id
            assert server_info_1.peer_id != server_info_3.peer_id
            assert server_info_3.peer_id == server_info_3.peer_id


def test_cli_run_server_identity_path():
    pattern = r"Running DHT node on (\[.+\]),"

    with TemporaryDirectory() as tempdir:
        id_path = os.path.join(tempdir, "id")

        server_1_proc = Popen(
            ["hivemind-server", "--expert_cls", "ffn", "--hidden_dim", "512", "--identity_path", id_path],
            stderr=PIPE,
            text=True,
            encoding="utf-8",
        )

        # Skip line "Generating new identity (libp2p private key) in"
        line = server_1_proc.stderr.readline()
        line = server_1_proc.stderr.readline()
        addrs_1 = set(re.search(pattern, line).group(1)[1:-1].split(", "))
        ids_1 = set(a.split("/")[-1] for a in addrs_1)

        assert len(ids_1) == 1

        server_2_proc = Popen(
            ["hivemind-server", "--expert_cls", "ffn", "--hidden_dim", "512", "--identity_path", id_path],
            stderr=PIPE,
            text=True,
            encoding="utf-8",
        )

        line = server_2_proc.stderr.readline()
        addrs_2 = set(re.search(pattern, line).group(1)[1:-1].split(", "))
        ids_2 = set(a.split("/")[-1] for a in addrs_2)

        assert len(ids_2) == 1

        assert ids_1 == ids_2
        assert addrs_1 != addrs_2

        server_1_proc.terminate()
        server_2_proc.terminate()

        server_1_proc.wait()
        server_2_proc.wait()
