import re
from subprocess import PIPE, Popen
from time import sleep

DHT_START_PATTERN = re.compile(r"Running a DHT instance. To connect other peers to this one, use (.+)$")


def test_dht_connection_successful():
    dht_refresh_period = 1

    dht_proc = Popen(
        ["hivemind-dht", "--host_maddrs", "/ip4/127.0.0.1/tcp/0", "--refresh_period", str(dht_refresh_period)],
        stderr=PIPE,
        text=True,
        encoding="utf-8",
    )

    first_line = dht_proc.stderr.readline()
    second_line = dht_proc.stderr.readline()
    dht_pattern_match = DHT_START_PATTERN.search(first_line)
    assert dht_pattern_match is not None, first_line
    assert "Full list of visible multiaddresses:" in second_line, second_line

    initial_peers = dht_pattern_match.group(1).split(" ")

    dht_client_proc = Popen(
        ["hivemind-dht", *initial_peers, "--host_maddrs", "/ip4/127.0.0.1/tcp/0"],
        stderr=PIPE,
        text=True,
        encoding="utf-8",
    )

    # skip first two lines with connectivity info
    for _ in range(2):
        dht_client_proc.stderr.readline()
    first_report_msg = dht_client_proc.stderr.readline()

    assert "2 DHT nodes (including this one) are in the local routing table" in first_report_msg, first_report_msg

    # ensure we get the output of dht_proc after the start of dht_client_proc
    sleep(dht_refresh_period)

    # expect that one of the next logging outputs from the first peer shows a new connection
    for _ in range(5):
        first_report_msg = dht_proc.stderr.readline()
        second_report_msg = dht_proc.stderr.readline()

        if (
            "2 DHT nodes (including this one) are in the local routing table" in first_report_msg
            and "Local storage contains 0 keys" in second_report_msg
        ):
            break
    else:
        assert (
            "2 DHT nodes (including this one) are in the local routing table" in first_report_msg
            and "Local storage contains 0 keys" in second_report_msg
        )

    dht_proc.terminate()
    dht_client_proc.terminate()

    dht_proc.wait()
    dht_client_proc.wait()
