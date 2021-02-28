import subprocess

import pytest

import hivemind.p2p
from hivemind.p2p import P2P

RUNNING = 'running'
NOT_RUNNING = 'not running'
CHECK_PID_CMD = '''
if ps -p {0} > /dev/null;
then
    echo "{1}"
else
    echo "{2}"
fi
'''


def is_process_running(pid: int) -> bool:
    cmd = CHECK_PID_CMD.format(pid, RUNNING, NOT_RUNNING)
    return subprocess.check_output(cmd, shell=True).decode('utf-8').strip() == RUNNING


@pytest.fixture()
def mock_p2p_class():
    P2P.LIBP2P_CMD = "sleep"


def test_daemon_killed_on_del(mock_p2p_class):
    p2p_daemon = P2P('10s')

    child_pid = p2p_daemon._child.pid
    assert is_process_running(child_pid)

    del p2p_daemon
    assert not is_process_running(child_pid)


def test_daemon_killed_on_exit(mock_p2p_class):
    with P2P('10s') as daemon:
        child_pid = daemon.pid
        assert is_process_running(child_pid)

    assert not is_process_running(child_pid)


def test_daemon_raises_on_faulty_args():
    with pytest.raises(RuntimeError):
        P2P(faulty='argument')
