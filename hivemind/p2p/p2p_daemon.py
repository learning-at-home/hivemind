import subprocess
import typing as tp


class P2P(object):
    """
    Forks a child process and executes p2pd command with given arguments.
    Sends SIGKILL to the child in destructor and on exit from contextmanager.
    """

    LIBP2P_CMD = 'p2pd'

    def __init__(self, *args, **kwargs):
        self._child = subprocess.Popen(args=self._make_process_args(args, kwargs))
        try:
            stdout, stderr = self._child.communicate(timeout=0.2)
        except subprocess.TimeoutExpired:
            pass
        else:
            raise RuntimeError(f'p2p daemon exited with stderr: {stderr}')

    def __enter__(self):
        return self._child

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._kill_child()

    def __del__(self):
        self._kill_child()

    def _kill_child(self):
        if self._child.poll() is None:
            self._child.kill()
            self._child.wait()

    def _make_process_args(self, args: tp.Tuple[tp.Any],
                           kwargs: tp.Dict[str, tp.Any]) -> tp.List[str]:
        proc_args = [self.LIBP2P_CMD]
        proc_args.extend(
            str(entry) for entry in args
        )
        proc_args.extend(
            f'-{key}={str(value)}' for key, value in kwargs.items()
        )
        return proc_args
