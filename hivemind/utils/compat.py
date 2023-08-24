import multiprocessing as mp
import time
from typing import Any


def safe_recv(pipe: mp.connection.Connection) -> Any:
    # Needed for macOS, see https://github.com/urllib3/urllib3/issues/63#issuecomment-4609289

    while True:
        try:
            return pipe.recv()
        except BlockingIOError as e:
            if str(e) == "[Errno 35] Resource temporarily unavailable":
                time.sleep(0)
                continue
            raise
