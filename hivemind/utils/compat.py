import multiprocessing as mp
import time
from typing import Any

from hivemind.utils.logging import get_logger

logger = get_logger(__name__)


def safe_recv(pipe: mp.connection.Connection) -> Any:
    # Needed for macOS, see https://github.com/urllib3/urllib3/issues/63#issuecomment-4609289

    while True:
        try:
            return pipe.recv()
        except Exception as e:
            if (isinstance(e, BlockingIOError) and str(e) == "[Errno 35] Resource temporarily unavailable") or (
                isinstance(e, EOFError) and str(e) == "Ran out of input"
            ):
                logger.warning(repr(e))
                time.sleep(0)
                continue
            raise
