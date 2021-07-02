import pytest
import psutil


@pytest.fixture(autouse=True, scope='session')
def cleanup_children():
    """ reset shared memory manager for isolation, terminate any leftover processes after the test is finished """
    yield
    for child in psutil.Process().children(recursive=True):
        child.terminate()
