import pytest
import psutil


@pytest.fixture(autouse=True, scope='session')
def cleanup_children():
    yield
    for child in psutil.Process().children(recursive=True):
        child.terminate()
