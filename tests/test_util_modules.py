import asyncio
import torch

import pytest
import hivemind

from concurrent.futures import CancelledError


def test_mpfuture_result():
    f1, f2 = hivemind.MPFuture.make_pair()
    f1.set_result(321)
    assert f2.result() == 321
    assert f1.result() == 321

    for future in [f1, f2]:
        with pytest.raises(RuntimeError):
            future.set_result(123)
        with pytest.raises(RuntimeError):
            future.set_exception(ValueError())
        assert future.cancel() is False
        assert future.done() and not future.running() and not future.cancelled()

    f1, f2 = hivemind.MPFuture.make_pair()
    with pytest.raises(TimeoutError):
        f1.result(timeout=1e-3)

    f2.set_result(['abacaba', 123])
    assert f1.result() == ['abacaba', 123]


def test_mpfuture_exception():
    f1, f2 = hivemind.MPFuture.make_pair()
    with pytest.raises(TimeoutError):
        f1.exception(timeout=1e-3)

    f2.set_exception(NotImplementedError())

    for future in [f1, f2]:
        assert isinstance(future.exception(), NotImplementedError)
        with pytest.raises(NotImplementedError):
            future.result()
        assert future.cancel() is False
        assert future.done() and not future.running() and not future.cancelled()


def test_mpfuture_cancel():
    f1, f2 = hivemind.MPFuture.make_pair()
    assert not f2.cancelled()
    f1.cancel()
    for future in [f1, f2]:
        with pytest.raises(CancelledError):
            future.result()
        with pytest.raises(CancelledError):
            future.exception()
        with pytest.raises(RuntimeError):
            future.set_result(123)
        with pytest.raises(RuntimeError):
            future.set_exception(NotImplementedError)
        assert future.cancelled() and future.done() and not future.running()


def test_mpfuture_status():
    f1, f2 = hivemind.MPFuture.make_pair()
    assert f1.set_running_or_notify_cancel() is True
    for future in [f1, f2]:
        assert future.running() and not future.done() and not future.cancelled()
        with pytest.raises(RuntimeError):
            future.set_running_or_notify_cancel()
    f2.cancel()
    for future in [f1, f2]:
        assert not future.running() and future.done() and future.cancelled()
        assert future.set_running_or_notify_cancel() is False

    f1, f2 = hivemind.MPFuture.make_pair()
    f1.cancel()
    for future in [f1, f2]:
        assert future.set_running_or_notify_cancel() is False


def test_await_mpfuture():
    async def _run():
        # await result
        f1, f2 = hivemind.MPFuture.make_pair()

        async def wait_and_assign():
            assert f2.set_running_or_notify_cancel() is True
            await asyncio.sleep(0.1)
            f2.set_result((123, 'ololo'))

        asyncio.create_task(wait_and_assign())
        for future in [f1, f2]:
            res = await future
            assert res == (123, 'ololo')

        # await cancel
        f1, f2 = hivemind.MPFuture.make_pair()

        async def wait_and_cancel():
            await asyncio.sleep(0.1)
            f1.cancel()

        asyncio.create_task(wait_and_cancel())
        for future in [f1, f2]:
            with pytest.raises(CancelledError):
                await future

        # await exception
        f1, f2 = hivemind.MPFuture.make_pair()

        async def wait_and_raise():
            await asyncio.sleep(0.1)
            f1.set_exception(SystemError())

        asyncio.create_task(wait_and_raise())
        for future in [f1, f2]:
            with pytest.raises(SystemError):
                await future

    asyncio.new_event_loop().run_until_complete(_run())


def test_vector_compression():
    test_X = torch.rand
