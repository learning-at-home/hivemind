from typing import Any, Generic, TypeVar, Callable, Optional, Union

from hivemind.utils.mpfuture import MPFuture

T = TypeVar("T")

class _Empty(Generic[T]):

    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(_Empty, cls).__new__(cls, *args, **kwargs)
        return cls._instance



class LazyValue(Generic[T]):

    def __init__(self, value: T = _Empty(), init: Optional[Callable[..., T]] = None):
        assert value != _Empty() or init is not None, "One should provide either value or intializer"
        self.value = value
        self.init = init

    def get(self, *args, **kwargs) -> T:
        if self.value == _Empty():
            self.value = self.init(*args, **kwargs)

        return self.value

RT = TypeVar("RT")

class LazyFutureCaller(Generic[T, RT]):

    def __init__(self, future: MPFuture[T], callback: Optional[Callable[[T], RT]] = None):
        self._fut = future
        self._cb = callback

    def result(self) -> Union[T, RT]:
        result = self._fut.result()
        if self._cb is not None:
            return self._cb(result)
        return result
