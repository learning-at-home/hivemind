import asyncio
import dataclasses
import importlib
from functools import partial
from typing import Any, Callable, Optional, Union

from hivemind.p2p.p2p_daemon import P2P, P2PContext
from hivemind.p2p.p2p_daemon_bindings.datastructures import PeerID


@dataclasses.dataclass
class RPCHandler:
    name: str
    method: Callable[[Any, P2PContext], Any]
    request_type: type
    response_type: type


class StubBase:
    def __init__(self, p2p: P2P, peer: PeerID):
        self.p2p = p2p
        self.peer = peer


class Servicer:
    def __init__(self):
        class_name = self.__class__.__name__

        self._rpc_handlers = []
        rpc_callers = {}
        for method_name, method in self.__class__.__dict__.items():
            if method_name.startswith('rpc_') and callable(method):
                handle_name = f'{class_name}.{method_name}'

                hints = method.__annotations__
                try:
                    request_type = self._hint_to_type(hints['request'])
                    response_type = self._hint_to_type(hints['return'])
                except (KeyError, ValueError):
                    raise ValueError(f'{handle_name} is expected to have type annotations like `dht_pb2.FindRequest` '
                                     f'(a type from the hivemind.proto module) for the `request` parameter '
                                     f'and the return value')

                self._rpc_handlers.append(RPCHandler(handle_name, partial(method, self), request_type, response_type))
                rpc_callers[method_name] = self._make_rpc_caller(method_name, handle_name, request_type, response_type)

        self._stub_type = type(f'{class_name}Stub', (StubBase,), rpc_callers)

    @staticmethod
    def _make_rpc_caller(method_name: str, handle_name: str, request_type: type, response_type: type):
        async def caller(stub: StubBase, request: request_type, timeout: Optional[float] = None) -> response_type:
            return await asyncio.wait_for(
                stub.p2p.call_unary_handler(stub.peer, handle_name, request, response_type),
                timeout=timeout)

        caller.__name__ = method_name
        return caller

    async def add_p2p_handlers(self, p2p: P2P) -> None:
        for handler in self._rpc_handlers:
            await p2p.add_unary_handler(handler.name, handler.method, handler.request_type, handler.response_type)

    def get_stub(self, p2p: P2P, peer: PeerID) -> StubBase:
        return self._stub_type(p2p, peer)

    @staticmethod
    def _hint_to_type(hint: Union[type, str]) -> type:
        if isinstance(hint, type):
            return hint

        module_name, proto_name = hint.split('.')
        module = importlib.import_module('hivemind.proto.' + module_name)
        result = getattr(module, proto_name)
        if not isinstance(result, type):
            raise ValueError(f'`hivemind.proto.{hint}` is not a type')
        return result
