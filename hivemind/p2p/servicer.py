import asyncio
import importlib
from dataclasses import dataclass
from typing import Any, Optional, Union

from hivemind.p2p.p2p_daemon import P2P
from hivemind.p2p.p2p_daemon_bindings.datastructures import PeerID


@dataclass
class RPCHandler:
    method_name: str
    handle_name: str
    request_type: type
    response_type: type


class StubBase:
    """
    Base class for P2P RPC stubs. The interface mimicks gRPC stubs.

    Servicer derives stub classes for particular services (e.g. DHT, averager, etc.) from StubBase,
    adding the necessary rpc_* methods. Calls to these methods are translated to calls to the remote peer.
    """

    def __init__(self, p2p: P2P, peer: PeerID):
        self._p2p = p2p
        self._peer = peer


class Servicer:
    """
    Base class for P2P RPC servicers (e.g. DHT, averager, MoE server). The interface mimicks gRPC servicers.

    - ``add_p2p_handlers(self, p2p)`` registers all rpc_* methods of the derived class as P2P unary handlers, allowing
      other peers to call them. It uses type annotations for the ``request`` parameter and the return value
      to infer protobufs the methods operate with.

    - ``get_stub(self, p2p, peer)`` creates a stub with all rpc_* methods. Calls to the stub methods are translated
      to calls to the remote peer.
    """

    def __init__(self):
        class_name = self.__class__.__name__

        self._rpc_handlers = []
        for method_name, method in self.__class__.__dict__.items():
            if method_name.startswith("rpc_") and callable(method):
                handle_name = f"{class_name}.{method_name}"

                hints = method.__annotations__
                try:
                    request_type = self._hint_to_type(hints["request"])
                    response_type = self._hint_to_type(hints["return"])
                except (KeyError, ValueError):
                    raise ValueError(
                        f"{handle_name} is expected to have type annotations like `dht_pb2.FindRequest` "
                        f"(a type from the hivemind.proto module) for the `request` parameter "
                        f"and the return value"
                    )

                self._rpc_handlers.append(RPCHandler(method_name, handle_name, request_type, response_type))

        self._stub_type = type(
            f"{class_name}Stub",
            (StubBase,),
            {handler.method_name: self._make_rpc_caller(handler) for handler in self._rpc_handlers},
        )

    @staticmethod
    def _make_rpc_caller(handler: RPCHandler):
        # This method will be added to a new Stub type (a subclass of StubBase)
        async def caller(
            self: StubBase, request: handler.request_type, timeout: Optional[float] = None
        ) -> handler.response_type:
            return await asyncio.wait_for(
                self._p2p.call_unary_handler(self._peer, handler.handle_name, request, handler.response_type),
                timeout=timeout,
            )

        caller.__name__ = handler.method_name
        return caller

    async def add_p2p_handlers(self, p2p: P2P, wrapper: Any = None) -> None:
        servicer = self if wrapper is None else wrapper
        for handler in self._rpc_handlers:
            await p2p.add_unary_handler(
                handler.handle_name,
                getattr(servicer, handler.method_name),
                handler.request_type,
            )

    def get_stub(self, p2p: P2P, peer: PeerID) -> StubBase:
        return self._stub_type(p2p, peer)

    @staticmethod
    def _hint_to_type(hint: Union[type, str]) -> type:
        if isinstance(hint, type):
            return hint

        module_name, proto_name = hint.split(".")
        module = importlib.import_module("hivemind.proto." + module_name)
        result = getattr(module, proto_name)
        if not isinstance(result, type):
            raise ValueError(f"`hivemind.proto.{hint}` is not a type")
        return result
