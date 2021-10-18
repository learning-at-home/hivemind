import asyncio
import inspect
from dataclasses import dataclass
from typing import Any, AsyncIterator, List, Optional, Tuple, Type, get_type_hints

from hivemind.p2p.p2p_daemon import P2P
from hivemind.p2p.p2p_daemon_bindings.datastructures import PeerID


@dataclass
class RPCHandler:
    method_name: str
    request_type: type
    response_type: type
    stream_input: bool
    stream_output: bool


class StubBase:
    """
    Base class for P2P RPC stubs. The interface mimicks gRPC stubs.

    Servicer derives stub classes for particular services (e.g. DHT, averager, etc.) from StubBase,
    adding the necessary rpc_* methods. Calls to these methods are translated to calls to the remote peer.
    """

    def __init__(self, p2p: P2P, peer: PeerID, namespace: Optional[str]):
        self._p2p = p2p
        self._peer = peer
        self._namespace = namespace


class ServicerBase:
    """
    Base class for P2P RPC servicers (e.g. DHT, averager, MoE server). The interface mimicks gRPC servicers.

    - ``add_p2p_handlers(self, p2p)`` registers all rpc_* methods of the derived class as P2P handlers, allowing
      other peers to call them. It uses type annotations for the ``request`` parameter and the return value
      to infer protobufs the methods operate with.

    - ``get_stub(self, p2p, peer)`` creates a stub with all rpc_* methods. Calls to the stub methods are translated
      to calls to the remote peer.
    """

    _rpc_handlers: Optional[List[RPCHandler]] = None
    _stub_type: Optional[Type[StubBase]] = None

    @classmethod
    def _collect_rpc_handlers(cls) -> None:
        if cls._rpc_handlers is not None:
            return

        cls._rpc_handlers = []
        for method_name, method in inspect.getmembers(cls, predicate=inspect.isfunction):
            if method_name.startswith("rpc_"):
                spec = inspect.getfullargspec(method)
                if len(spec.args) < 3:
                    raise ValueError(
                        f"{method_name} is expected to at least three positional arguments "
                        f"(self, request: TInputProtobuf | AsyncIterator[TInputProtobuf], context: P2PContext)"
                    )
                request_arg = spec.args[1]
                hints = get_type_hints(method)
                try:
                    request_type = hints[request_arg]
                    response_type = hints["return"]
                except KeyError:
                    raise ValueError(
                        f"{method_name} is expected to have type annotations "
                        f"like `dht_pb2.FindRequest` or `AsyncIterator[dht_pb2.FindRequest]` "
                        f"for the `{request_arg}` parameter and the return value"
                    )
                request_type, stream_input = cls._strip_iterator_hint(request_type)
                response_type, stream_output = cls._strip_iterator_hint(response_type)

                cls._rpc_handlers.append(
                    RPCHandler(method_name, request_type, response_type, stream_input, stream_output)
                )

        cls._stub_type = type(
            f"{cls.__name__}Stub",
            (StubBase,),
            {handler.method_name: cls._make_rpc_caller(handler) for handler in cls._rpc_handlers},
        )

    @classmethod
    def _make_rpc_caller(cls, handler: RPCHandler):
        input_type = AsyncIterator[handler.request_type] if handler.stream_input else handler.request_type
        output_type = AsyncIterator[handler.response_type] if handler.stream_output else handler.response_type

        # This method will be added to a new Stub type (a subclass of StubBase)
        async def caller(self: StubBase, input: input_type, timeout: Optional[float] = None) -> output_type:
            handle_name = cls._get_handle_name(self._namespace, handler.method_name)
            if not handler.stream_output:
                return await asyncio.wait_for(
                    self._p2p.call_protobuf_handler(self._peer, handle_name, input, handler.response_type),
                    timeout=timeout,
                )

            if timeout is not None:
                raise ValueError("Timeouts for handlers returning streams are not supported")
            return await self._p2p.iterate_protobuf_handler(self._peer, handle_name, input, handler.response_type)

        caller.__name__ = handler.method_name
        return caller

    async def add_p2p_handlers(self, p2p: P2P, wrapper: Any = None, *, namespace: Optional[str] = None) -> None:
        self._collect_rpc_handlers()

        servicer = self if wrapper is None else wrapper

        await asyncio.gather(
            *[
                p2p.add_protobuf_handler(
                    self._get_handle_name(namespace, handler.method_name),
                    getattr(servicer, handler.method_name),
                    handler.request_type,
                    stream_input=handler.stream_input,
                    stream_output=handler.stream_output,
                )
                for handler in self._rpc_handlers
            ]
        )

    @classmethod
    def get_stub(cls, p2p: P2P, peer: PeerID, *, namespace: Optional[str] = None) -> StubBase:
        cls._collect_rpc_handlers()
        return cls._stub_type(p2p, peer, namespace)

    @classmethod
    def _get_handle_name(cls, namespace: Optional[str], method_name: str) -> str:
        handle_name = f"{cls.__name__}.{method_name}"
        if namespace is not None:
            handle_name = f"{namespace}::{handle_name}"
        return handle_name

    @staticmethod
    def _strip_iterator_hint(hint: type) -> Tuple[type, bool]:
        if hasattr(hint, "_name") and hint._name in ("AsyncIterator", "AsyncIterable"):
            return hint.__args__[0], True

        return hint, False
