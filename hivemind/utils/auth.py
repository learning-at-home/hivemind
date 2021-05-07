import functools
import secrets
import threading
import time
from abc import ABC, abstractmethod
from enum import Enum
from datetime import datetime, timedelta
from typing import Optional, Tuple

import grpc

from hivemind.proto.auth_pb2 import AccessToken, RequestAuthInfo, ResponseAuthInfo
from hivemind.utils.crypto import RSAPrivateKey, RSAPublicKey
from hivemind.utils.timed_storage import TimedStorage, get_dht_time


class AuthorizedRequest:
    auth: RequestAuthInfo


class AuthorizedResponse:
    auth: ResponseAuthInfo


class AuthorizerBase(ABC):
    @abstractmethod
    async def sign_request(self, request: AuthorizedRequest, service_public_key: Optional[RSAPublicKey]) -> None:
        ...

    @abstractmethod
    async def validate_request(self, request: AuthorizedRequest) -> None:
        ...

    @abstractmethod
    async def sign_response(self, response: AuthorizedResponse, request: AuthorizedRequest) -> None:
        ...

    @abstractmethod
    async def validate_response(self, response: AuthorizedResponse, request: AuthorizedRequest) -> None:
        ...


class SignedTokenAuthorizer(AuthorizerBase):
    def __init__(self, local_private_key: Optional[RSAPrivateKey]=None):
        if local_private_key is None:
            local_private_key = RSAPrivateKey.process_wide()
        self._local_private_key = local_private_key
        self._local_public_key = local_private_key.get_public_key()

        self._local_access_token = None
        self._auth_server_public_key = None

        self._recent_nonces = TimedStorage()
        self._recent_nonces_lock = threading.RLock()

    @abstractmethod
    async def get_access_token(self) -> Tuple[AccessToken, RSAPublicKey]:
        ...

    _WORST_TIMEDELTA = timedelta(minutes=1)

    async def _update_access_token(self) -> None:
        worst_recv_time = datetime.utcnow() + self._WORST_TIMEDELTA
        if (self._local_access_token is not None and
                worst_recv_time < datetime.fromisoformat(self._local_access_token.expiration_time)):
            return

        self._local_access_token, self._auth_server_public_key = await self.get_access_token()

    def _verify_access_token(self, access_token: AccessToken) -> bool:
        data = self._access_token_to_bytes(access_token)
        return (
            self._auth_server_public_key.verify(data, access_token.signature) and
            datetime.fromisoformat(access_token.expiration_time) >= datetime.utcnow()
        )

    @staticmethod
    def _access_token_to_bytes(access_token: AccessToken) -> bytes:
        return b' '.join([access_token.username.encode(),
                          access_token.public_key,
                          access_token.expiration_time.encode()])

    @property
    def local_public_key(self) -> RSAPublicKey:
        return self._local_public_key

    async def sign_request(self, request: AuthorizedRequest, service_public_key: Optional[RSAPublicKey]) -> None:
        await self._update_access_token()
        auth = request.auth

        auth.client_access_token.CopyFrom(self._local_access_token)

        if service_public_key is not None:
            auth.service_public_key = service_public_key.to_bytes()
        auth.time = get_dht_time()
        auth.nonce = secrets.token_bytes(8)

        assert auth.signature == b''
        auth.signature = self._local_private_key.sign(request.SerializeToString())

    async def validate_request(self, request: AuthorizedRequest) -> None:
        await self._update_access_token()
        auth = request.auth

        if not self._verify_access_token(auth.client_access_token):
            raise AuthorizationError('Client failed to prove that it (still) has access to the network')

        client_public_key = RSAPublicKey.from_bytes(auth.client_access_token.public_key)
        signature = auth.signature
        auth.signature = b''
        if not client_public_key.verify(request.SerializeToString(), signature):
            raise AuthorizationError('Request has invalid signature')

        if auth.service_public_key and auth.service_public_key != self._local_public_key.to_bytes():
            raise AuthorizationError('Request is generated for a peer with another public key')

        with self._recent_nonces_lock:
            with self._recent_nonces.freeze():
                current_time = get_dht_time()
                if abs(auth.time - current_time) > self._WORST_TIMEDELTA.total_seconds():
                    raise AuthorizationError('Clocks are not synchronized or a previous request is replayed again')
                if auth.nonce in self._recent_nonces:
                    raise AuthorizationError('Previous request is replayed again')

            self._recent_nonces.store(auth.nonce, None, current_time + self._WORST_TIMEDELTA.total_seconds() * 3)

    async def sign_response(self, response: AuthorizedResponse, request: AuthorizedRequest) -> None:
        await self._update_access_token()
        auth = response.auth

        auth.service_access_token.CopyFrom(self._local_access_token)
        auth.nonce = request.auth.nonce

        assert auth.signature == b''
        auth.signature = self._local_private_key.sign(response.SerializeToString())

    async def validate_response(self, response: AuthorizedResponse, request: AuthorizedRequest) -> None:
        await self._update_access_token()
        auth = response.auth

        if not self._verify_access_token(auth.service_access_token):
            raise AuthorizationError('Service failed to prove that it (still) has access to the network')

        service_public_key = RSAPublicKey.from_bytes(auth.service_access_token.public_key)
        signature = auth.signature
        auth.signature = b''
        if not service_public_key.verify(response.SerializeToString(), signature):
            raise AuthorizationError('Response has invalid signature')

        if request.auth.service_public_key and request.auth.service_public_key != auth.service_access_token.public_key:
            raise AuthorizationError('Response is generated by a service different from the one we requested')

        if auth.nonce != request.auth.nonce:
            raise AuthorizationError('Response is generated for another request')


class AuthRole(Enum):
    CLIENT = 0
    SERVICER = 1


class AuthRPCWrapper:
    def __init__(self, stub, role: AuthRole,
                 authorizer: Optional[AuthorizerBase], service_public_key: Optional[RSAPublicKey]=None):
        self._stub = stub
        self._role = role
        self._authorizer = authorizer
        self._service_public_key = service_public_key

    def __getattribute__(self, name: str):
        if not name.startswith('rpc_'):
            return object.__getattribute__(self, name)

        method = getattr(self._stub, name)

        @functools.wraps(method)
        async def wrapped_rpc(request: AuthorizedRequest, *args, **kwargs):
            if self._authorizer is not None:
                if self._role == AuthRole.CLIENT:
                    await self._authorizer.sign_request(request, self._service_public_key)
                elif self._role == AuthRole.SERVICER:
                    await self._authorizer.validate_request(request)

            response = await method(request, *args, **kwargs)

            if self._authorizer is not None:
                if self._role == AuthRole.SERVICER:
                    await self._authorizer.sign_response(response, request)
                elif self._role == AuthRole.CLIENT:
                    await self._authorizer.validate_response(response, request)

            return response

        return wrapped_rpc


class AuthorizationError(Exception):
    pass
