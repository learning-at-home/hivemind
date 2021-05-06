import functools
import secrets
import threading
import time
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Tuple

import grpc

from hivemind.proto.auth_pb2 import AccessToken, RequestAuthInfo, ResponseAuthInfo
from hivemind.utils.crypto import RSAPrivateKey, RSAPublicKey
from hivemind.utils.timed_storage import TimedStorage, get_dht_time


class AuthorizedRequest:
    auth: RequestAuthInfo


class AuthorizedResponse:
    auth: ResponseAuthInfo


class AuthorizerBase(ABC):
    _NONCE_STORAGE_DURATION = 60

    def __init__(self, local_private_key: RSAPrivateKey):
        if local_private_key is None:
            local_private_key = RSAPrivateKey.process_wide()
        self._local_private_key = local_private_key
        self._local_public_key = local_private_key.get_public_key()

        self._local_access_token = None
        self._auth_server_public_key = None

        self._recent_nonces = TimedStorage()
        self._recent_nonces_lock = threading.RLock()

    async def authorize(self, *args, **kwargs) -> None:
        if (self._local_access_token is not None and
                not self._is_access_token_expired(self._local_access_token)):
            return

        self._local_access_token, self._auth_server_public_key = await self.get_access_token(
            self._local_public_key, *args, **kwargs)

    @abstractmethod
    async def get_access_token(
            self, local_public_key: RSAPublicKey, *args, **kwargs) -> Tuple[AccessToken, RSAPublicKey]:
        ...

    def _validate_access_token(self, access_token: AccessToken) -> bool:
        data = self._access_token_to_bytes(access_token)
        return (
            self._auth_server_public_key.verify(data, access_token.signature) and
            not self._is_access_token_expired(access_token)
        )

    @staticmethod
    def _access_token_to_bytes(access_token: AccessToken) -> bytes:
        return b' '.join([access_token.username.encode(),
                          access_token.public_key,
                          access_token.expiration_time.encode()])

    def _is_access_token_expired(self, access_token: AccessToken) -> bool:
        return datetime.fromisoformat(access_token.expiration_time) < datetime.utcnow()

    @property
    def local_public_key(self) -> RSAPublicKey:
        return self._local_public_key

    async def sign_request(self, request: AuthorizedRequest, service_public_key: RSAPublicKey) -> None:
        await self.authorize()
        auth = request.auth

        auth.client_access_token.CopyFrom(self._local_access_token)

        auth.service_public_key = service_public_key.to_bytes()
        auth.time = get_dht_time()
        auth.nonce = secrets.token_bytes(8)

        assert auth.signature == b''
        auth.signature = self._local_private_key.sign(request.SerializeToString())

    async def validate_request(self, request: AuthorizedRequest) -> None:
        await self.authorize()
        auth = request.auth

        if not self._validate_access_token(auth.client_access_token):
            raise AuthorizationError('Client failed to prove that it (still) has access to the network')

        client_public_key = RSAPublicKey.from_bytes(auth.client_access_token.public_key)
        signature = auth.signature
        auth.signature = b''
        if not client_public_key.verify(request.SerializeToString(), signature):
            raise AuthorizationError('Request has invalid signature')

        if auth.service_public_key != self._local_public_key.to_bytes():
            raise AuthorizationError('Request is generated for a peer with another public key')

        nonce_expiration = auth.time + self._NONCE_STORAGE_DURATION  # TODO: Check it
        with self._recent_nonces_lock:
            with self._recent_nonces.freeze():
                current_time = get_dht_time()
                if nonce_expiration < current_time:
                    raise AuthorizationError('Request is too old to trust it')
                if auth.nonce in self._recent_nonces:
                    raise AuthorizationError('An attacker replays a previous request again')

            self._recent_nonces.store(auth.nonce, None, nonce_expiration)

    async def sign_response(self, response: AuthorizedResponse, request: AuthorizedRequest) -> None:
        await self.authorize()
        auth = response.auth

        auth.service_access_token.CopyFrom(self._local_access_token)
        auth.nonce = request.auth.nonce

        assert auth.signature == b''
        auth.signature = self._local_private_key.sign(response.SerializeToString())

    async def validate_response(self, response: AuthorizedResponse, request: AuthorizedRequest) -> bool:
        await self.authorize()
        auth = response.auth

        if not self._validate_access_token(auth.service_access_token):
            raise AuthorizationError('Service failed to prove that it (still) has access to the network')

        service_public_key = RSAPublicKey.from_bytes(auth.service_access_token.public_key)
        signature = auth.signature
        auth.signature = b''
        if not service_public_key.verify(response.SerializeToString(), signature):
            raise AuthorizationError('Response has invalid signature')

        if auth.service_access_token.public_key != request.auth.service_public_key:
            raise AuthorizationError('Response is generated by a service different from the one we requested')

        if auth.nonce != request.auth.nonce:
            raise AuthorizationError('Response is generated for another request')


def rpc_requires_auth(method):
    @functools.wraps(method)
    async def patched_rpc(self, request: AuthorizedRequest, context: grpc.ServicerContext):
        if self.authorizer is not None:
            await self.authorizer.validate_request(request)

        response = await method(self, request, context)

        if self.authorizer is not None:
            await self.authorizer.sign_response(response, request)
        return response

    return patched_rpc


class AuthorizationError(Exception):
    pass
