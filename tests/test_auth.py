from datetime import datetime, timedelta
from typing import Optional, Tuple

import pytest

from hivemind.proto import dht_pb2
from hivemind.proto.auth_pb2 import AccessToken
from hivemind.utils.auth import AuthRPCWrapper, AuthRole, AuthorizationError, SignedTokenAuthorizer
from hivemind.utils.crypto import RSAPrivateKey, RSAPublicKey


class MockAuthorizer(SignedTokenAuthorizer):
    _auth_server_private_key = None
    _auth_server_public_key = None

    def __init__(self, local_private_key: Optional[RSAPrivateKey], username: str='mock'):
        super().__init__(local_private_key)
        self._username = username

    async def get_access_token(self) -> Tuple[AccessToken, RSAPublicKey]:
        if MockAuthorizer._auth_server_private_key is None:
            MockAuthorizer._auth_server_private_key = RSAPrivateKey()
            MockAuthorizer._auth_server_public_key = MockAuthorizer._auth_server_private_key.get_public_key()

        token = AccessToken(username=self._username,
                            public_key=self.local_public_key.to_bytes(),
                            expiration_time=str(datetime.utcnow() + timedelta(minutes=1)))
        token.signature = MockAuthorizer._auth_server_private_key.sign(self._access_token_to_bytes(token))
        return token, MockAuthorizer._auth_server_public_key


@pytest.mark.asyncio
async def test_valid_request_and_response():
    client_authorizer = MockAuthorizer(RSAPrivateKey())
    service_authorizer = MockAuthorizer(RSAPrivateKey())

    request = dht_pb2.PingRequest()
    request.peer.endpoint = '127.0.0.1:7777'
    await client_authorizer.sign_request(request, service_authorizer.local_public_key)
    await service_authorizer.validate_request(request)

    response = dht_pb2.PingResponse()
    response.sender_endpoint = '127.0.0.1:31337'
    await service_authorizer.sign_response(response, request)
    await client_authorizer.validate_response(response, request)


@pytest.mark.asyncio
async def test_invalid_access_token():
    client_authorizer = MockAuthorizer(RSAPrivateKey())
    service_authorizer = MockAuthorizer(RSAPrivateKey())

    request = dht_pb2.PingRequest()
    request.peer.endpoint = '127.0.0.1:7777'
    await client_authorizer.sign_request(request, service_authorizer.local_public_key)

    # Break the access token signature
    request.auth.client_access_token.signature = b'broken'

    with pytest.raises(AuthorizationError):
        await service_authorizer.validate_request(request)

    response = dht_pb2.PingResponse()
    response.sender_endpoint = '127.0.0.1:31337'
    await service_authorizer.sign_response(response, request)

    # Break the access token signature
    response.auth.service_access_token.signature = b'broken'

    with pytest.raises(AuthorizationError):
        await client_authorizer.validate_response(response, request)


@pytest.mark.asyncio
async def test_invalid_signatures():
    client_authorizer = MockAuthorizer(RSAPrivateKey())
    service_authorizer = MockAuthorizer(RSAPrivateKey())

    request = dht_pb2.PingRequest()
    request.peer.endpoint = '127.0.0.1:7777'
    await client_authorizer.sign_request(request, service_authorizer.local_public_key)

    # A man-in-the-middle attacker changes the request content
    request.peer.endpoint = '127.0.0.2:7777'

    with pytest.raises(AuthorizationError):
        await service_authorizer.validate_request(request)

    response = dht_pb2.PingResponse()
    response.sender_endpoint = '127.0.0.1:31337'
    await service_authorizer.sign_response(response, request)

    # A man-in-the-middle attacker changes the response content
    response.sender_endpoint = '127.0.0.2:31337'

    with pytest.raises(AuthorizationError):
        await client_authorizer.validate_response(response, request)


@pytest.mark.asyncio
async def test_auth_rpc_wrapper():
    class Servicer:
        async def rpc_increment(self, request: dht_pb2.PingRequest) -> dht_pb2.PingResponse:
            assert request.peer.endpoint == '127.0.0.1:1111'
            assert request.auth.client_access_token.username == 'alice'

            response = dht_pb2.PingResponse()
            response.sender_endpoint = '127.0.0.1:2222'
            return response

    class Client:
        def __init__(self, servicer: Servicer):
            self._servicer = servicer

        async def rpc_increment(self, request: dht_pb2.PingRequest) -> dht_pb2.PingResponse:
            return await self._servicer.rpc_increment(request)

    servicer = AuthRPCWrapper(Servicer(), AuthRole.SERVICER, MockAuthorizer(RSAPrivateKey(), 'bob'))
    client = AuthRPCWrapper(Client(servicer), AuthRole.CLIENT, MockAuthorizer(RSAPrivateKey(), 'alice'))

    request = dht_pb2.PingRequest()
    request.peer.endpoint = '127.0.0.1:1111'

    response = await client.rpc_increment(request)

    assert response.sender_endpoint == '127.0.0.1:2222'
    assert response.auth.service_access_token.username == 'bob'
