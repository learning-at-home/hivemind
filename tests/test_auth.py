from datetime import datetime, timedelta
from typing import Tuple

import pytest

from hivemind.proto import dht_pb2
from hivemind.proto.auth_pb2 import AccessToken
from hivemind.utils.auth import AuthorizationError, AuthorizerBase
from hivemind.utils.crypto import RSAPrivateKey, RSAPublicKey


class MockAuthorizer(AuthorizerBase):
    _auth_server_private_key = None
    _auth_server_public_key = None

    async def get_access_token(self, local_public_key: RSAPublicKey) -> Tuple[AccessToken, RSAPublicKey]:
        if MockAuthorizer._auth_server_private_key is None:
            MockAuthorizer._auth_server_private_key = RSAPrivateKey()
            MockAuthorizer._auth_server_public_key = MockAuthorizer._auth_server_private_key.get_public_key()

        token = AccessToken(username='mock',
                            public_key=local_public_key.to_bytes(),
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
