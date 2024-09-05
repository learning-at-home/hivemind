from datetime import datetime, timedelta
from typing import Optional
import pytest
from hivemind.proto import dht_pb2
from hivemind.proto.auth_pb2 import AccessToken
from hivemind.utils.auth import AuthRole, AuthRPCWrapper, TokenAuthorizerBase
from hivemind.utils.crypto import RSAPrivateKey, RSAPublicKey  # updated to ed25519 in entire repo
from hivemind.utils.logging import get_logger

logger = get_logger(__name__)


class MockAuthorizer(TokenAuthorizerBase):
    _authority_private_key = None
    _authority_public_key = None

    def __init__(self, local_private_key: Optional[RSAPrivateKey], username: str = "mock"):
        super().__init__(local_private_key)
        self._username = username
        self._authority_public_key = None

    async def get_token(self) -> AccessToken:
        if MockAuthorizer._authority_private_key is None:
            MockAuthorizer._authority_private_key = RSAPrivateKey()  # updated to ed25519 in entire repo

        self._authority_public_key = MockAuthorizer._authority_private_key.get_public_key()

        token = AccessToken(
            username=self._username,
            public_key=self.local_public_key.to_bytes(),
            expiration_time=str(datetime.utcnow() + timedelta(minutes=1)),
        )
        token.signature = MockAuthorizer._authority_private_key.sign(self._token_to_bytes(token))
        return token

    def is_token_valid(self, access_token: AccessToken) -> bool:
        data = self._token_to_bytes(access_token)
        if not self._authority_public_key.verify(data, access_token.signature):
            logger.exception("Access token has invalid signature")
            return False

        try:
            expiration_time = datetime.fromisoformat(access_token.expiration_time)
        except ValueError:
            logger.exception(
                f"datetime.fromisoformat() failed to parse expiration time: {access_token.expiration_time}"
            )
            return False
        if expiration_time.tzinfo is not None:
            logger.exception(f"Expected to have no timezone for expiration time: {access_token.expiration_time}")
            return False
        if expiration_time < datetime.utcnow():
            logger.exception("Access token has expired")
            return False

        return True

    _MAX_LATENCY = timedelta(minutes=1)

    def does_token_need_refreshing(self, access_token: AccessToken) -> bool:
        expiration_time = datetime.fromisoformat(access_token.expiration_time)
        return expiration_time < datetime.utcnow() + self._MAX_LATENCY

    @staticmethod
    def _token_to_bytes(access_token: AccessToken) -> bytes:
        return f"{access_token.username} {access_token.public_key} {access_token.expiration_time}".encode()


@pytest.mark.asyncio
async def test_valid_request_and_response():
    client_authorizer = MockAuthorizer(RSAPrivateKey())  # updated to ed25519 in entire repo
    service_authorizer = MockAuthorizer(RSAPrivateKey())  # updated to ed25519 in entire repo
    request = dht_pb2.PingRequest()
    request.peer.node_id = b"ping"
    await client_authorizer.sign_request(request, service_authorizer.local_public_key)
    assert await service_authorizer.validate_request(request)
    response = dht_pb2.PingResponse()
    response.peer.node_id = b"pong"
    await service_authorizer.sign_response(response, request)
    assert await client_authorizer.validate_response(response, request)
