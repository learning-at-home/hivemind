from typing import Dict, List, Tuple

from multiaddr import Multiaddr
from pydantic import BaseModel, StrictFloat, confloat, conint

from hivemind.dht.crypto import RSASignatureValidator
from hivemind.dht.schema import BytesWithPublicKey, SchemaValidator
from hivemind.dht.validation import RecordValidatorBase
from hivemind.utils.logging import get_logger


logger = get_logger(__name__)


class LocalMetrics(BaseModel):
    step: conint(ge=0, strict=True)
    samples_per_second: confloat(ge=0.0, strict=True)
    samples_accumulated: conint(ge=0, strict=True)
    loss: StrictFloat
    mini_steps: conint(ge=0, strict=True)


class MetricSchema(BaseModel):
    metrics: Dict[BytesWithPublicKey, LocalMetrics]


def make_validators(experiment_prefix: str) -> Tuple[List[RecordValidatorBase], bytes]:
    signature_validator = RSASignatureValidator()
    validators = [SchemaValidator(MetricSchema, prefix=experiment_prefix), signature_validator]
    return validators, signature_validator.local_public_key


class TextStyle:
    BOLD = '\033[1m'
    BLUE = '\033[34m'
    RESET = '\033[0m'


def log_visible_maddrs(visible_maddrs: List[Multiaddr], only_p2p: bool) -> None:
    if only_p2p:
        unique_addrs = {addr['p2p'] for addr in visible_maddrs}
        initial_peers_str = ' '.join(f'/p2p/{addr}' for addr in unique_addrs)
    else:
        initial_peers_str = ' '.join(str(addr) for addr in visible_maddrs)
    logger.info(f"Running a DHT peer. To connect other peers to this one, use "
                f"{TextStyle.BOLD}{TextStyle.BLUE}--initial_peers {initial_peers_str}{TextStyle.RESET}")
