from typing import Dict, List, Tuple

from pydantic import BaseModel, StrictFloat, confloat, conint

import hivemind
from hivemind.dht.crypto import RSASignatureValidator
from hivemind.dht.schema import BytesWithPublicKey, SchemaValidator
from hivemind.dht.validation import RecordValidatorBase


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
    validators = [SchemaValidator(MetricSchema, prefix=experiment_prefix),
                  signature_validator]
    return validators, signature_validator.local_public_key


class TextStyle:
    BOLD = '\033[1m'
    BLUE = '\033[34m'
    RESET = '\033[0m'


def format_visible_maddrs(dht: hivemind.DHT) -> None:
    initial_peers_str = ' '.join(str(addr) for addr in dht.get_visible_maddrs())
    return f"{TextStyle.BOLD}{TextStyle.BLUE}--initial_peers {initial_peers_str}{TextStyle.RESET}"
