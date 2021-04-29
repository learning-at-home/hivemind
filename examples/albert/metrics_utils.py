from typing import Dict, List, Tuple

from hivemind.dht.crypto import RSASignatureValidator
from hivemind.dht.schema import SchemaValidator
from hivemind.dht.validation import RecordValidatorBase
from pydantic import BaseModel, StrictFloat, confloat, conint


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
