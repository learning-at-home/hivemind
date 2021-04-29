from hivemind.dht.schema import SchemaValidator
from pydantic import BaseModel, StrictFloat, UUID4, confloat, conint


class LocalMetrics(BaseModel):
    step: conint(ge=0, strict=True)
    samples_per_second: confloat(ge=0.0, strict=True)
    samples_accumulated: conint(ge=0, strict=True)
    loss: StrictFloat
    mini_steps: conint(ge=0, strict=True)


class MetricSchema(BaseModel):
    metrics: Dict[UUID4, LocalMetrics]


def make_schema_validator(experiment_prefix: str) -> SchemaValidator:
    return SchemaValidator(MetricSchema, prefix=experiment_prefix)
