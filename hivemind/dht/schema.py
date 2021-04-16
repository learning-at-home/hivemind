from typing import Any, Dict

import cerberus

from hivemind.dht.protocol import DHTProtocol
from hivemind.dht.routing import DHTID, DHTKey
from hivemind.dht.validation import DHTRecord, RecordValidatorBase
from hivemind.utils import get_logger


logger = get_logger(__name__)


class SchemaValidator(RecordValidatorBase):
    def __init__(self, schema: Dict[DHTKey, Any]):
        schema = {DHTID.generate(source=key).to_bytes(): value
                  for key, value in schema.items()}

        self._validator = cerberus.Validator(schema)
        self._validator.allow_unknown = True

    def validate(self, record: DHTRecord) -> bool:
        value = DHTProtocol.serializer.loads(record.value)
        if record.subkey not in DHTProtocol.RESERVED_SUBKEYS:
            subkey = DHTProtocol.serializer.loads(record.subkey)
            document = {record.key: {subkey: value}}
        else:
            document = {record.key: value}

        is_valid = self._validator.validate(document)
        if not is_valid:
            logger.debug(f"Record {record} doesn't match the schema: {self._validator.errors}")
            return False
        return True
