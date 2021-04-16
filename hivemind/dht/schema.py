import re
from typing import Any, Dict, Union

import cerberus

from hivemind.dht.protocol import DHTProtocol
from hivemind.dht.routing import DHTID, DHTKey
from hivemind.dht.validation import DHTRecord, RecordValidatorBase
from hivemind.utils import get_logger


logger = get_logger(__name__)


class SchemaValidator(RecordValidatorBase):
    """
    Restricts specified DHT keys to match a Cerberus schema.
    This allows to enforce types, min/max values, require a subkey to contain a public key, etc.
    """

    class _ExtendedCerberusValidator(cerberus.Validator):
        """
        By default, the Cerberus `regex` rule does not work with patterns for `bytes`.
        We extend it to support both `str` and `bytes`.
        The schema in the method's docstring describes the rule parameter.
        """

        def _validate_regex(self, pattern: Union[str, bytes], field: Any, value: Any):
            """ {'anyof': [{'type': 'string'}, {'type': 'binary'}]} """

            if type(pattern) != type(value):
                self._error(field,
                    f"Pattern type {type(pattern)} and value type {type(value)} don't match")
                return
            if re.fullmatch(pattern, value) is None:
                self._error(field, cerberus.errors.REGEX_MISMATCH)

    def __init__(self, schema: Dict[DHTKey, Any]):
        """
        :param schema: The Cerberus validation schema.
          See https://docs.python-cerberus.org/en/stable/validation-rules.html
        """

        schema = {DHTID.generate(source=key).to_bytes(): value
                  for key, value in schema.items()}

        self._validator = self._ExtendedCerberusValidator(schema)
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
