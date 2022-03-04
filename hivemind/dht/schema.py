import re
from typing import Any, Dict, Optional, Type

import pydantic

from hivemind.dht.crypto import RSASignatureValidator
from hivemind.dht.protocol import DHTProtocol
from hivemind.dht.routing import DHTID
from hivemind.dht.validation import DHTRecord, RecordValidatorBase
from hivemind.utils import get_logger

logger = get_logger(__name__)


class SchemaValidator(RecordValidatorBase):
    """
    Restricts specified DHT keys to match a Pydantic schema.
    This allows to enforce types, min/max values, require a subkey to contain a public key, etc.
    """

    def __init__(self, schema: Type[pydantic.BaseModel], allow_extra_keys: bool = True, prefix: Optional[str] = None):
        """
        :param schema: The Pydantic model (a subclass of pydantic.BaseModel).

            You must always use strict types for the number fields
            (e.g. ``StrictInt`` instead of ``int``,
            ``confloat(strict=True, ge=0.0)`` instead of ``confloat(ge=0.0)``, etc.).
            See the validate() docstring for details.

            The model will be patched to adjust it for the schema validation.

        :param allow_extra_keys: Whether to allow keys that are not defined in the schema.

            If a SchemaValidator is merged with another SchemaValidator, this option applies to
            keys that are not defined in each of the schemas.

        :param prefix: (optional) Add ``prefix + '_'`` to the names of all schema fields.
        """

        self._patch_schema(schema)
        self._schemas = [schema]

        self._key_id_to_field_name = {}
        for field in schema.__fields__.values():
            raw_key = f"{prefix}_{field.name}" if prefix is not None else field.name
            self._key_id_to_field_name[DHTID.generate(source=raw_key).to_bytes()] = field.name
        self._allow_extra_keys = allow_extra_keys

    @staticmethod
    def _patch_schema(schema: pydantic.BaseModel):
        # We set required=False because the validate() interface provides only one key at a time
        for field in schema.__fields__.values():
            field.required = False

        schema.Config.extra = pydantic.Extra.forbid

    def validate(self, record: DHTRecord) -> bool:
        """
        Validates ``record`` in two steps:

        1. Create a Pydantic model and ensure that no exceptions are thrown.

        2. Ensure that Pydantic has not made any type conversions [1]_ while creating the model.
           To do this, we check that the value of the model field is equal
           (in terms of == operator) to the source value.

           This works for the iterable default types like str, list, and dict
           (they are equal only if the types match) but does not work for numbers
           (they have a special case allowing ``3.0 == 3`` to be true). [2]_

           Because of that, you must always use strict types [3]_ for the number fields
           (e.g. to avoid ``3.0`` to be validated successfully for the ``field: int``).

           .. [1] https://pydantic-docs.helpmanual.io/usage/models/#data-conversion
           .. [2] https://stackoverflow.com/a/52557261
           .. [3] https://pydantic-docs.helpmanual.io/usage/types/#strict-types
        """

        if record.key not in self._key_id_to_field_name:
            if not self._allow_extra_keys:
                logger.debug(
                    f"Record {record} has a key ID that is not defined in any of the "
                    f"schemas (therefore, the raw key is unknown)"
                )
            return self._allow_extra_keys

        try:
            record = self._deserialize_record(record)
        except ValueError as e:
            logger.debug(e)
            return False
        [field_name] = list(record.keys())

        n_outside_schema = 0
        validation_errors = []
        for schema in self._schemas:
            try:
                parsed_record = schema.parse_obj(record)
            except pydantic.ValidationError as e:
                if not self._is_failed_due_to_extra_field(e):
                    validation_errors.append(e)
                continue

            parsed_value = parsed_record.dict(by_alias=True)[field_name]
            if parsed_value != record[field_name]:
                validation_errors.append(
                    ValueError(
                        f"The record {record} needed type conversions to match "
                        f"the schema: {parsed_value}. Type conversions are not allowed"
                    )
                )
            else:
                return True

        logger.debug(f"Record {record} doesn't match any of the schemas: {validation_errors}")
        return False

    def _deserialize_record(self, record: DHTRecord) -> Dict[str, Any]:
        field_name = self._key_id_to_field_name[record.key]
        deserialized_value = DHTProtocol.serializer.loads(record.value)
        if record.subkey not in DHTProtocol.RESERVED_SUBKEYS:
            deserialized_subkey = DHTProtocol.serializer.loads(record.subkey)
            return {field_name: {deserialized_subkey: deserialized_value}}
        else:
            if isinstance(deserialized_value, dict):
                raise ValueError(
                    f"Record {record} contains an improperly serialized dictionary (you must use "
                    f"a DictionaryDHTValue of serialized values instead of a `dict` subclass)"
                )
            return {field_name: deserialized_value}

    @staticmethod
    def _is_failed_due_to_extra_field(exc: pydantic.ValidationError):
        inner_errors = exc.errors()
        return (
            len(inner_errors) == 1
            and inner_errors[0]["type"] == "value_error.extra"
            and len(inner_errors[0]["loc"]) == 1  # Require the extra field to be on the top level
        )

    def merge_with(self, other: RecordValidatorBase) -> bool:
        if not isinstance(other, SchemaValidator):
            return False

        self._schemas.extend(other._schemas)
        self._key_id_to_field_name.update(other._key_id_to_field_name)
        self._allow_extra_keys = self._allow_extra_keys or other._allow_extra_keys
        return True

    def __setstate__(self, state):
        self.__dict__.update(state)

        # If unpickling happens in another process, the previous model modifications may be lost
        for schema in self._schemas:
            self._patch_schema(schema)


def conbytes(*, regex: bytes = None, **kwargs) -> Type[pydantic.BaseModel]:
    """
    Extend pydantic.conbytes() to support ``regex`` constraints (like pydantic.constr() does).
    """

    compiled_regex = re.compile(regex) if regex is not None else None

    class ConstrainedBytesWithRegex(pydantic.conbytes(**kwargs)):
        @classmethod
        def __get_validators__(cls):
            yield from super().__get_validators__()
            yield cls.match_regex

        @classmethod
        def match_regex(cls, value: bytes) -> bytes:
            if compiled_regex is not None and compiled_regex.match(value) is None:
                raise ValueError(f"Value `{value}` doesn't match regex `{regex}`")
            return value

    return ConstrainedBytesWithRegex


BytesWithPublicKey = conbytes(regex=b".*" + RSASignatureValidator.PUBLIC_KEY_REGEX + b".*")
