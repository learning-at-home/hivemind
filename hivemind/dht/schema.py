import binascii
import re
from typing import Type

import pydantic

from hivemind.dht.protocol import DHTProtocol
from hivemind.dht.routing import DHTID, DHTKey
from hivemind.dht.validation import DHTRecord, RecordValidatorBase
from hivemind.utils import get_logger


logger = get_logger(__name__)


class SchemaValidator(RecordValidatorBase):
    """
    Restricts specified DHT keys to match a Pydantic schema.
    This allows to enforce types, min/max values, require a subkey to contain a public key, etc.
    """

    def __init__(self, schema: pydantic.BaseModel):
        """
        :param schema: The Pydantic model (a subclass of pydantic.BaseModel).

            You must always use strict types for the number fields
            (e.g. ``StrictInt`` instead of ``int``,
            ``confloat(strict=True, ge=0.0)`` instead of ``confloat(ge=0.0)``, etc.).
            See the validate() docstring for details.
        """

        self._alias_to_name = {}
        for field in schema.__fields__.values():
            field.alias = self._key_id_to_str(DHTID.generate(source=field.name.encode()).to_bytes())
            self._alias_to_name[field.alias] = field.name

            # Because validate() interface provides one key at a time
            field.required = False

        schema.Config.extra = pydantic.Extra.allow
        self._schema = schema

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

        key_alias = self._key_id_to_str(record.key)
        deserialized_value = DHTProtocol.serializer.loads(record.value)
        if record.subkey not in DHTProtocol.RESERVED_SUBKEYS:
            deserialized_subkey = DHTProtocol.serializer.loads(record.subkey)
            deserialized_record = {key_alias: {deserialized_subkey: deserialized_value}}
        else:
            if isinstance(deserialized_value, dict):
                logger.warning(
                    f'Record {record} contains an improperly serialized dictionary (you must use '
                    f'a DictionaryDHTValue of serialized values instead of a `dict` subclass)')
                return False
            deserialized_record = {key_alias: deserialized_value}

        try:
            parsed_record = self._schema.parse_obj(deserialized_record)
        except pydantic.ValidationError as e:
            readable_record = {self._alias_to_name.get(key_alias, key_alias):
                               deserialized_record[key_alias]}
            logger.warning(f"Record {readable_record} doesn't match the schema: {e}")
            return False

        parsed_value = parsed_record.dict(by_alias=True)[key_alias]
        if parsed_value != deserialized_record[key_alias]:
            logger.warning(
                f"Value {deserialized_record[key_alias]} needed type conversions to match "
                f" the schema: {parsed_value}. The type conversions are not allowed")
            return False
        return True

    @staticmethod
    def _key_id_to_str(key_id: bytes) -> str:
        """
        Represent ``key_id`` as a ``str`` since Pydantic does not support field aliases
        of type ``bytes``.
        """

        return binascii.hexlify(key_id).decode()


def conbytes(*, regex: bytes=None, **kwargs) -> Type[pydantic.BaseModel]:
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
