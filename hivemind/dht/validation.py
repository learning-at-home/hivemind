import dataclasses
from abc import ABC, abstractmethod
from typing import Dict


@dataclasses.dataclass(init=True, repr=True, frozen=True)
class DHTRecord:
    key: bytes
    subkey: bytes
    value: bytes
    expiration_time: float


class RecordValidatorBase(ABC):
    """
    Record validators are a generic mechanism for checking the DHT records including:
      - Enforcing a data schema (e.g. checking content types)
      - Enforcing security requirements (e.g. allowing only the owner to update the record)
    """

    @abstractmethod
    def validate(self, record: DHTRecord) -> bool:
        """
        Should return whether the `record` is valid.
        The valid records should have been extended with sign_value().

        validate() is called when another DHT peer:
          - Asks us to store the record
          - Returns the record by our request
        """

        pass

    def sign_value(self, record: DHTRecord) -> bytes:
        """
        Should return `record.value` extended with the record's signature.

        Note: there's no need to overwrite this method if a validator doesn't use a signature.

        sign_value() is called after the application asks the DHT to store the record.
        """

        return record.value

    def strip_value(self, record: DHTRecord) -> bytes:
        """
        Should return `record.value` stripped of the record's signature.
        strip_value() is only called if validate() was successful.

        Note: there's no need to overwrite this method if a validator doesn't use a signature.

        strip_value() is called before the DHT returns the record by the application's request.
        """

        return record.value


class CompositeValidator(RecordValidatorBase):
    def __init__(self, validators: Dict[str, RecordValidatorBase]=None):
        self._validators = validators if validators is not None else {}

    def extend(self, validators: Dict[str, RecordValidatorBase]) -> None:
        for key, val in validators.items():
            self._validators.setdefault(key, val)

    def validate(self, record: DHTRecord) -> bool:
        for i, (_, val) in enumerate(sorted(self._validators.items(), reverse=True)):
            if not val.validate(record):
                return False
            if i < len(self._validators) - 1:
                record = dataclasses.replace(record, value=val.strip_value(record))
        return True

    def sign_value(self, record: DHTRecord) -> bytes:
        for _, val in sorted(self._validators.items()):
            record = dataclasses.replace(record, value=val.sign_value(record))
        return record.value

    def strip_value(self, record: DHTRecord) -> bytes:
        for _, val in sorted(self._validators.items(), reverse=True):
            record = dataclasses.replace(record, value=val.strip_value(record))
        return record.value
