import dataclasses
from abc import ABC, abstractmethod
from typing import Iterable


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

    @property
    def priority(self) -> int:
        """
        Defines the order of applying this validator with respect to other validators.

        The validators are applied:
          - In order of increasing priority for signing a record
          - In order of decreasing priority for validating and stripping a record
        """

        return 0

    def merge_with(self, other: "RecordValidatorBase") -> bool:
        """
        By default, all validators are applied sequentially (i.e. we require all validate() calls
        to return True for a record to be validated successfully).

        However, you may want to define another policy for combining your validator classes
        (e.g. for schema validators, we want to require only one validate() call to return True
        because each validator bears a part of the schema).

        This can be achieved with overriding merge_with(). It should:

          - Return True if it has successfully merged the `other` validator to `self`,
            so that `self` became a validator that combines the old `self` and `other` using
            the necessary policy. In this case, `other` should remain unchanged.

          - Return False if the merging has not happened. In this case, both `self` and `other`
            should remain unchanged. The DHT will try merging `other` to another validator or
            add it as a separate validator (to be applied sequentially).
        """

        return False


class CompositeValidator(RecordValidatorBase):
    def __init__(self, validators: Iterable[RecordValidatorBase] = ()):
        self._validators = []
        self.extend(validators)

    def extend(self, validators: Iterable[RecordValidatorBase]) -> None:
        for new_validator in validators:
            for existing_validator in self._validators:
                if existing_validator.merge_with(new_validator):
                    break
            else:
                self._validators.append(new_validator)
        self._validators.sort(key=lambda item: item.priority)

    def validate(self, record: DHTRecord) -> bool:
        for i, validator in enumerate(reversed(self._validators)):
            if not validator.validate(record):
                return False
            if i < len(self._validators) - 1:
                record = dataclasses.replace(record, value=validator.strip_value(record))
        return True

    def sign_value(self, record: DHTRecord) -> bytes:
        for validator in self._validators:
            record = dataclasses.replace(record, value=validator.sign_value(record))
        return record.value

    def strip_value(self, record: DHTRecord) -> bytes:
        for validator in reversed(self._validators):
            record = dataclasses.replace(record, value=validator.strip_value(record))
        return record.value
