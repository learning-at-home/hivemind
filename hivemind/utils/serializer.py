""" A unified interface for several common serialization methods """
from abc import ABC, abstractmethod
from typing import Any, Dict

import msgpack

from hivemind.utils.logging import get_logger

logger = get_logger(__name__)


class SerializerBase(ABC):
    @staticmethod
    @abstractmethod
    def dumps(obj: object) -> bytes:
        pass

    @staticmethod
    @abstractmethod
    def loads(buf: bytes) -> object:
        pass


class MSGPackSerializer(SerializerBase):
    _ext_types: Dict[Any, int] = {}
    _ext_type_codes: Dict[int, Any] = {}
    _TUPLE_EXT_TYPE_CODE = 0x40

    @classmethod
    def ext_serializable(cls, type_code: int):
        assert isinstance(type_code, int), "Please specify a (unique) int type code"

        def wrap(wrapped_type: type):
            assert callable(getattr(wrapped_type, "packb", None)) and callable(
                getattr(wrapped_type, "unpackb", None)
            ), f"Every ext_type must have 2 methods: packb(self) -> bytes and classmethod unpackb(cls, bytes)"
            if type_code in cls._ext_type_codes:
                logger.warning(f"{cls.__name__}: type {type_code} is already registered, overwriting")
            cls._ext_type_codes[type_code], cls._ext_types[wrapped_type] = wrapped_type, type_code
            return wrapped_type

        return wrap

    @classmethod
    def _encode_ext_types(cls, obj):
        type_code = cls._ext_types.get(type(obj))
        if type_code is not None:
            return msgpack.ExtType(type_code, obj.packb())
        elif isinstance(obj, tuple):
            # Tuples need to be handled separately to ensure that
            # 1. tuple serialization works and 2. tuples serialized not as lists
            data = msgpack.packb(list(obj), strict_types=True, use_bin_type=True, default=cls._encode_ext_types)
            return msgpack.ExtType(cls._TUPLE_EXT_TYPE_CODE, data)
        return obj

    @classmethod
    def _decode_ext_types(cls, type_code: int, data: bytes):
        if type_code in cls._ext_type_codes:
            return cls._ext_type_codes[type_code].unpackb(data)
        elif type_code == cls._TUPLE_EXT_TYPE_CODE:
            return tuple(msgpack.unpackb(data, ext_hook=cls._decode_ext_types, raw=False))

        logger.warning(f"Unknown ExtType code: {type_code}, leaving it as is")
        return data

    @classmethod
    def dumps(cls, obj: object) -> bytes:
        return msgpack.dumps(obj, use_bin_type=True, default=cls._encode_ext_types, strict_types=True)

    @classmethod
    def loads(cls, buf: bytes) -> object:
        return msgpack.loads(buf, ext_hook=cls._decode_ext_types, raw=False)
