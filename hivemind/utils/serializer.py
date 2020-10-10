""" A unified interface for several common serialization methods """
import pickle
from io import BytesIO
from typing import Dict, Any

import torch
import msgpack
from hivemind.utils.logging import get_logger

logger = get_logger(__name__)


class SerializerBase:
    @staticmethod
    def dumps(obj: object) -> bytes:
        raise NotImplementedError()

    @staticmethod
    def loads(buf: bytes) -> object:
        raise NotImplementedError()


class PickleSerializer(SerializerBase):
    @staticmethod
    def dumps(obj: object) -> bytes:
        return pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL)

    @staticmethod
    def loads(buf: bytes) -> object:
        return pickle.loads(buf)


class PytorchSerializer(SerializerBase):
    @staticmethod
    def dumps(obj: object) -> bytes:
        s = BytesIO()
        torch.save(obj, s, pickle_protocol=pickle.HIGHEST_PROTOCOL)
        return s.getvalue()

    @staticmethod
    def loads(buf: bytes) -> object:
        return torch.load(BytesIO(buf))


class MSGPackSerializer(SerializerBase):
    _ExtTypes: Dict[Any, int] = {}
    _ExtTypeCodes: Dict[int, Any] = {}

    @classmethod
    def ext_serializable(cls, type_code: int):
        assert isinstance(type_code, int), "Please specify a (unique) int type code"

        def wrap(wrapped_type: type):
            assert callable(getattr(wrapped_type, 'packb', None)) and callable(getattr(wrapped_type, 'unpackb', None)),\
                f"Every ext_type must have 2 methods: packb(self) -> bytes and classmethod unpackb(cls, bytes)"
            if type_code in cls._ExtTypeCodes:
                logger.warning(f"{cls.__name__}: type {type_code} is already registered, overwriting.")
            cls._ExtTypeCodes[type_code], cls._ExtTypes[wrapped_type] = wrapped_type, type_code
            return wrapped_type
        return wrap

    @classmethod
    def _encode_ext_types(cls, obj):
        type_code = cls._ExtTypes.get(type(obj))
        if type_code is not None:
            return msgpack.ExtType(type_code, obj.packb())
        return obj

    @classmethod
    def _decode_ext_types(cls, type_code: int, data: bytes):
        if type_code in cls._ExtTypeCodes:
            return cls._ExtTypeCodes[type_code].unpackb(data)
        logger.warning(f"Unknown ExtType code: {type_code}, leaving it as is.")
        return data

    @classmethod
    def dumps(cls, obj: object) -> bytes:
        return msgpack.dumps(obj, use_bin_type=True, default=cls._encode_ext_types, strict_types=True)

    @classmethod
    def loads(cls, buf: bytes) -> object:
        return msgpack.loads(buf, ext_hook=cls._decode_ext_types, raw=False)

