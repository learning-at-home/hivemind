""" A unified interface for several common serialization methods """
import pickle
from io import BytesIO

import torch
import umsgpack

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
    @staticmethod
    def dumps(obj: object) -> bytes:
        return umsgpack.dumps(obj, use_bin_type=False, strict_types=True)

    @staticmethod
    def loads(buf: bytes) -> object:
        return umsgpack.loads(buf, raw=False)
