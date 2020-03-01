import pickle
from io import BytesIO

import joblib
import torch


class JoblibSerializer:

    @staticmethod
    def dumps(obj) -> bytes:
        s = BytesIO()
        joblib.dump(obj, s)
        return s.getvalue()

    @staticmethod
    def loads(buf: bytes):
        return joblib.load(BytesIO(buf))


class PickleSerializer:
    @staticmethod
    def dumps(obj) -> bytes:
        return pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL)

    @staticmethod
    def loads(buf: bytes):
        return pickle.loads(buf)


class PytorchSerializer:

    @staticmethod
    def dumps(obj) -> bytes:
        s = BytesIO()
        torch.save(obj, s, pickle_protocol=pickle.HIGHEST_PROTOCOL)
        return s.getvalue()

    @staticmethod
    def loads(buf: bytes):
        return torch.load(BytesIO(buf))
