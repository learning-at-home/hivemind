from typing import Sequence, Tuple, List, Dict

import torch

from hivemind.utils import Endpoint, CompressionType

TensorID = int
Part = Tuple[torch.Tensor, ...]


class TensorPartContainer:
    """
    Auxiliary data structure that is responsible for splitting tensors into parts and assembling them back together
    """

    @staticmethod
    def build_from_tensors(tensors: Sequence[torch.Tensor],
                           part_sizes: Tuple[int, ...],
                           endpoints: Sequence[Endpoint],
                           compression_type: Sequence[CompressionType] = None):
        assert len(endpoints) == len(part_sizes), "length of part_sizes mismatch with endpoints"

        # Sort tensors descending by size
        tensor_ids, tensors = zip(*sorted(enumerate(tensors), key=lambda idx_tensor: -idx_tensor[-1].numel()))
        # Sort peers ascending by part_size
        endpoints, part_sizes = zip(*sorted(zip(endpoints, part_sizes), key=lambda peer_size: peer_size[-1]))

        parts, part_ids = TensorPartContainer._split_into_parts(tensors, tensor_ids=tensor_ids, part_sizes=part_sizes)
        tensor_shapes = {idx: tensor.shape for idx, tensor in zip(tensor_ids, tensors)}

        return TensorPartContainer(parts, part_ids, tensor_shapes, endpoints, compression_type)

    def __init__(self,
                 parts: Sequence[Part],
                 part_ids: Sequence[List[TensorID]],
                 tensor_shapes: Sequence[torch.Size],
                 endpoints: Sequence[Endpoint],
                 compression_type: Sequence[CompressionType] = None):
        assert len(parts) == len(endpoints)
        assert len(parts) == len(part_ids)

        if compression_type is not None:
            assert len(compression_type) == len(tensor_shapes), \
                "length of compression type mismatch with number of tensors"
            self._compression_type = compression_type
        else:
            self._compression_type = None

        self._orig_shapes = tensor_shapes
        self.endpoints = tuple(endpoints)
        self._make_views(parts, part_ids, endpoints)

    def _make_views(self, pieces, piece_ids, endpoints):
        self._peer_tensor_id_view: Dict[Endpoint, Tuple[TensorID, ...]] = dict()
        self._piece_view: Dict[Tuple[Endpoint, TensorID], torch.Tensor] = dict()

        tensor_parts = []
        for part_pieces, part_ids, endpoint in zip(pieces, piece_ids, endpoints):
            self._peer_tensor_id_view[endpoint] = tuple(part_ids)

            part_keys = zip([endpoint] * len(part_ids), part_ids)
            tensor_parts.extend(zip(part_keys, part_pieces))

        self._piece_view = dict(tensor_parts)

    def get_piece(self, peer: Endpoint, tensor_id: int) -> torch.Tensor:
        return self._piece_view[(peer, tensor_id)]

    def get_part(self, peer: Endpoint) -> Part:
        """Return peer part of tensor pieces. pieces ordered by ids"""
        return tuple(self.get_piece(peer, idx) for idx in self._peer_tensor_id_view[peer])

    def get_part_with_ids(self, peer: Endpoint) -> Tuple[Tuple[TensorID, ...], Part]:
        """Return peer part of tensor pieces. pieces ordered by ids"""
        return self._peer_tensor_id_view[peer], self.get_part(peer)

    def set_piece(self, piece: torch.Tensor, peer: Endpoint, tensor_id: int):
        assert self._piece_view[(peer, tensor_id)].shape == piece.shape, "piece shape mismatch"
        assert self._piece_view[(peer, tensor_id)].dtype == piece.dtype, "piece dtype mismatch"
        self._piece_view[(peer, tensor_id)] = piece

    def set_part(self, peer_part: Part, peer: Endpoint, tensor_ids: List[TensorID] = None):
        """pieces must be ordered by tensor id"""
        if tensor_ids is None:
            tensor_ids = self._peer_tensor_id_view[peer]
        for tensor_id, tensor in zip(tensor_ids, peer_part):
            self.set_piece(tensor, peer, tensor_id)

    def get_shapes(self, peer: Endpoint) -> Tuple[torch.Size, ...]:
        return tuple(piece.shape for piece in self.get_part(peer))

    def get_dtypes(self, peer: Endpoint) -> Tuple[torch.dtype, ...]:
        return tuple(piece.dtype for piece in self.get_part(peer))

    def get_part_compression_type(self, peer: Endpoint) -> Tuple[CompressionType, ...]:
        if self._compression_type is None:
            return tuple(CompressionType.NONE)
        return tuple(self._compression_type[tensor_id] for tensor_id in self._peer_tensor_id_view[peer])

    @property
    def tensors(self) -> Sequence[torch.Tensor]:
        part_keys, pieces = zip(*self._piece_view.items())
        _, piece_ids = zip(*part_keys)

        tensor_ids = sorted(self._orig_shapes.keys())
        restored = self._restore_from_parts(piece_ids, pieces)
        restored = [restored.get(idx, torch.tensor([])) for idx in tensor_ids]
        shapes = [self._orig_shapes[idx] for idx in tensor_ids]

        return list(map(torch.Tensor.reshape, restored, shapes))

    @staticmethod
    def _split_into_parts(tensors: Sequence[torch.Tensor],
                          tensor_ids: Sequence[torch.Tensor],
                          part_sizes: Tuple[int]) -> Tuple[Sequence[Part], Sequence[List[int]]]:
        """ combines averaged_tensors into one tensor and splits them into equal pieces of size group_size """
        tensors = tuple(map(torch.Tensor.flatten, tensors))
        enumerated_tensors = zip(tensor_ids, tensors)
        peer_parts, peer_part_ids = [], []

        for part_size in part_sizes:
            part, part_ids = [], []
            accum_size = 0

            for tensor_id, tensor in enumerated_tensors:
                tensor_size = tensor.numel()

                if accum_size + tensor_size <= part_size:
                    part.append(tensor)
                    part_ids.append(tensor_id)
                    accum_size += tensor_size

                    if accum_size == part_size:
                        break
                else:
                    residue = part_size - accum_size
                    shards = tensor[:residue], tensor[residue:]

                    part.append(shards[0])
                    part_ids.append(tensor_id)
                    enumerated_tensors = iter([(tensor_id, shards[1])] + list(enumerated_tensors))
                    break

            part = tuple(part)
            peer_parts.append(part)
            peer_part_ids.append(part_ids)
        return peer_parts, peer_part_ids

    @staticmethod
    def _restore_from_parts(piece_ids: Sequence[TensorID],
                            pieces: Sequence[torch.Tensor]) -> Dict[TensorID, torch.Tensor]:
        """ restores the original tensor shapes from pieces obtained by split_into_pieces """
        restored, shapes = dict(), []
        for tensor_id in piece_ids:
            if tensor_id not in restored:
                tensor = torch.cat([tensor for tid, tensor in zip(piece_ids, pieces) if tid == tensor_id])
                restored[tensor_id] = tensor

        return restored
