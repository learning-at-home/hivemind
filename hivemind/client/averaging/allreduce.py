import asyncio
from typing import Sequence, Set, Dict, Tuple

import torch

from hivemind.utils import Endpoint, get_logger

# flavour types
GroupID = bytes
logger = get_logger(__name__)


class GroupAllReduce:
    """
    An internal class that runs butterfly AllReduce in a predefined group of averagers

    :param tensors: local tensors that should be averaged with groupmates
    :param endpoint: your endpoint, must be included in ordered_group_endpoints
    :param ordered_group_endpoints: group endpoints ordered s.t. i-th endpoint is responsible for averaging i-th part
    """
    def __init__(self, *, group_id: GroupID, tensors: Sequence[torch.Tensor], endpoint: Endpoint,
                 ordered_group_endpoints: Sequence[Endpoint]):
        assert endpoint in ordered_group_endpoints, "my endpoint is not a part of the group"
        self.group_id, self.endpoint, self.ordered_group_endpoints = group_id, endpoint, ordered_group_endpoints
        self.local_tensor_parts = dict(zip(ordered_group_endpoints, split_into_parts(tensors, self.group_size)))
        self.tensor_shapes = tuple(tensor.shape for tensor in tensors)

        self.accumulator = self.local_tensor_parts[self.endpoint].clone()  # sum inputs from peers to this tensor
        self.accumulated_from: Set[Endpoint] = {self.endpoint}  # peers that we have accumulated our part from
        self.averaged_part: asyncio.Future[torch.Tensor] = asyncio.Future()  # will be set to [accumulator / group size]
        self.averaged_tensor_parts: Dict[Endpoint, torch.Tensor] = {}  # averaged chunks from all peers will be put here
        self.averaged_tensors: asyncio.Future[Sequence[torch.Tensor]] = asyncio.Future()  # final result or exception

    def __repr__(self):
        return f"{self.__class__.__name__}({self.endpoint}, group_size={self.group_size})"

    def __await__(self):
        return self.averaged_tensors.__await__()

    @property
    def group_size(self):
        return len(self.ordered_group_endpoints)

    async def accumulate_part(self, source: Endpoint, remote_part: torch.Tensor) -> torch.Tensor:
        """ Add vector part to accumulator, wait for all other vectors to be added, then return the average part """
        assert not self.averaged_part.done(), f"already finished averaging part: {self.averaged_part}"
        assert not self.averaged_tensors.done(), f"already finished allreduce: {self.averaged_tensors}"
        assert source in self.local_tensor_parts, "unexpected source, not a part of current group"
        assert source not in self.accumulated_from, "duplicate source, already received that part"
        logger.debug(f"{self} - accumulating tensor part from {source}")

        self.accumulator.add_(remote_part)
        self.accumulated_from.add(source)

        assert len(self.accumulated_from) <= self.group_size
        if len(self.accumulated_from) == len(self.local_tensor_parts):
            self.averaged_part.set_result(self.accumulator.div_(len(self.accumulated_from)))
            self.register_averaged_part(self.endpoint, self.averaged_part.result())

        return await self.averaged_part

    def register_averaged_part(self, source: Endpoint, averaged_part: torch.Tensor):
        assert not self.averaged_tensors.done(), f"already finished allreduce: {self.averaged_tensors}"
        assert source in self.local_tensor_parts, "the provider of averaged part is not from my group"
        assert source not in self.averaged_tensor_parts, "already registered the average from this peer"
        assert averaged_part.shape == self.local_tensor_parts[source].shape, "averaged part shape mismatch"
        assert averaged_part.dtype == self.local_tensor_parts[source].dtype, "averaged part dtype mismatch"
        logger.debug(f"{self} - receiving averaged tensor part from {source}")
        self.averaged_tensor_parts[source] = averaged_part
        if len(self.averaged_tensor_parts) == len(self.local_tensor_parts):
            ordered_averaged_parts = [self.averaged_tensor_parts[endpoint] for endpoint in self.ordered_group_endpoints]
            self.averaged_tensors.set_result(restore_from_parts(ordered_averaged_parts, self.tensor_shapes))

    def cancel(self) -> bool:
        if not self.averaged_tensors.done():
            logger.debug(f"{self} - cancelled")
            self.averaged_tensors.cancel()
            if not self.averaged_part.done():
                self.averaged_part.cancel()
            return True
        else:
            logger.debug(f"{self} - failed to cancel, allreduce is already finished: {self.averaged_tensors}")
            return False

    def set_exception(self, exception: Exception) -> bool:
        if not self.averaged_tensors.done():
            logger.debug(f"{self} - {exception}")
            self.averaged_tensors.set_exception(exception)
            if not self.averaged_part.done():
                self.averaged_part.cancel()
            return True
        else:
            logger.debug(f"{self} - failed to set {exception}, allreduce already finished: {self.averaged_tensors}")
            return False


def split_into_parts(tensors: Sequence[torch.Tensor], group_size: int) -> Tuple[torch.Tensor]:
    """ combines averaged_tensors into one tensor and splits them into equal chunks of size group_size """
    flat_tensor = torch.cat(tuple(map(torch.Tensor.flatten, tensors)))
    chunk_slices = torch.linspace(start=0, end=len(flat_tensor), steps=group_size + 1, dtype=torch.int64)
    chunk_slices[-1] = len(flat_tensor)
    return tuple(torch.as_tensor(flat_tensor[chunk_slices[i]: chunk_slices[i + 1]]) for i in range(group_size))


def restore_from_parts(chunks: Sequence[torch.Tensor], shapes: Sequence[torch.Size]) -> Tuple[torch.Tensor, ...]:
    """ restores the original tensor shapes from chunks obtained by split_into_chunks """
    flat_tensor = torch.cat(list(chunks))
    result_sizes = tuple(map(torch.Size.numel, shapes))
    flat_original_tensors = torch.split_with_sizes(flat_tensor, result_sizes)
    return tuple(map(torch.Tensor.reshape, flat_original_tensors, shapes))


class AllreduceException(Exception):
    """ A special exception that is raised when allreduce can't continue normally (e.g. disbanded/bad request/etc) """
