from __future__ import annotations

from enum import Enum, auto

import pytest

import hivemind
from hivemind.averaging.averager import *
from hivemind.averaging.group_info import GroupInfo
from hivemind.averaging.load_balancing import load_balance_peers
from hivemind.averaging.matchmaking import MatchmakingException
from hivemind.proto import averaging_pb2
from hivemind.utils.asyncio import aenumerate, as_aiter, azip, enter_asynchronously
from hivemind.utils.logging import get_logger

logger = get_logger(__name__)


class Fault(Enum):
    NONE = auto()
    FAIL_BEFORE = auto()
    FAIL_SENDING = auto()
    SLOW_SENDING = auto()
    FAIL_REDUCING = auto()
    SLOW_REDUCING = auto()
    CANCEL = auto()


class FaultyAverager(hivemind.DecentralizedAverager):
    def __init__(self, *args, fault: Fault = Fault.NONE, **kwargs):
        self.fault = fault
        super().__init__(*args, **kwargs)

    async def _aggregate_with_group(self, group_info: GroupInfo, min_vector_size: int, **kwargs) -> GatheredData:
        """Run All-Reduce in a given group and update tensors in place, return gathered metadata"""
        try:
            bandwidths, mode_ids, user_gathered_bytes = zip(*map(self.serializer.loads, group_info.gathered))
            user_gathered = dict(zip(group_info.peer_ids, map(self.serializer.loads, user_gathered_bytes)))
            modes = tuple(map(AveragingMode, mode_ids))
            download_bandwidths = [
                thr if mode != AveragingMode.CLIENT else 0.0 for thr, mode in zip(bandwidths, modes)
            ]
            peer_fractions = await asyncio.get_event_loop().run_in_executor(
                None, load_balance_peers, self.total_size, download_bandwidths, min_vector_size
            )

            if self.fault == Fault.FAIL_BEFORE:
                raise Exception("Oops, I failed!")

            async with enter_asynchronously(self.get_tensors()) as local_tensors:
                allreduce = FaultyAllReduceRunner(
                    p2p=self._p2p,
                    servicer_type=type(self),
                    prefix=self.prefix,
                    group_id=group_info.group_id,
                    tensors=local_tensors,
                    ordered_peer_ids=group_info.peer_ids,
                    peer_fractions=peer_fractions,
                    modes=modes,
                    fault=self.fault,
                    **kwargs,
                )

                self._running_groups[group_info.group_id].set_result(allreduce)
                # TODO maybe this can be extracted into a method that checks if register_... context is active.

                if modes[group_info.peer_ids.index(self.peer_id)] != AveragingMode.AUX:
                    iter_results = allreduce.run()
                    async for tensor, update in azip(as_aiter(*local_tensors), iter_results):
                        # all-reduce is performed asynchronously while iterating
                        tensor.add_(update, alpha=self._averaging_alpha)
                    self._state_updated.set()

                else:
                    async for _ in allreduce:  # trigger all-reduce by iterating
                        raise ValueError("aux peers should not receive averaged tensors")

                return user_gathered
        except BaseException as e:
            logger.exception(e)
            raise MatchmakingException(f"Unable to run All-Reduce: {e}")


class FaultyAllReduceRunner(AllReduceRunner):
    def __init__(self, *args, fault: Fault, **kwargs):
        self.fault = fault
        super().__init__(*args, **kwargs)

    async def rpc_aggregate_part(self, stream, context) -> AsyncIterator[averaging_pb2.AveragingData]:
        if self.fault in (Fault.FAIL_REDUCING, Fault.SLOW_REDUCING):
            async for i, message in aenumerate(super().rpc_aggregate_part(stream, context)):
                yield message
                if i == 2:
                    if self.fault == Fault.FAIL_SENDING:
                        yield averaging_pb2.AveragingData(code=averaging_pb2.INTERNAL_ERROR)
                        break
                    else:
                        await asyncio.sleep(10)

        elif self.fault == Fault.CANCEL:
            yield averaging_pb2.AveragingData(code=averaging_pb2.CANCELLED)
        else:
            async for message in super().rpc_aggregate_part(stream, context):
                yield message

    async def _generate_input_for_peer(self, peer_index: int) -> AsyncIterator[averaging_pb2.AveragingData]:
        parts_aiter = self.tensor_part_container.iterate_input_parts_for(peer_index)

        first_part = await anext(parts_aiter)
        yield averaging_pb2.AveragingData(
            code=averaging_pb2.PART_FOR_AVERAGING,
            group_id=self.group_id,
            tensor_part=first_part,
            weight=self.weight,
        )
        if self.fault in (Fault.FAIL_SENDING, Fault.SLOW_SENDING):
            last_reducer_index = self.group_size - 1 - (self.tensor_part_container.num_parts_by_peer[-1] == 0)
            if peer_index == last_reducer_index:
                if self.fault == Fault.FAIL_SENDING:
                    raise Exception("Oops, I failed!")
                else:
                    await asyncio.sleep(10)
        async for part in parts_aiter:
            yield averaging_pb2.AveragingData(tensor_part=part, weight=self.weight)


@pytest.mark.forked
@pytest.mark.parametrize(
    "fault0, fault1",
    [
        (Fault.NONE, Fault.FAIL_BEFORE),
        (Fault.FAIL_BEFORE, Fault.FAIL_BEFORE),
        (Fault.SLOW_SENDING, Fault.FAIL_SENDING),
        (Fault.FAIL_SENDING, Fault.FAIL_BEFORE),
        (Fault.SLOW_REDUCING, Fault.FAIL_SENDING),
        (Fault.FAIL_REDUCING, Fault.FAIL_REDUCING),
        (Fault.NONE, Fault.CANCEL),
    ],
)
def test_fault_tolerance(fault0: Fault, fault1: Fault):
    def _make_tensors():
        return [torch.rand(16, 1024), -torch.rand(3, 8192), 2 * torch.randn(4, 4, 4), torch.randn(1024, 1024)]

    dht = hivemind.DHT(start=True)

    averagers = []
    for i in range(5):
        averager = FaultyAverager(
            _make_tensors(),
            hivemind.DHT(initial_peers=dht.get_visible_maddrs(), start=True),
            prefix="test",
            request_timeout=0.3,
            min_matchmaking_time=1.0,
            next_chunk_timeout=0.5,
            allreduce_timeout=5,
            part_size_bytes=2**16,
            client_mode=(i == 1),
            start=True,
            fault=fault0 if i == 0 else fault1 if i == 1 else Fault.NONE,
        )
        averagers.append(averager)

    ref_numerators = [0, 0, 0, 0]
    ref_denominator = 0

    for averager in averagers:
        if averager.fault not in (Fault.FAIL_BEFORE, Fault.CANCEL):
            with averager.get_tensors() as tensors:
                for i, tensor in enumerate(tensors):
                    ref_numerators[i] = ref_numerators[i] + tensor.clone()
                ref_denominator += 1

    ref_tensors = [ref_numerator / ref_denominator for ref_numerator in ref_numerators]
    flat_ref = torch.cat(list(map(torch.flatten, ref_tensors)))

    flat_local_tensors = []
    for averager in averagers:
        with averager.get_tensors() as tensors:
            flat_local_tensors.append(torch.cat(list(map(torch.flatten, tensors))))

    futures = [averager.step(timeout=5, wait=False, allow_retries=False) for averager in averagers]
    for i, averager in enumerate(averagers):
        if averager.fault == Fault.CANCEL:
            futures[i].cancel()

    for future in futures[2:]:
        assert future.result()

    for averager, prev_local_tensors in zip(averagers[2:], flat_local_tensors[2:]):
        with averager.get_tensors() as tensors:
            flat_tensors = torch.cat(list(map(torch.flatten, tensors)))

        diff_with_reference = abs(flat_ref - flat_tensors)

        if all(fault == (Fault.FAIL_SENDING, Fault.SLOW_SENDING) for fault in (fault0, fault1)):
            assert fault0 != Fault.FAIL_REDUCING and fault1 != Fault.FAIL_REDUCING
            assert diff_with_reference[: len(diff_with_reference) // 2].max() < 1e-5
        elif all(fault in (Fault.FAIL_REDUCING, Fault.SLOW_REDUCING) for fault in (fault0, fault1)):
            diff_to_reference = abs(flat_ref - flat_tensors)
            diff_to_local = abs(prev_local_tensors - flat_tensors)
            assert (diff_with_reference < 1e-5).numpy().mean() > 0.5
            assert torch.all(torch.minimum(diff_to_reference, diff_to_local) < 1e-5).item()
        elif any(fault == Fault.CANCEL for fault in (fault0, fault1)):
            pass  # late cancel may result in an arbitrary mix of averaging results with and without the cancelled peer
        elif fault0 == Fault.NONE:  # only peer1 in client mode may have failed
            assert diff_with_reference.max() < 1e-5
        else:
            assert (diff_with_reference < 1e-5).numpy().mean() > 0.5

    for averager in averagers:
        averager.shutdown()
