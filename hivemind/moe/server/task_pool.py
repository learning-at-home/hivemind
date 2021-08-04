"""
Task pool is responsible for receiving tasks and grouping them together for processing (but not processing itself)
"""
import ctypes
import multiprocessing as mp
import os
import threading
import time
from abc import ABCMeta, abstractmethod
from collections import namedtuple
from concurrent.futures import Future
from queue import Empty
from typing import Any, Dict, Generator, List, Tuple

import torch

from hivemind.utils import get_logger
from hivemind.utils.mpfuture import InvalidStateError, MPFuture

logger = get_logger(__name__)
Task = namedtuple("Task", ("future", "args"))


class TaskPoolBase(mp.context.ForkProcess, metaclass=ABCMeta):
    """A pool that accepts tasks and forms batches for parallel processing, interacts with Runtime"""

    def __init__(self, process_func: callable, daemon=True, **kwargs):
        super().__init__(daemon=daemon, **kwargs)
        self.process_func = process_func
        self._priority = mp.Value(ctypes.c_double, 1.0)  # higher priority = the more urgent to process this pool

    @abstractmethod
    def run(self):
        pass

    @abstractmethod
    def submit_task(self, *args: torch.Tensor) -> Future:
        pass

    @abstractmethod
    def iterate_minibatches(self, *args, **kwargs) -> Generator[List[Task], None, None]:
        pass

    @property
    def priority(self):
        return self._priority.value

    @priority.setter
    def priority(self, value):
        self._priority.value = float(value)

    @property
    @abstractmethod
    def empty(self):
        pass


class TaskPool(TaskPoolBase):
    """
    Request aggregator that accepts processing requests, groups them into batches, waits for Runtime
    to process these batches and dispatches results back to request sources. Operates as a background process.

    :param process_func: function to be applied to every formed batch; called by Runtime
        Note that process_func should accept only positional args (Tensors) and return a flat tuple of Tensors
    :param max_batch_size: process at most this many inputs in a batch (task contains have one or several inputs)
    :param name: pool name
    :param min_batch_size: process at least this many inputs in a batch, otherwise wait for more
    :param timeout: wait for a subsequent task for at most this many seconds
    :param pool_size: store at most this many unprocessed tasks in a queue
    :param prefetch_batches: prepare up to this many *batches* in background for faster off-loading to runtime
    :param start: if True, start automatically at the end of __init__
    """

    def __init__(
        self,
        process_func: callable,
        max_batch_size: int,
        name: str,
        min_batch_size=1,
        timeout=None,
        pool_size=None,
        prefetch_batches=1,
        daemon=True,
        start=False,
    ):
        super().__init__(process_func, daemon=daemon, name=name)
        self.min_batch_size, self.max_batch_size, self.timeout = min_batch_size, max_batch_size, timeout
        self.prefetch_batches = prefetch_batches

        # interaction with ConnectionHandlers
        self.tasks = mp.Queue(maxsize=pool_size or 0)
        self.undispatched_task_timestamps = mp.SimpleQueue()

        # interaction with Runtime
        self.batch_receiver, self.batch_sender = mp.Pipe(duplex=False)  # send/recv arrays that contain batch inputs
        self.outputs_receiver, self.outputs_sender = mp.Pipe(duplex=False)  # send/recv arrays that contain outputs

        if start:
            self.start()

    def submit_task(self, *args: torch.Tensor) -> Future:
        """Add task to this pool's queue, return Future for its output"""
        task = Task(MPFuture(), args)
        if self.get_task_size(task) > self.max_batch_size:
            exc = ValueError(f"Task size greater than max_batch_size ({self.max_batch_size}), it can't be processed")
            task.future.set_exception(exc)
        else:
            self.tasks.put(task)
            self.undispatched_task_timestamps.put(time.time())
        return task.future

    def iterate_minibatches(self, *args, **kwargs):
        """Form minibatches by grouping one or more tasks together up to self.max_batch_size"""
        batch = []
        total_size = 0

        while True:
            if total_size >= self.min_batch_size and self.tasks.empty():
                yield batch
                batch = []
                total_size = 0
            try:
                logger.debug(f"{self.name} getting next task")
                task = self.tasks.get(timeout=self.timeout)
            except Empty:
                logger.warning(f"Timeout reached but batch doesn't contain >={self.min_batch_size} elements yet")
                continue

            task_size = self.get_task_size(task)

            if total_size + task_size > self.max_batch_size:
                yield batch
                batch = []
                total_size = 0

            try:
                if task.future.set_running_or_notify_cancel():
                    batch.append(task)
                    total_size += task_size
            except InvalidStateError as e:
                logger.debug(f"Failed to add task to batch: {task.future} raised {e}")

    def run(self, *args, **kwargs):
        torch.set_num_threads(1)
        logger.info(f"{self.name} starting, pid={os.getpid()}")
        pending_batches = {}  # Dict[batch uuid, List[MPFuture]] for each batch currently in runtime

        output_thread = threading.Thread(
            target=self._pool_output_loop, args=[pending_batches], name=f"{self.name}_output", daemon=True
        )

        try:
            output_thread.start()
            self._pool_input_loop(pending_batches, *args, **kwargs)
        except KeyboardInterrupt:
            logger.debug("Caught KeyboardInterrupt, shutting down")
        finally:
            output_thread.join()

    def _pool_input_loop(self, pending_batches: Dict[Any, List[Task]], *args, **kwargs):
        """Infinite loop: aggregate tasks into batches and send them to runtime"""

        prev_num_tasks = 0  # number of tasks currently in shared buffer
        batch_index = max(pending_batches.keys(), default=0)
        batch_iterator = self.iterate_minibatches(*args, **kwargs)

        while True:
            # SIDE-EFFECT - compute pool priority from timestamp of earliest undispatched task
            # assumes that tasks are processed in the same order as they are created
            for skip_i in range(prev_num_tasks):
                finished_task_timestamp = (
                    self.undispatched_task_timestamps.get()
                )  # earlier timestamp = higher priority
                if skip_i == prev_num_tasks - 1:
                    self.priority = finished_task_timestamp

            logger.debug(f"{self.name} getting next batch")
            batch_tasks = next(batch_iterator)
            # save batch futures, _output_loop will deliver on them later
            pending_batches[batch_index] = batch_tasks

            logger.debug(f"{self.name}, batch  {batch_index}: aggregating inputs")
            # find or create shared arrays for current batch size
            batch_inputs = [torch.cat([task.args[i] for task in batch_tasks]) for i in range(len(batch_tasks[0].args))]
            batch_inputs = [inp.detach().requires_grad_(inp.requires_grad).share_memory_() for inp in batch_inputs]

            logger.debug(f"{self.name}, batch {batch_index}: sending to runtime")
            self.batch_sender.send((batch_index, batch_inputs))
            logger.debug(f"{self.name}, batch {batch_index}: sent to runtime")
            prev_num_tasks = len(batch_tasks)
            batch_index += 1

    def _pool_output_loop(self, pending_batches: Dict[Any, List[Task]]):
        """Infinite loop: receive results from runtime and dispatch them to task Futures"""

        while True:
            logger.debug(f"{self.name} waiting for results from runtime")
            batch_index, batch_outputs = self.outputs_receiver.recv()
            logger.debug(f"{self.name}, batch {batch_index}: got results")

            # split batch into partitions for individual tasks
            batch_tasks = pending_batches.pop(batch_index)
            task_sizes = [self.get_task_size(task) for task in batch_tasks]
            outputs_per_task = zip(*(torch.split_with_sizes(tensor, task_sizes, dim=0) for tensor in batch_outputs))
            logger.debug(f"{self.name}, batch {batch_index}: sending outputs to handlers")

            # dispatch results to futures
            for task, task_outputs in zip(batch_tasks, outputs_per_task):
                try:
                    task.future.set_result(tuple(task_outputs))
                except InvalidStateError as e:
                    logger.debug(f"Failed to send task result due to an exception: {e}")

    @property
    def empty(self):
        return not self.batch_receiver.poll()

    def load_batch_to_runtime(self, timeout=None, device=None) -> Tuple[Any, List[torch.Tensor]]:
        """receive next batch of numpy arrays"""
        if not self.batch_receiver.poll(timeout):
            raise TimeoutError()

        batch_index, batch_inputs = self.batch_receiver.recv()
        batch_inputs = [tensor.to(device, non_blocking=True) for tensor in batch_inputs]
        return batch_index, batch_inputs

    def send_outputs_from_runtime(self, batch_index: int, batch_outputs: List[torch.Tensor]):
        """send results for a processed batch, previously loaded through load_batch_to_runtime"""
        batch_outputs = [
            tensor.to(device="cpu").share_memory_().detach().requires_grad_(tensor.requires_grad)
            for tensor in batch_outputs
        ]
        self.outputs_sender.send((batch_index, batch_outputs))

    def get_task_size(self, task: Task) -> int:
        """compute task processing complexity (used for batching); defaults to batch size"""
        return len(task.args[0]) if task.args else 1
