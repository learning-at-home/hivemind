import multiprocessing as mp
import threading
from itertools import chain
from selectors import DefaultSelector, EVENT_READ
from typing import Dict

import torch
import tqdm
from prefetch_generator import BackgroundGenerator

from .expert_backend import ExpertBackend
from .task_pool import TaskPool, TaskPoolBase


class Runtime(threading.Thread):
    """
    A group of processes that processes incoming requests for multiple experts on a shared device.
    Runtime is usually created and managed by Server, humans need not apply.

    For debugging, you can start runtime manually with .start() or .run()

    >>> expert_backends = {'expert_name': ExpertBackend(**kwargs)}
    >>> runtime = Runtime(expert_backends)
    >>> runtime.start()  # start runtime in background thread. To start in current thread, use runtime.run()
    >>> runtime.ready.wait()  # await for runtime to load all experts on device and create request pools
    >>> future = runtime.expert_backends['expert_name'].forward_pool.submit_task(*expert_inputs)
    >>> print("Returned:", future.result())
    >>> runtime.shutdown()

    :param expert_backends: a dict [expert uid -> ExpertBackend]
    :param prefetch_batches: form up to this many batches in advance
    :param start: start runtime immediately (at the end of __init__)
    :param sender_threads: dispatches outputs from finished batches using this many asynchronous threads
    :param device: if specified, moves all experts and data to this device via .to(device=device).
      If you want to manually specify devices for each expert (in their forward pass), leave device=None (default)
    """

    def __init__(self, expert_backends: Dict[str, ExpertBackend], prefetch_batches=64, sender_threads: int = 1,
                 device: torch.device = None):
        super().__init__()
        self.expert_backends = expert_backends
        self.pools = tuple(chain(*(expert.get_pools() for expert in expert_backends.values())))
        self.device, self.prefetch_batches, self.sender_threads = device, prefetch_batches, sender_threads
        self.shutdown_recv, self.shutdown_send = mp.Pipe(duplex=False)
        self.ready = mp.Event()  # event is set iff server is currently running and ready to accept batches

    def run(self):
        progress = tqdm.tqdm(bar_format='{desc}, {rate_fmt}')
        for pool in self.pools:
            if not pool.is_alive():
                pool.start()
        if self.device is not None:
            for expert_backend in self.expert_backends.values():
                expert_backend.to(self.device)

        with mp.pool.ThreadPool(self.sender_threads) as output_sender_pool:
            try:
                self.ready.set()
                for pool, batch_index, batch in BackgroundGenerator(
                        self.iterate_minibatches_from_pools(), self.prefetch_batches):
                    outputs = pool.process_func(*batch)
                    output_sender_pool.apply_async(pool.send_outputs_from_runtime, args=[batch_index, outputs])
                    progress.update(len(outputs[0]))
                    progress.desc = f'pool.uid={pool.uid} batch_size={len(outputs[0])}'
            finally:
                self.shutdown()

    SHUTDOWN_TRIGGER = "RUNTIME SHUTDOWN TRIGGERED"

    def shutdown(self):
        """ Gracefully terminate a running runtime. """
        self.ready.clear()
        self.shutdown_send.send(self.SHUTDOWN_TRIGGER)  # trigger background thread to shutdown
        for pool in self.pools:
            if pool.is_alive():
                pool.terminate()
                pool.join()

    def iterate_minibatches_from_pools(self, timeout=None):
        """
        Chooses pool according to priority, then copies exposed batch and frees the buffer
        """
        with DefaultSelector() as selector:
            selector.register(self.shutdown_recv, EVENT_READ, self.SHUTDOWN_TRIGGER)
            for pool in self.pools:
                selector.register(pool.batch_receiver, EVENT_READ, pool)

            while True:
                # wait until at least one batch_receiver becomes available
                ready_fds = selector.select()
                ready_objects = {key.data for (key, events) in ready_fds}
                if self.SHUTDOWN_TRIGGER in ready_objects:
                    break  # someone asked us to shutdown, break from the loop

                pool = max(ready_objects, key=lambda pool: pool.priority)

                batch_index, batch_tensors = pool.load_batch_to_runtime(timeout, self.device)
                yield pool, batch_index, batch_tensors
