import threading

from hivemind.dht import DHT
from hivemind.utils import Endpoint, get_port
from hivemind.client.dht_ops import declare_experts


class DHTHandlerThread(threading.Thread):
    def __init__(self, experts, dht: DHT, endpoint: Endpoint, update_period: int = 5):
        super().__init__()
        assert get_port(endpoint) is not None
        self.endpoint = endpoint
        self.experts = experts
        self.dht = dht
        self.update_period = update_period
        self.stop = threading.Event()

    def run(self) -> None:
        declare_experts(self.dht, self.experts.keys(), self.endpoint)
        while not self.stop.wait(self.update_period):
            declare_experts(self.dht, self.experts.keys(), self.endpoint)
