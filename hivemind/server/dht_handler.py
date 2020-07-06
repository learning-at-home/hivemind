import threading
import time

from ..dht import DHT


class DHTHandlerThread(threading.Thread):
    def __init__(self, experts, dht: DHT,
                 update_period: int = 5, addr: str = '127.0.0.1', port: int = 8080):
        super(DHTHandlerThread, self).__init__()
        self.port = port
        self.addr = addr
        self.experts = experts
        self.dht = dht
        self.update_period = update_period
        self.stop = False

    def run(self) -> None:
        while not self.stop:
            self.dht.declare_experts(self.experts.keys(), self.addr, self.port)
            time.sleep(self.update_period)
