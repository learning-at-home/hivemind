import threading
import time

from ..network import TesseractNetwork


class NetworkHandlerThread(threading.Thread):
    def __init__(self, experts, network: TesseractNetwork,
                 update_period: int = 5, addr: str = '127.0.0.1', port: int = 8080):
        super(NetworkHandlerThread, self).__init__()
        self.port = port
        self.addr = addr
        self.experts = experts
        self.network = network
        self.update_period = update_period

    def run(self) -> None:
        while True:
            self.network.declare_experts(self.experts.keys(), self.addr, self.port)
            time.sleep(self.update_period)
