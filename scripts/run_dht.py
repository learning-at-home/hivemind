import argparse
import resource
import os
import sys

import torch
import tesseract
from tesseract.utils import find_open_port


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, default=None, required=False)
    parser.add_argument('--initial_peers', type=str, default="[]", required=False)

    args = parser.parse_args()
    initial_peers = eval(args.initial_peers)
    print("Parsed initial peers:", initial_peers)

    network = tesseract.TesseractNetwork(*initial_peers, port=args.port or find_open_port(), start=False)
    print(f"Running network node on port {network.port}")

    try:
        network.run()
    finally:
        network.shutdown()
