import os
import argparse

from mosec import Server
from collections import OrderedDict
from backend_model import (SnifferGPT2Model,
                           SnifferGPTJModel,
                           SnifferLlama2Model,
                           SnifferAlpacaModel,
                           SnifferVicunaModel)

MODEL_MAPPING_NAMES = OrderedDict([
    ("gpt2", SnifferGPT2Model),
    ("gptj", SnifferGPTJModel),
    ("llama2", SnifferLlama2Model),
    ("alpaca", SnifferAlpacaModel),
    ("vicuna", SnifferVicunaModel)
])


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        default="gpt2"
    )
    parser.add_argument("--gpu",
                        type=str,
                        required=False,
                        default='0',
                        help="Set os.environ['CUDA_VISIBLE_DEVICES'].")

    parser.add_argument("--port", help="mosec args.")
    parser.add_argument("--timeout", help="mosec args.")
    parser.add_argument("--debug", action="store_true", help="mosec args.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    sniffer_model = MODEL_MAPPING_NAMES[args.model]
    server = Server()
    server.append_worker(sniffer_model)
    server.run()