import os
import random
import sys

import numpy as np
import torch


class Logger:
    def __init__(self, path):
        self.console = sys.stdout
        self.file = open(path, "a", buffering=1)

    def write(self, message):
        self.console.write(message)
        self.file.write(message)

    def flush(self):
        self.console.flush()
        self.file.flush()


def setup_system(seed=42, cudnn_benchmark=True, cudnn_deterministic=False):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = cudnn_benchmark
    torch.backends.cudnn.deterministic = cudnn_deterministic


def make_model_path(root, encoder_name, fold):
    path = os.path.join(root, encoder_name, f"fold-{fold}")
    os.makedirs(path, exist_ok=True)
    return path


def print_line():
    print("-" * 80)
