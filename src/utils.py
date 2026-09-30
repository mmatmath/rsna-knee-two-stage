import os
import random
import sys

import numpy as np
import torch


# timm's convnext_base.dinov3_lvd1689m pretrained data configuration.
IMAGE_MEAN = (0.485, 0.456, 0.406)
IMAGE_STD = (0.229, 0.224, 0.225)


def normalize_images(images):
    mean = torch.tensor(IMAGE_MEAN, dtype=torch.float32, device=images.device).view(1, 3, 1, 1)
    std = torch.tensor(IMAGE_STD, dtype=torch.float32, device=images.device).view(1, 3, 1, 1)
    return (images.float() / 255.0 - mean) / std


def make_windows(images, slot_mask, step):
    windows = []
    slot_ids = []
    for slot in range(6):
        if not slot_mask[slot]:
            continue
        for center in range(images.shape[1]):
            indices = np.clip([center - step, center, center + step], 0, images.shape[1] - 1)
            windows.append(images[slot, indices])
            slot_ids.append(slot)
    if not windows:
        windows.append(np.zeros((3, *images.shape[-2:]), dtype=np.uint8))
        slot_ids.append(0)
    return np.stack(windows), np.asarray(slot_ids, dtype=np.int64)


def cut_features(features, slot_ids, cut):
    if len(features) > cut:
        indices = np.linspace(0, len(features) - 1, cut).round().astype(int)
        if torch.is_tensor(features):
            indices = torch.as_tensor(indices, device=features.device)
        features = features[indices]
        slot_ids = slot_ids[indices]
    return features, slot_ids


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
