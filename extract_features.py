import os
import pickle
import json
from zipfile import BadZipFile

import numpy as np
import pandas as pd
import torch
from dataclasses import dataclass
from tqdm import tqdm

from src.model import Stage1
from src.utils import make_windows, normalize_images


classes = [
    "ACL", "MCL", "Medial Meniscus", "Lateral Meniscus",
    "Medial OA", "Lateral OA", "PF OA", "Effusion", "Synovitis",
    "Baker's", "Contusion", "Fracture",
]
CACHE_VERSION = 1


@dataclass
class Configuration:
    model: str = "convnext_base.dinov3_lvd1689m"
    fold: int = 0
    batch_size: int = 64
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


@torch.no_grad()
def encode(model, windows, config):
    features = []
    for start in range(0, len(windows), config.batch_size):
        images = normalize_images(torch.from_numpy(windows[start:start + config.batch_size]).to(config.device))
        features.append(model.image_encoder(images).cpu().numpy())
    return np.concatenate(features).astype(np.float16)


def file_stamp(path):
    stat = os.stat(path)
    return np.array([stat.st_mtime_ns, stat.st_size], dtype=np.int64)


def study_features(study, cache_dir, checkpoint_path, model, config):
    input_path = f"./data/npy_study/{study}.npz"
    cache_path = os.path.join(cache_dir, f"{study}.npz")
    checkpoint_stamp = file_stamp(checkpoint_path)
    input_stamp = file_stamp(input_path)
    if os.path.exists(cache_path):
        try:
            with np.load(cache_path) as data:
                if (int(data["cache_version"]) == CACHE_VERSION
                        and np.array_equal(data["checkpoint_stamp"], checkpoint_stamp)
                        and np.array_equal(data["input_stamp"], input_stamp)):
                    return ([data["features_1"], data["features_2"]],
                            [data["slot_ids_1"], data["slot_ids_2"]], True)
        except (OSError, ValueError, KeyError, BadZipFile):
            pass

    with np.load(input_path) as data:
        images = data["images"]
        slot_mask = data["slot_mask"]
    features, slot_ids = [], []
    for step in (1, 2):
        windows, slots = make_windows(images, slot_mask, step)
        features.append(encode(model, windows, config))
        slot_ids.append(slots)
    os.makedirs(cache_dir, exist_ok=True)
    temp_path = cache_path + ".tmp.npz"
    np.savez_compressed(temp_path, features_1=features[0], features_2=features[1],
                        slot_ids_1=slot_ids[0], slot_ids_2=slot_ids[1],
                        cache_version=CACHE_VERSION, checkpoint_stamp=checkpoint_stamp,
                        input_stamp=input_stamp)
    os.replace(temp_path, cache_path)
    return features, slot_ids, False


def main():
    config = Configuration()
    model_path = f"./model/{config.model}/fold-{config.fold}"
    with open(os.path.join(model_path, "stage1_split.json")) as file:
        split = json.load(file)
    if (split["outer_fold"] != config.fold or config.fold in split["train_folds"]
            or config.fold == split["selection_fold"]):
        raise ValueError("Stage 1 checkpoint used the outer evaluation fold")
    checkpoint_path = os.path.join(model_path, "best_stage1.pth")
    model = Stage1(config.model, pretrained=False)
    model.load_state_dict(torch.load(checkpoint_path, map_location="cpu"))
    model = model.to(config.device).eval()
    df = pd.read_csv("./data/train_5_folds.csv")

    features_dict_list = [dict(), dict()]
    slot_dict_list = [dict(), dict()]
    cache_dir = os.path.join(model_path, "feature_cache")
    reused = 0
    for study in tqdm(df.StudyInstanceUID.astype(str)):
        features, slot_ids, cached = study_features(study, cache_dir, checkpoint_path, model, config)
        reused += cached
        for step in (0, 1):
            features_dict_list[step][study] = features[step]
            slot_dict_list[step][study] = slot_ids[step]

    for name, value in (("features_dict.pkl", features_dict_list), ("slot_dict.pkl", slot_dict_list)):
        path = os.path.join(model_path, name)
        temp_path = path + ".tmp"
        with open(temp_path, "wb") as file:
            pickle.dump(value, file)
        os.replace(temp_path, path)
    print("Reused feature caches:", reused, "of", len(df))
    print("Saved features_dict.pkl and slot_dict.pkl")


if __name__ == "__main__":
    main()
