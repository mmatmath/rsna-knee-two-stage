import os
import pickle
import json

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


def main():
    config = Configuration()
    model_path = f"./model/{config.model}/fold-{config.fold}"
    with open(os.path.join(model_path, "stage1_split.json")) as file:
        split = json.load(file)
    if (split["outer_fold"] != config.fold or config.fold in split["train_folds"]
            or config.fold == split["selection_fold"]):
        raise ValueError("Stage 1 checkpoint used the outer evaluation fold")
    model = Stage1(config.model, pretrained=False)
    model.load_state_dict(torch.load(os.path.join(model_path, "best_stage1.pth"), map_location="cpu"))
    model = model.to(config.device).eval()
    df = pd.read_csv("./data/train_5_folds.csv")

    features_dict_list = [dict(), dict()]
    slot_dict_list = [dict(), dict()]
    for study in tqdm(df.StudyInstanceUID.astype(str)):
        with np.load(f"./data/npy_study/{study}.npz") as data:
            images = data["images"]
            slot_mask = data["slot_mask"]
        for step in (1, 2):
            windows, slot_ids = make_windows(images, slot_mask, step)
            features_dict_list[step - 1][study] = encode(model, windows, config)
            slot_dict_list[step - 1][study] = slot_ids

    with open(os.path.join(model_path, "features_dict.pkl"), "wb") as file:
        pickle.dump(features_dict_list, file)
    with open(os.path.join(model_path, "slot_dict.pkl"), "wb") as file:
        pickle.dump(slot_dict_list, file)
    print("Saved features_dict.pkl and slot_dict.pkl")


if __name__ == "__main__":
    main()
