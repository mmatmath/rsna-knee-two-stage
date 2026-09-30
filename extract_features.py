import os
import pickle

import numpy as np
import pandas as pd
import torch
from dataclasses import dataclass
from tqdm import tqdm

from src.model import Stage1


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
    return np.stack(windows), np.asarray(slot_ids, dtype=np.int64)


@torch.no_grad()
def encode(model, windows, config):
    features = []
    for start in range(0, len(windows), config.batch_size):
        images = torch.from_numpy(windows[start:start + config.batch_size]).float() / 255.0
        images = ((images - 0.5) / 0.5).to(config.device)
        features.append(model.image_encoder(images).cpu().numpy())
    return np.concatenate(features).astype(np.float16)


def main():
    config = Configuration()
    model_path = f"./model/{config.model}/fold-{config.fold}"
    model = Stage1(config.model, pretrained=False)
    model.load_state_dict(torch.load(os.path.join(model_path, "best_stage1.pth"), map_location="cpu"))
    model = model.to(config.device).eval()
    df = pd.read_csv("./data/train_5_folds.csv")

    features_dict_list = [dict(), dict()]
    slot_dict_list = [dict(), dict()]
    for study in tqdm(df.StudyInstanceUID.astype(str)):
        data = np.load(f"./data/npy_study/{study}.npz")
        for step in (1, 2):
            windows, slot_ids = make_windows(data["images"], data["slot_mask"], step)
            features_dict_list[step - 1][study] = encode(model, windows, config)
            slot_dict_list[step - 1][study] = slot_ids

    with open(os.path.join(model_path, "features_dict.pkl"), "wb") as file:
        pickle.dump(features_dict_list, file)
    with open(os.path.join(model_path, "slot_dict.pkl"), "wb") as file:
        pickle.dump(slot_dict_list, file)
    print("Saved features_dict.pkl and slot_dict.pkl")


if __name__ == "__main__":
    main()
