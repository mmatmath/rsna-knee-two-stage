import os
import pickle

import numpy as np
import pandas as pd
import torch
from dataclasses import dataclass
from tqdm import tqdm

from preprocess_data import IMG_SIZE, N_SLICES, choose_series, load_series, sample_series
from src.model import Model


classes = [
    "ACL", "MCL", "Medial Meniscus", "Lateral Meniscus",
    "Medial OA", "Lateral OA", "PF OA", "Effusion", "Synovitis",
    "Baker's", "Contusion", "Fracture",
]


@dataclass
class Configuration:
    model: str = "convnext_base.dinov3_lvd1689m"
    fold: int = 0
    step: int = 1
    batch_size: int = 64
    classifier_dropout: float = 0.15
    pool: str = "gem"
    test_csv: str = "./data/test.csv"
    series_csv: str = "./data/test_series.csv"
    dicom_path: str = "./data/test_images"
    output_path: str = "./submission.csv"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


def preprocess_study(df_study, config):
    selected = choose_series(df_study, config.dicom_path)
    images = np.zeros((6, N_SLICES, IMG_SIZE, IMG_SIZE), dtype=np.uint8)
    slot_mask = np.zeros(6, dtype=np.uint8)
    for slot, (_, _, files) in selected.items():
        volume = load_series(files)
        if volume is not None:
            images[slot] = sample_series(volume)
            slot_mask[slot] = 1
    return images, slot_mask


def make_windows(images, slot_mask, step):
    windows = []
    slot_ids = []
    for slot in range(6):
        if not slot_mask[slot]:
            continue
        for center in range(images.shape[1]):
            index = np.clip([center - step, center, center + step], 0, images.shape[1] - 1)
            windows.append(images[slot, index])
            slot_ids.append(slot)
    if not windows:
        windows.append(np.zeros((3, IMG_SIZE, IMG_SIZE), dtype=np.uint8))
        slot_ids.append(0)
    return np.stack(windows), np.asarray(slot_ids, dtype=np.int64)


@torch.no_grad()
def predict_study(model, images, slot_mask, config):
    windows, slot_ids = make_windows(images, slot_mask, config.step)
    features = []
    for start in range(0, len(windows), config.batch_size):
        x = torch.from_numpy(windows[start:start + config.batch_size]).float() / 255.0
        x = ((x - 0.5) / 0.5).to(config.device)
        features.append(model.forward_encoder(x))
    features = torch.cat(features).unsqueeze(0)
    slot_ids = torch.from_numpy(slot_ids).unsqueeze(0).to(config.device)
    attention_mask = torch.ones_like(slot_ids)
    logits = model.forward_transformer(features, slot_ids, attention_mask)
    return torch.sigmoid(logits)[0].cpu().numpy()


def main():
    config = Configuration()
    model_path = f"./model/{config.model}/fold-{config.fold}"
    with open(os.path.join(model_path, "config_stage2.pkl"), "rb") as file:
        transformer_config = pickle.load(file)
    model = Model(config.model, transformer_config, config.classifier_dropout, pool=config.pool)
    model.load_state_dict(torch.load(os.path.join(model_path, "weights_inference.pth"), map_location="cpu"))
    model = model.to(config.device).eval()

    df_test = pd.read_csv(config.test_csv)
    df_series = pd.read_csv(config.series_csv)
    predictions = []
    for study in tqdm(df_test.StudyInstanceUID):
        rows = df_series[df_series.StudyInstanceUID == study]
        images, slot_mask = preprocess_study(rows, config)
        predictions.append(predict_study(model, images, slot_mask, config))

    submission = pd.DataFrame(np.stack(predictions), columns=classes)
    submission.insert(0, "StudyInstanceUID", df_test.StudyInstanceUID.values)
    submission.to_csv(config.output_path, index=False)
    print("Saved:", config.output_path)


if __name__ == "__main__":
    main()
