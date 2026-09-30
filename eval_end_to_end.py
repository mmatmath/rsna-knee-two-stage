import os
import pickle

import numpy as np
import pandas as pd
import torch
from dataclasses import dataclass
from tqdm import tqdm

from preprocess_data import IMG_SIZE, N_SLICES, choose_series, load_series, sample_series
from src.metric import multilabel_auc
from src.model import Model
from src.utils import cut_features, make_windows, normalize_images


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
    cut: int = 96
    batch_size: int = 64
    classifier_dropout: float = 0.15
    pool: str = "mean"
    dicom_path: str = "./data/train_series"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


def load_study(df_study, config):
    selected = choose_series(df_study, config.dicom_path)
    images = np.zeros((6, N_SLICES, IMG_SIZE, IMG_SIZE), dtype=np.uint8)
    slot_mask = np.zeros(6, dtype=np.uint8)
    for slot, (_, _, files) in selected.items():
        volume = load_series(files)
        if volume is not None:
            images[slot] = sample_series(volume)
            slot_mask[slot] = 1
    return images, slot_mask


@torch.no_grad()
def predict_study(model, images, slot_mask, config):
    windows, slot_ids = make_windows(images, slot_mask, config.step)
    features = []
    for start in range(0, len(windows), config.batch_size):
        x = normalize_images(torch.from_numpy(windows[start:start + config.batch_size]).to(config.device))
        features.append(model.forward_encoder(x))
    features, slot_ids = cut_features(torch.cat(features), torch.from_numpy(slot_ids).to(config.device), config.cut)
    features = features.unsqueeze(0)
    slot_ids = slot_ids.unsqueeze(0)
    attention_mask = torch.ones_like(slot_ids)
    return torch.sigmoid(model.forward_transformer(features, slot_ids, attention_mask))[0].cpu().numpy()


def main():
    config = Configuration()
    model_path = f"./model/{config.model}/fold-{config.fold}"
    with open(os.path.join(model_path, "config_stage2.pkl"), "rb") as file:
        transformer_config = pickle.load(file)
    model = Model(config.model, transformer_config, config.classifier_dropout, pool=config.pool)
    model.load_state_dict(torch.load(os.path.join(model_path, "weights_inference.pth"), map_location="cpu"))
    model = model.to(config.device).eval()

    df = pd.read_csv("./data/train_5_folds.csv")
    df = df[df.fold == config.fold].reset_index(drop=True)
    df_series = pd.read_csv("./data/train_series.csv")
    predictions = []
    for study in tqdm(df.StudyInstanceUID):
        rows = df_series[df_series.StudyInstanceUID == study]
        images, slot_mask = load_study(rows, config)
        predictions.append(predict_study(model, images, slot_mask, config))
    predictions = np.stack(predictions)
    print("Weak-label CV AUC (known targets)")
    multilabel_auc(df[classes].values, predictions, classes)
    gold = df.label_source.to_numpy() == "gold"
    if gold.any():
        print("\nGold-only diagnostic AUC")
        multilabel_auc(df.loc[gold, classes].values, predictions[gold], classes)

    output = df[["StudyInstanceUID", "label_source"]].copy()
    for i, name in enumerate(classes):
        output[name] = predictions[:, i]
    output.to_csv(os.path.join(model_path, "predictions_end_to_end.csv"), index=False)


if __name__ == "__main__":
    main()
