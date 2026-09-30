import os
import pickle

import pandas as pd
import torch
from dataclasses import dataclass
from torch.utils.data import DataLoader

from src.dataset_stage2 import TrainDataset
from src.evaluator_stage2 import predict
from src.metric import multilabel_auc
from src.model import Stage2


classes = [
    "ACL", "MCL", "Medial Meniscus", "Lateral Meniscus",
    "Medial OA", "Lateral OA", "PF OA", "Effusion", "Synovitis",
    "Baker's", "Contusion", "Fracture",
]


@dataclass
class Configuration:
    model: str = "convnext_base.dinov3_lvd1689m"
    fold: int = 0
    cut: int = 96
    batch_size: int = 32
    classifier_dropout: float = 0.15
    pool: str = "gem"
    num_workers: int = 0 if os.name == "nt" else 4
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


def loader_for(df, features, slots, config, step):
    dataset = TrainDataset(df.StudyInstanceUID.astype(str), features, slots,
                           df[classes].values, config.cut, step, False)
    return DataLoader(dataset, batch_size=config.batch_size, shuffle=False,
                      num_workers=config.num_workers, collate_fn=dataset.collate)


def main():
    config = Configuration()
    model_path = f"./model/{config.model}/fold-{config.fold}"
    with open(os.path.join(model_path, "features_dict.pkl"), "rb") as file:
        features = pickle.load(file)
    with open(os.path.join(model_path, "slot_dict.pkl"), "rb") as file:
        slots = pickle.load(file)
    with open(os.path.join(model_path, "config_stage2.pkl"), "rb") as file:
        transformer_config = pickle.load(file)

    model = Stage2(
        hidden_size=transformer_config.hidden_size,
        intermediate_size=transformer_config.intermediate_size,
        attention_heads=transformer_config.num_attention_heads,
        num_hidden_layers=transformer_config.num_hidden_layers,
        classifier_dropout=config.classifier_dropout,
        pool=config.pool,
    )
    model.load_state_dict(torch.load(os.path.join(model_path, "best_stage2.pth"), map_location="cpu"))
    model = model.to(config.device)
    df = pd.read_csv("./data/train_5_folds.csv")
    df = df[df.fold == config.fold].reset_index(drop=True)
    pred1, targets = predict(model, loader_for(df, features, slots, config, 1), config.device)
    pred2, _ = predict(model, loader_for(df, features, slots, config, 2), config.device)
    print("Step 1")
    multilabel_auc(targets, pred1, classes)
    print("\nStep 2")
    multilabel_auc(targets, pred2, classes)


if __name__ == "__main__":
    main()
