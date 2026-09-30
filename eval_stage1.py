import os

import pandas as pd
import torch
from dataclasses import dataclass
from torch.utils.data import DataLoader

from src.dataset_stage1 import TrainDataset
from src.evaluator_stage1 import predict
from src.metric import multilabel_auc
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
    groups_per_slot: int = 2
    steps: tuple = (1, 2)
    batch_size: int = 4
    num_workers: int = 0 if os.name == "nt" else 4
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    dump_predictions: bool = True


def main():
    config = Configuration()
    model_path = f"./model/{config.model}/fold-{config.fold}"
    df = pd.read_csv("./data/train_5_folds.csv")
    df = df[df.fold == config.fold].reset_index(drop=True)
    dataset = TrainDataset(df, classes, groups_per_slot=config.groups_per_slot,
                           steps=config.steps, train=False)
    loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=False,
                        num_workers=config.num_workers)
    model = Stage1(config.model, pretrained=False)
    model.load_state_dict(torch.load(os.path.join(model_path, "best_stage1.pth"), map_location="cpu"))
    model = model.to(config.device)
    predictions, targets = predict(model, loader, config.device)
    multilabel_auc(targets, predictions, classes)

    gold = df.label_source.to_numpy() == "gold"
    if gold.any():
        print("\nGold-only validation AUC")
        multilabel_auc(targets[gold], predictions[gold], classes)
    if config.dump_predictions:
        output = df[["StudyInstanceUID", "label_source"]].copy()
        for i, name in enumerate(classes):
            output[name] = predictions[:, i]
        output.to_csv(os.path.join(model_path, "predictions_stage1.csv"), index=False)


if __name__ == "__main__":
    main()
