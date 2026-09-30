import os
import sys
import json

import pandas as pd
import torch
from dataclasses import dataclass
from torch.utils.data import DataLoader
from torch_ema import ExponentialMovingAverage

from src.dataset_stage1 import TrainDataset
from src.evaluator_stage1 import predict
from src.focal_loss import FocalLoss, MaskedBCEWithLogitsLoss
from src.metric import multilabel_auc
from src.model import Stage1
from src.scheduler import get_scheduler
from src.trainer_stage1 import train
from src.utils import Logger, make_model_path, setup_system


classes = [
    "ACL", "MCL", "Medial Meniscus", "Lateral Meniscus",
    "Medial OA", "Lateral OA", "PF OA", "Effusion", "Synovitis",
    "Baker's", "Contusion", "Fracture",
]


#----------------------------------------------------------------------------------------------------------------------#
# Configuration                                                                                                        #
#----------------------------------------------------------------------------------------------------------------------#
@dataclass
class Configuration:
    model: str = "convnext_base.dinov3_lvd1689m"
    fold: int = 0
    groups_per_slot: int = 2
    steps: tuple = (1, 2)
    epochs: int = 15
    batch_size: int = 1
    lr: float = 3e-5
    focal: bool = True
    gamma: float = 2.0
    use_ema: bool = True
    ema_decay: float = 0.999
    img_size: int = 224
    warmup_epochs: float = 0.0
    scheduler: str = "cosine"
    gradient_clipping: bool = True
    gc: bool = True
    seed: int = 42
    num_workers: int = 0 if os.name == "nt" else 4
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    data_path: str = "./data/npy_study"
    model_path: str = "./model"


def evaluate(model, loader, df_valid, config):
    predictions, targets = predict(model, loader, config.device)
    print("\nStage 1 inner-fold selection AUC (known targets)")
    mean, _, _ = multilabel_auc(targets, predictions, classes)
    gold = df_valid.label_source.to_numpy() == "gold"
    if gold.any():
        print("\nGold-only diagnostic AUC")
        gold_mean, _, _ = multilabel_auc(targets[gold], predictions[gold], classes)
        print("Gold Mean            {:.4f}".format(gold_mean))
    return mean


def main():
    config = Configuration()
    setup_system(config.seed)
    model_path = make_model_path(config.model_path, config.model, config.fold)
    sys.stdout = Logger(os.path.join(model_path, "log_stage1.txt"))

    #------------------------------------------------------------------------------------------------------------------#
    # Data                                                                                                             #
    #------------------------------------------------------------------------------------------------------------------#
    df = pd.read_csv("./data/train_5_folds.csv")
    inner_fold = (config.fold + 1) % 5
    print(f"Outer evaluation fold: {config.fold}; Stage 1 selection fold: {inner_fold}")
    df_train = df[~df.fold.isin([config.fold, inner_fold])].reset_index(drop=True)
    df_valid = df[df.fold == inner_fold].reset_index(drop=True)
    if set(df_train.StudyInstanceUID) & set(df_valid.StudyInstanceUID):
        raise ValueError("Stage 1 train and selection studies overlap")
    if set(df_train.StudyInstanceUID) & set(df[df.fold == config.fold].StudyInstanceUID):
        raise ValueError("Outer evaluation study entered Stage 1 training")
    with open(os.path.join(model_path, "stage1_split.json"), "w") as file:
        json.dump({"outer_fold": config.fold, "selection_fold": inner_fold,
                   "train_folds": sorted(df_train.fold.unique().tolist())}, file)
    train_dataset = TrainDataset(df_train, classes, config.data_path, config.groups_per_slot, config.steps, True)
    valid_dataset = TrainDataset(df_valid, classes, config.data_path, config.groups_per_slot, config.steps, False)
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True,
                              num_workers=config.num_workers, pin_memory=True)
    valid_loader = DataLoader(valid_dataset, batch_size=config.batch_size, shuffle=False,
                              num_workers=config.num_workers, pin_memory=True)

    #------------------------------------------------------------------------------------------------------------------#
    # Model                                                                                                            #
    #------------------------------------------------------------------------------------------------------------------#
    model = Stage1(config.model, pretrained=True, gc=config.gc).to(config.device)
    loss_function = FocalLoss(gamma=config.gamma, ignore_index=-1) if config.focal else MaskedBCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr)
    scheduler = get_scheduler(config, optimizer, len(train_loader))
    ema = ExponentialMovingAverage(model.parameters(), decay=config.ema_decay) if config.use_ema else None
    scaler = torch.cuda.amp.GradScaler(enabled=config.device.startswith("cuda"))

    best_score = -1.0
    for epoch in range(config.epochs):
        print(f"\nEpoch: {epoch + 1}")
        loss = train(model, train_loader, loss_function, optimizer, scheduler, config.device,
                     scaler, ema, config.gradient_clipping)
        print("Avg. Train Loss = {:.5f}".format(loss))
        print("\nEvaluate")
        if ema is not None:
            with ema.average_parameters():
                score = evaluate(model, valid_loader, df_valid, config)
                if score > best_score or epoch == 0:
                    torch.save(model.state_dict(), os.path.join(model_path, "best_stage1.pth"))
        else:
            score = evaluate(model, valid_loader, df_valid, config)
            if score > best_score or epoch == 0:
                torch.save(model.state_dict(), os.path.join(model_path, "best_stage1.pth"))
        best_score = max(best_score, score)
        print("Best Mean            {:.4f}".format(best_score))


if __name__ == "__main__":
    main()
