import os
import pickle
import sys

import numpy as np
import pandas as pd
import torch
from dataclasses import dataclass
from torch.utils.data import DataLoader
from torch_ema import ExponentialMovingAverage

from src.dataset_stage2 import TrainDataset
from src.evaluator_stage2 import predict
from src.focal_loss import FocalLoss, MaskedBCEWithLogitsLoss
from src.metric import multilabel_auc
from src.model import Model, Stage1, Stage2
from src.scheduler import get_scheduler
from src.trainer_stage2 import train
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
    transformer: str = "microsoft/deberta-v3-base"
    fold: int = 0
    cut: int = 96
    epochs: int = 12
    batch_size: int = 32
    lr: float = 1e-5
    hidden_size: int = 1024
    intermediate_size: int = 1024
    num_hidden_layers: int = 3
    attention_heads: int = 8
    attention_dropout: float = 0.05
    hidden_dropout: float = 0.15
    classifier_dropout: float = 0.15
    focal: bool = True
    gamma: float = 2.0
    use_ema: bool = True
    ema_decay: float = 0.999
    pool: str = "gem"
    warmup_epochs: float = 0.0
    scheduler: str = "constant"
    gradient_clipping: bool = True
    gc: bool = False
    seed: int = 42
    num_workers: int = 0 if os.name == "nt" else 4
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    model_path: str = "./model"


def make_loader(df, features, slots, config, train_mode, step=(1, 2)):
    dataset = TrainDataset(
        df.StudyInstanceUID.astype(str), features, slots, df[classes].values,
        cut=config.cut, step=step, train=train_mode,
    )
    return DataLoader(dataset, batch_size=config.batch_size, shuffle=train_mode,
                      num_workers=config.num_workers, collate_fn=dataset.collate)


def evaluate(model, loader1, loader2, df_valid, config):
    pred1, targets = predict(model, loader1, config.device)
    pred2, _ = predict(model, loader2, config.device)
    print("\nStep 1")
    score1, _, _ = multilabel_auc(targets, pred1, classes)
    print("\nStep 2")
    score2, _, _ = multilabel_auc(targets, pred2, classes)
    gold = df_valid.label_source.to_numpy() == "gold"
    if gold.any():
        print("\nGold-only validation AUC")
        multilabel_auc(targets[gold], pred1[gold], classes)
    return float(np.nanmean([score1, score2]))


def save_inference_weights(stage2, config, model_path):
    with open(os.path.join(model_path, "config_stage2.pkl"), "rb") as file:
        transformer_config = pickle.load(file)
    combined = Model(config.model, transformer_config, config.classifier_dropout,
                     pool=config.pool)
    stage1 = Stage1(config.model, pretrained=False)
    stage1.load_state_dict(torch.load(os.path.join(model_path, "best_stage1.pth"), map_location="cpu"))
    combined.image_encoder.load_state_dict(stage1.image_encoder.state_dict())
    combined.transformer.load_state_dict(stage2.transformer.state_dict())
    combined.slot_embedding.load_state_dict(stage2.slot_embedding.state_dict())
    combined.cls_embedding.data.copy_(stage2.cls_embedding.data)
    combined.pool.load_state_dict(stage2.pool.state_dict())
    combined.fc.load_state_dict(stage2.fc.state_dict())
    torch.save(combined.state_dict(), os.path.join(model_path, "weights_inference.pth"))


def main():
    config = Configuration()
    setup_system(config.seed)
    model_path = make_model_path(config.model_path, config.model, config.fold)
    sys.stdout = Logger(os.path.join(model_path, "log_stage2.txt"))
    #------------------------------------------------------------------------------------------------------------------#
    # Data                                                                                                             #
    #------------------------------------------------------------------------------------------------------------------#
    with open(os.path.join(model_path, "features_dict.pkl"), "rb") as file:
        features = pickle.load(file)
    with open(os.path.join(model_path, "slot_dict.pkl"), "rb") as file:
        slots = pickle.load(file)
    feature_dim = int(np.asarray(next(iter(features[0].values()))).shape[-1])
    if feature_dim != config.hidden_size:
        raise ValueError(f"ConvNeXt features have D={feature_dim}; set hidden_size={feature_dim}")

    df = pd.read_csv("./data/train_5_folds.csv")
    df_train = df[df.fold != config.fold].reset_index(drop=True)
    df_valid = df[df.fold == config.fold].reset_index(drop=True)
    train_loader = make_loader(df_train, features, slots, config, True)
    valid_loader1 = make_loader(df_valid, features, slots, config, False, 1)
    valid_loader2 = make_loader(df_valid, features, slots, config, False, 2)

    #------------------------------------------------------------------------------------------------------------------#
    # Model                                                                                                            #
    #------------------------------------------------------------------------------------------------------------------#
    model = Stage2(
        config.transformer, config.hidden_size, config.intermediate_size,
        config.attention_heads, config.num_hidden_layers, config.attention_dropout,
        config.hidden_dropout, config.classifier_dropout, gc=config.gc, pool=config.pool,
    ).to(config.device)
    with open(os.path.join(model_path, "config_stage2.pkl"), "wb") as file:
        pickle.dump(model.config, file)

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
                score = evaluate(model, valid_loader1, valid_loader2, df_valid, config)
                if score > best_score:
                    torch.save(model.state_dict(), os.path.join(model_path, "best_stage2.pth"))
        else:
            score = evaluate(model, valid_loader1, valid_loader2, df_valid, config)
            if score > best_score:
                torch.save(model.state_dict(), os.path.join(model_path, "best_stage2.pth"))
        best_score = max(best_score, score)
        print("Best Mean            {:.4f}".format(best_score))

    model.load_state_dict(torch.load(os.path.join(model_path, "best_stage2.pth"), map_location=config.device))
    model = model.cpu()
    save_inference_weights(model, config, model_path)
    print("Saved: weights_inference.pth")


if __name__ == "__main__":
    main()
