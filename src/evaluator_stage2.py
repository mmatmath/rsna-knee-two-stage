import numpy as np
import torch
from tqdm import tqdm


@torch.no_grad()
def predict(model, loader, device, verbose=True):
    model.eval()
    predictions = []
    targets = []

    for features, slot_ids, attention_mask, target in tqdm(loader, disable=not verbose):
        logits = model(features.to(device), slot_ids.to(device), attention_mask.to(device))
        predictions.append(torch.sigmoid(logits).cpu().numpy())
        targets.append(target.numpy())

    return np.concatenate(predictions), np.concatenate(targets)
