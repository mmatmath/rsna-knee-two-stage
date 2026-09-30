import numpy as np
import torch
from tqdm import tqdm


@torch.no_grad()
def predict(model, loader, device, verbose=True, return_attention=False):
    model.eval()
    predictions = []
    targets = []
    attentions = []

    for images, token_mask, slot_ids, target in tqdm(loader, disable=not verbose):
        logits, _, attention = model(images.to(device), token_mask.to(device), slot_ids.to(device))
        predictions.append(torch.sigmoid(logits).cpu().numpy())
        targets.append(target.numpy())
        if return_attention:
            attentions.append(attention.cpu().numpy())

    output = (np.concatenate(predictions), np.concatenate(targets))
    if return_attention:
        output += (np.concatenate(attentions),)
    return output
