import numpy as np
import torch
from tqdm import tqdm


def train(model, loader, loss_function, optimizer, scheduler, device, scaler=None,
          ema=None, gradient_clipping=True, verbose=True):
    model.train()
    losses = []
    progress = tqdm(loader, disable=not verbose)

    for images, token_mask, slot_ids, targets in progress:
        images = images.to(device)
        token_mask = token_mask.to(device)
        slot_ids = slot_ids.to(device)
        targets = targets.to(device)
        optimizer.zero_grad(set_to_none=True)

        with torch.autocast(device_type="cuda", enabled=device.startswith("cuda")):
            logits, _, _ = model(images, token_mask, slot_ids)
            loss = loss_function(logits, targets)

        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            if gradient_clipping:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if gradient_clipping:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        if scheduler is not None:
            scheduler.step()
        if ema is not None:
            ema.update()
        losses.append(loss.item())
        progress.set_postfix(loss=f"{np.mean(losses[-20:]):.4f}")

    return float(np.mean(losses))
