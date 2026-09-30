import os

import albumentations as A
import numpy as np
import torch
from torch.utils.data import Dataset

from src.utils import normalize_images


def get_augmentation(train=True):
    if not train:
        return None
    return A.Compose([
        A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=7,
                           border_mode=0, p=0.5),
        A.CoarseDropout(num_holes_range=(1, 4), hole_height_range=(0.05, 0.15),
                        hole_width_range=(0.05, 0.15), fill=0, p=0.3),
    ])


class TrainDataset(Dataset):
    def __init__(self, df, classes, data_path="./data/npy_study", groups_per_slot=2,
                 steps=(1, 2), train=True):
        self.df = df.reset_index(drop=True)
        self.classes = classes
        self.data_path = data_path
        self.groups_per_slot = groups_per_slot
        self.steps = steps
        self.train = train
        self.augmentation = get_augmentation(train)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        row = self.df.iloc[index]
        path = os.path.join(self.data_path, str(row.StudyInstanceUID) + ".npz")
        with np.load(path) as data:
            images = data["images"]       # [6, 24, H, W]
            slot_mask = data["slot_mask"] # [6]
        if not slot_mask.any():
            raise ValueError(f"No usable MRI slot for study {row.StudyInstanceUID}")

        windows = []
        token_mask = []
        slot_ids = []

        for slot in range(6):
            for group in range(self.groups_per_slot):
                if slot_mask[slot]:
                    step = int(np.random.choice(self.steps)) if self.train else self.steps[group % len(self.steps)]
                    low = step
                    high = images.shape[1] - step
                    if self.train:
                        center = np.random.randint(low, high)
                    else:
                        fraction = (group + 1) / (self.groups_per_slot + 1)
                        center = int(round(low + fraction * (high - low - 1)))
                    window = images[slot, [center - step, center, center + step]]
                    token_mask.append(1)
                else:
                    window = np.zeros((3, images.shape[2], images.shape[3]), dtype=np.uint8)
                    token_mask.append(0)

                if self.augmentation is not None:
                    image_hwc = window.transpose(1, 2, 0)
                    window = self.augmentation(image=image_hwc)["image"].transpose(2, 0, 1)
                windows.append(window)
                slot_ids.append(slot)

        images = normalize_images(torch.from_numpy(np.stack(windows)))
        targets = row[self.classes].to_numpy(dtype=np.float32)

        return (
            images,
            torch.tensor(token_mask, dtype=torch.long),
            torch.tensor(slot_ids, dtype=torch.long),
            torch.from_numpy(targets),
        )
