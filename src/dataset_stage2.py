import numpy as np
import torch
from torch.utils.data import Dataset


class TrainDataset(Dataset):
    def __init__(self, studies, features, slots, labels, cut=96, step=(1, 2), train=False):
        self.studies = list(studies)
        self.features = features
        self.slots = slots
        self.labels = np.asarray(labels, dtype=np.float32)
        self.cut = cut
        self.step = step
        self.train = train

    def __len__(self):
        return len(self.studies)

    def __getitem__(self, index):
        study = self.studies[index]
        if self.train:
            step_index = int(np.random.choice(self.step)) - 1
        else:
            step_index = int(self.step) - 1

        features = np.asarray(self.features[step_index][study], dtype=np.float32)
        slot_ids = np.asarray(self.slots[step_index][study], dtype=np.int64)

        if len(features) > self.cut:
            indices = np.linspace(0, len(features) - 1, self.cut).round().astype(int)
            features = features[indices]
            slot_ids = slot_ids[indices]

        return (
            torch.from_numpy(features),
            torch.from_numpy(slot_ids),
            torch.tensor(self.labels[index], dtype=torch.float32),
            len(features),
        )

    def collate(self, batch):
        max_length = max(item[3] for item in batch)
        dim = batch[0][0].shape[-1]
        batch_size = len(batch)
        features = torch.zeros(batch_size, max_length, dim, dtype=torch.float32)
        slot_ids = torch.zeros(batch_size, max_length, dtype=torch.long)
        attention_mask = torch.zeros(batch_size, max_length, dtype=torch.long)

        for i, (item_features, item_slots, _, length) in enumerate(batch):
            features[i, :length] = item_features
            slot_ids[i, :length] = item_slots
            attention_mask[i, :length] = 1

        targets = torch.stack([item[2] for item in batch])
        return features, slot_ids, attention_mask, targets
