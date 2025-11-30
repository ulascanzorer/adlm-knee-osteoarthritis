# data_t.py
from data import iter_mri_dataset
import torch
from torch.utils.data import Dataset


class KneeMRIDataset(Dataset):
    def __init__(self, root, side='left', max_patients=None):
        self.volumes = []
        self.patient_ids = []

        for pid, vol in iter_mri_dataset(root, side, max_patients=max_patients):
            # vol should already be a torch.Tensor [1, D, 224, 224]
            if not torch.is_tensor(vol):
                vol = torch.tensor(vol, dtype=torch.float32)
            self.volumes.append(vol)
            self.patient_ids.append(pid)

    def __len__(self):
        return len(self.volumes)

    def __getitem__(self, idx):
        vol = self.volumes[idx]  # (1, D, 224, 224)
        return vol
