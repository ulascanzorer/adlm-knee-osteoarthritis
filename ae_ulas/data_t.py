# data_t.py
from data import iter_mri_dataset, load_single_patient_mri
import torch
from torch.utils.data import Dataset


class KneeMRIDataset(Dataset):
    def __init__(self, root, side='left', max_patients=None):
        self.root = root
        self.side = side
        self.max_patients = max_patients

        self.patient_ids = []
        for pid, _ in iter_mri_dataset(root, side, max_patients=max_patients):
            self.patient_ids.append(pid)

    def __len__(self):
        return len(self.patient_ids)

    def __getitem__(self, idx):
        # Load the specific volume on-demand.
        target_pid = self.patient_ids[idx]

        vol = load_single_patient_mri(self.root, target_pid, self.side)
        if not torch.is_tensor(vol):
            vol = torch.tensor(vol, dtype=torch.float32)

        return vol  # (1, D, 224, 224)