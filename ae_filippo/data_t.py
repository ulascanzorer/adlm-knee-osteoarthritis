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
        EXCLUDE_L = {
        "9070923",
        "9388265",
        "9594253",
        "9860568",}
        EXCLUDE_R = {
        "9004315",
        "9537947",
        "9637394",
        "9462278",
        "9522128",}
        for pid, _ in iter_mri_dataset(root, side, max_patients=max_patients):
            if side == "left" and pid in EXCLUDE_L:
                continue
            if side == "right" and pid in EXCLUDE_R:
                continue
            
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
