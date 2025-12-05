import torch
from torch.utils.data import Dataset
from typing import List

def list_patient_ids_fast(dataset_root, side="left", max_patients=None):
    subset_dirs = [
        d for d in os.listdir(dataset_root)
        if os.path.isdir(os.path.join(dataset_root, d))
    ]

    ids = []
    for subset in subset_dirs:
        subset_path = os.path.join(dataset_root, subset)
        for pid in os.listdir(subset_path):
            mri_side_dir = os.path.join(subset_path, pid, "mri", side)
            if os.path.isdir(mri_side_dir):
                ids.append(pid)

    if max_patients is not None:
        ids = ids[:max_patients]

    return ids

class PatientAndTabularDataset(Dataset):
    def __init__(self, root: str, clinical_csv_path: str, tabular_variable: str, side='left', split='train', max_patients=None):
        # The beginning of this function is the same as the KneeMRIDataset.

        self.root = root
        self.side = side
        self.max_patients = max_patients

        all_ids = []
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
        
        
        all_ids = list_patient_ids_fast(root, side, max_patients)

        if side == "left":
            all_ids = [pid for pid in all_ids if pid not in EXCLUDE_L]
        else:
            all_ids = [pid for pid in all_ids if pid not in EXCLUDE_R]


        random.shuffle(all_ids)
        n_total = len(all_ids)
        n_train = int(0.7 * n_total)
        n_val   = int(0.15 * n_total)

        if split == "train":
            self.patient_ids = all_ids[:n_train]
        elif split == "val":
            self.patient_ids = all_ids[n_train:n_train+n_val]
        elif split == "test":
            self.patient_ids = all_ids[n_train+n_val:]
        else:
            raise ValueError("split must be 'train', 'val', or 'test'")
        

        # Here we get the tabular data for the patients.
        
        self.tabular_variable = tabular_variable    # Save the tabular variable, for which we want to provide the dataset (for example subjective pain level).

        df_clinical = pd.read_csv(clinical_csv_path) # Load the dataframe containing patient Ids and their tabular values.

        # Convert to dict with ID as keys and all other columns as nested dicts
        self.df_clinical_dict = df_clinical.set_index('ID').to_dict('index')
        # Result: {9000099: {'V00WOMTSR': 14.0, 'V00WOMTSL': 0.0, ...}, ...}


    def __len__(self):
        return len(self.patient_ids)

    def __getitem__(self, idx):
        target_pid = self.patient_ids[idx]

        vol = load_single_patient_mri(self.root, target_pid, self.side)
        if not torch.is_tensor(vol):
            vol = torch.tensor(vol, dtype=torch.float32)

        # vol: (1, D, 224, 224)

        return (vol, self.df_clinical_dict[self.patient_ids[idx]][self.tabular_variable])   # Returns the (volume, tabular variable value) tuple to be used in training.