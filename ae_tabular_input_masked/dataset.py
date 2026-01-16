# ae_tabular_input/dataset.py
import os
import sys
import random
from typing import Optional, List, Tuple, Dict, Any

import torch
from torch.utils.data import Dataset
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from ae_pain.data import list_patient_ids_fast, load_single_patient_mri


EXCLUDE_L = {"9070923", "9388265", "9594253", "9860568"}
EXCLUDE_R = {"9004315", "9537947", "9637394", "9462278", "9522128"}


TAB_INPUTS_L = [
    "V00WOMTSL", "V00KOOSKPL", "V00KOOSYML", "V00KOOSQOL",
    "V00XRJSL_L", "V00XRJSM_L",
    "V00XRSCFM_L", "V00XRSCFL_L",
    "V00XRSCTM_L", "V00XRSCTL_L",
    "V00XROSFM_L", "V00XROSFL_L",
    "V00XROSTM_L", "V00XROSTL_L",
    "V00XRKL_L",
    "V99ELKVSAF",
]

TAB_INPUTS_R = [
    "V00WOMTSR", "V00KOOSKPR", "V00KOOSYMR", "V00KOOSQOL",
    "V00XRJSL_R", "V00XRJSM_R",
    "V00XRSCFM_R", "V00XRSCFL_R",
    "V00XRSCTM_R", "V00XRSCTL_R",
    "V00XROSFM_R", "V00XROSFL_R",
    "V00XROSTM_R", "V00XROSTL_R",
    "V00XRKL_R",
    "V99ERKVSAF",
]

TARGETS_L = ["V00WOMTSL", "V00XRJSM_L", "V99ELKVSAF"]
TARGETS_R = ["V00WOMTSR", "V00XRJSM_R", "V99ERKVSAF"]


# -------------------------
# Normalization
# -------------------------

def normalize_tab(col: str, raw: float) -> float:
    if col.startswith(("V00WOMTS", "V00KOOS")):
        return raw / 100.0
    if col.startswith("V00XRKL"):
        return raw / 4.0
    if col.startswith("V00XR"):
        return raw / 3.0
    if col.startswith("V99"):
        return raw
    raise ValueError(f"Unknown column for normalization: {col}")


def is_missing(v: Any) -> bool:
    if v is None:
        return True
    if isinstance(v, str):
        return v.strip() == ""
    return bool(pd.isna(v))


class KneeMRITabularDataset(Dataset):
    def __init__(
        self,
        root: str,
        clinical_csv_path: str,
        side: str = "left",
        split: str = "train",
        max_patients: Optional[int] = None,
        train_frac: float = 0.7,
        val_frac: float = 0.15,
        seed: int = 42,
        require_all_targets: bool = False,
    ):
        super().__init__()

        self.root = root
        self.side = side

        if side == "left":
            self.tab_inputs = TAB_INPUTS_L
            self.targets = TARGETS_L
            exclude = EXCLUDE_L
        elif side == "right":
            self.tab_inputs = TAB_INPUTS_R
            self.targets = TARGETS_R
            exclude = EXCLUDE_R
        else:
            raise ValueError("side must be 'left' or 'right'")

        all_ids = list_patient_ids_fast(root, side, max_patients)
        all_ids = [str(pid) for pid in all_ids if str(pid) not in exclude]

        df = pd.read_csv(clinical_csv_path)
        df.columns = df.columns.str.strip()
        df["ID"] = df["ID"].astype(str)

        needed_cols = {"ID"} | set(self.tab_inputs) | set(self.targets)
        missing_cols = [c for c in needed_cols if c not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing columns in CSV: {missing_cols}")

        valid_ids = set(df["ID"].tolist())
        all_ids = [pid for pid in all_ids if pid in valid_ids]

        if require_all_targets:
            sub = df.set_index("ID")
            keep = []
            for pid in all_ids:
                row = sub.loc[pid]
                if isinstance(row, pd.DataFrame):
                    row = row.iloc[0]
                if all(not is_missing(row[t]) for t in self.targets):
                    keep.append(pid)
            all_ids = keep

        rng = random.Random(seed)
        rng.shuffle(all_ids)

        n_total = len(all_ids)
        n_train = int(train_frac * n_total)
        n_val = int(val_frac * n_total)

        if split == "train":
            self.patient_ids = all_ids[:n_train]
        elif split == "val":
            self.patient_ids = all_ids[n_train:n_train + n_val]
        elif split == "test":
            self.patient_ids = all_ids[n_train + n_val:]
        else:
            raise ValueError("split must be 'train', 'val', or 'test'")

        self.clinical = df.set_index("ID").to_dict("index")

    def __len__(self) -> int:
        return len(self.patient_ids)

    def _load_mri(self, pid: str) -> Optional[torch.Tensor]:
        vol = load_single_patient_mri(self.root, pid, self.side)
        if vol is None:
            return None
        if isinstance(vol, tuple):
            vol = vol[1]
        if not torch.is_tensor(vol):
            return None

        vol = vol.float()
        vmin, vmax = vol.min(), vol.max()
        if vmax > 5.0:
            vol = vol - vmin
            if vol.max() > 0:
                vol = vol / vol.max()
            vol = vol * 2.0 - 1.0
        elif vmin >= 0.0 and vmax <= 1.0:
            vol = vol * 2.0 - 1.0
        return vol

    def _build_tab(self, pid: str, cols: List[str]) -> Tuple[torch.Tensor, torch.Tensor]:
        row = self.clinical[pid]
        vals, mask = [], []

        for c in cols:
            v = row[c]
            if is_missing(v):
                vals.append(0.0)
                mask.append(0.0)
            else:
                vals.append(normalize_tab(c, float(v)))
                mask.append(1.0)

        return (
            torch.tensor(vals, dtype=torch.float32),
            torch.tensor(mask, dtype=torch.float32),
        )

    def __getitem__(self, idx: int):
        retries = 0
        while retries < 10:
            pid = self.patient_ids[idx]
            vol = self._load_mri(pid)
            if vol is None:
                idx = random.randint(0, len(self.patient_ids) - 1)
                retries += 1
                continue

            tab_x, tab_mask = self._build_tab(pid, self.tab_inputs)
            y, y_mask = self._build_tab(pid, self.targets)

            return vol, tab_x, tab_mask, y, y_mask

        raise RuntimeError("Failed to load valid sample after retries")
