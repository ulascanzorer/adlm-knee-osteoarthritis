import os
import sys
import numpy as np
import torch
import torch.nn as nn

from data import iter_mri_dataset

sys.path.append(os.path.abspath("MedicalNet"))
from models.resnet import resnet50  


def run_inference(
    data_root: str,
    side: str,
    weights_path: str = "MedicalNet/pretrain/resnet_50.pth",
    max_patients: int | None = None,
    features_dir: str = "features",
) -> str:
    """
    Run 3D ResNet-50 inference on MRIs for a given side and save latent features.

    Saves:
        {features_dir}/features_{side}.npz
            - ids: np.ndarray of patient IDs (str)
            - features: np.ndarray of shape [N, 2048]

    Returns:
        Path to the saved .npz file.
    """

    # Build 3D ResNet-50 backbone
    model = resnet50(
        sample_input_D=64,
        sample_input_H=224,
        sample_input_W=224,
        num_seg_classes=1,
    )

    # Load pretrained MedicalNet weights
    checkpoint = torch.load(weights_path, map_location="cpu")
    state_dict = checkpoint.get("state_dict", checkpoint)
    model.load_state_dict(state_dict, strict=False)

    # Remove classification head -> keep feature extractor
    model = nn.Sequential(*list(model.children())[:-1])
    model.eval()

    use_cuda = torch.cuda.is_available()
    if use_cuda:
        model = model.cuda()

    print(f"[{side}] Streaming MRIs from {data_root} ...")

    patients_features: dict[str, torch.Tensor] = {}

    with torch.no_grad():
        for p_id, vol in iter_mri_dataset(
            data_root,
            side=side,
            max_patients=max_patients,
        ):
            # vol: [1, D, 224, 224] -> [1, 1, D, 224, 224]
            vol = vol.unsqueeze(0)
            if use_cuda:
                vol = vol.cuda()

            # Extract features: [1, 2048, 20, 28, 28]
            feats = model(vol)
            # Global average pooling over depth/height/width -> [1, 2048]
            feats = feats.mean(dim=(2, 3, 4))

            patients_features[p_id] = feats.cpu()
            print(f"[{side}] Processed patient {p_id}")

    if not patients_features:
        print(f"[{side}] No patients processed, nothing to save.")
        os.makedirs(features_dir, exist_ok=True)
        return os.path.join(features_dir, f"features_{side}.npz")

    # Save latent features
    os.makedirs(features_dir, exist_ok=True)
    feat_out = os.path.join(features_dir, f"features_{side}.npz")

    patient_ids = np.array(list(patients_features.keys()))
    features = np.stack(
        [
            t.detach().numpy().squeeze()
            for t in patients_features.values()
        ]
    )  # [N, 2048]

    np.savez(feat_out, ids=patient_ids, features=features)
    print(f"[{side}] Saved features to {feat_out}")

    return feat_out
