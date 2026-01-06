import os
import sys
import numpy as np
import torch
import torch.nn as nn

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from tabular_input_inference.data import iter_mri_dataset_with_tabular_variable


def build_feature_extractor(
    weights_path: str,
    device: torch.device,
) -> nn.Module:
    """
    Return a model that, given vol [B, 1, D, H, W], outputs an embedding [B, F].
    All model-specific stuff stays in here.
    """    
    from ae_tabular_input.model import build_tabular_input_ae

    # You can adjust latent_channels / num_tabular_outputs as needed
    model = build_tabular_input_ae(
        pretrained_ae_path=None,
        device=device,
        latent_channels=64,
        num_tabular_outputs=1,
    )

    state_dict = torch.load(weights_path, map_location=device)
    model.load_state_dict(state_dict)

    model = model.to(device)
    model.eval()
    return model


def extract_features(
    model: nn.Module,
    vol: torch.Tensor,
    tabular_variable: torch.Tensor,
) -> torch.Tensor:
    """
    Forward one volume and the tabular variable through the model and return a [1, F] embedding tensor.
    """
    encoded_mri = model.encoder(vol)
    encoded_tabular = model.tabular_encoder(tabular_variable)

    # Get spatial dimensions from encoded MRI.
    B, C, D, H, W = encoded_mri.shape
    
    # Reshape and broadcast tabular to match spatial dimensions.
    # (B, tabular_dim) -> (B, tabular_dim, 1, 1, 1) -> (B, tabular_dim, D', H', W')
    encoded_tabular = encoded_tabular.view(B, -1, 1, 1, 1)
    encoded_tabular = encoded_tabular.expand(-1, -1, D, H, W)
    
    # Now concatenate along channel dimension.
    z = torch.cat([encoded_mri, encoded_tabular], dim=1)  # (B, C+tabular_dim, D', H', W')

    # Project back to 64 channels for decoder.
    z = model.channel_projection(z)  # Shape of z is (B, 64, D', H', W').

    z = z.mean(dim=(2, 3, 4))   # Shape of z is (B, 64).
    return z


def run_inference(
    data_root: str,
    clinical_csv_path: str,
    tabular_variable: str,
    side: str,
    weights_path: str = "MedicalNet/pretrain/resnet_50.pth",
    max_patients: int | None = None,
    features_dir: str = "features",
) -> str:
    """
    Run inference on MRIs for a given side combined with a tabular variable using a feature extractor and save latent features.

    Saves:
        {features_dir}/features_{side}.npz
            - ids: np.ndarray of patient IDs (str)
            - features: np.ndarray of shape [N, F] where F depends on the model.

    Returns:
        Path to the saved .npz file.
    """
    
    # Build feature extractor.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_feature_extractor(
        weights_path=weights_path,
        device=device,
    )   # NOTE: This model takes both the mri image and a single scalar tabular value (for example KOOS pain score) as input.

    print(f"[{side}] Streaming MRIs and tabular variable {tabular_variable} from {data_root} ...")

    patients_features: dict[str, torch.Tensor] = {}
    processed = 0

    with torch.no_grad():
        for p_id, vol, tabular_value in iter_mri_dataset_with_tabular_variable(
            data_root,
            side=side,
            max_patients=max_patients,
            tabular_variable=tabular_variable,
            clinical_csv_path=clinical_csv_path,
        ):
            # vol: [1, D, 224, 224] -> [1, 1, D, 224, 224]
            vol = vol.unsqueeze(0).to(device)

            # Generic feature extraction for the chosen model -> [1, F]
            feats = extract_features(
                model=model,
                vol=vol,
                tabular_variable=tabular_value,
            )

            patients_features[p_id] = feats.cpu()
            processed += 1
            print(f"[{side}] Processed {processed} patients (last ID: {p_id})")

    if not patients_features:
        print(f"[{side}] No patients processed, nothing to save.")
        os.makedirs(features_dir, exist_ok=True)
        return os.path.join(features_dir, f"features_{side}.npz")

    # Save latent features.
    os.makedirs(features_dir, exist_ok=True)
    feat_out = os.path.join(features_dir, f"features_{side}.npz")

    patient_ids = np.array(list(patients_features.keys()))
    features = np.stack(
        [t.detach().numpy().squeeze() for t in patients_features.values()]
    )  # [N, F]

    np.savez(feat_out, ids=patient_ids, features=features)
    print(f"[{side}] Saved features to {feat_out}")

    return feat_out
