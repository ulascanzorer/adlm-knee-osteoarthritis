import argparse
import torch
import torch.nn as nn
import pandas as pd
import sys,os
from data import load_mri_dataset
from cluster import run_kmeans

sys.path.append(os.path.abspath("MedicalNet"))
from models.resnet import resnet50
# Import MedicalNet model builder


def main(args):
    # Build 3D ResNet-50 backbone
    model = resnet50(sample_input_D=64, sample_input_H=224, sample_input_W=224, num_seg_classes=1)

    # Load pretrained MedicalNet weights
    checkpoint = torch.load(args.weights_path, map_location="cpu")
    state_dict = checkpoint.get("state_dict", checkpoint)
    model.load_state_dict(state_dict, strict=False)
    model = nn.Sequential(*list(model.children())[:-1])  # remove classification head
    model.eval()

    # Load dataset
    print(f"Loading {args.side} MRIs from {args.data_root} ...")
    patients_tensors = load_mri_dataset(args.data_root, side=args.side)

    patients_features = {}
    with torch.no_grad():
        for p_id, vol in patients_tensors.items():
            # Move to GPU if available
            vol = vol.unsqueeze(0)  # [1, 1, D, 224, 224]
            if torch.cuda.is_available():
                model = model.cuda()
                vol = vol.cuda()

            # Extract features
            feats = model(vol)  # [1, 2048, 1, 1, 1]
            feats = feats.view(feats.size(0), -1)  # flatten to [1, 2048]
            patients_features[p_id] = feats.cpu()
            print(f"Processed patient {p_id}")

    # Combine and cluster
    all_patient_features = torch.cat(list(patients_features.values()), dim=0)
    print(f"Final feature tensor shape: {all_patient_features.shape}")

    df_clusters = run_kmeans(patients_features, k=4)
    df_clusters.to_csv("csv/mri_clusters.csv", index=False)
    print("Saved cluster assignments to csv/mri_clusters_3d.csv")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, default="cleaned_images_baseline",
                        help="Root folder containing 0.C.2 and 0.E.1 directories")
    parser.add_argument("--side", type=str, default="left", choices=["left", "right"],
                        help="Knee side to process (default: left)")
    parser.add_argument("--weights_path", type=str, default="MedicalNet_weigths/resnet_50.pth",
                        help="Path to MedicalNet pretrained weights")
    args = parser.parse_args()

    main(args)
