import argparse
import torch
import torch.nn as nn
import pandas as pd
import sys,os
from data import iter_mri_dataset
from cluster import run_kmeans
from tsne_visualization import tsne_plot, tsne_plot_with_k_means_clustering

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

    print(f"Streaming {args.side} MRIs from {args.data_root} ...")

    patients_features = {}
    with torch.no_grad():
        for i, (p_id, vol) in enumerate(
            iter_mri_dataset(
                args.data_root,
                side=args.side,
                max_patients=args.max_patients,
            )
        ):
            # Move to GPU if available
            vol = vol.unsqueeze(0)  # [1, 1, D, 224, 224]
            if torch.cuda.is_available():
                model = model.cuda()
                vol = vol.cuda()

            # Extract features
            feats = model(vol)  # [1, 2048, 20, 28, 28]
            feats = feats.mean(dim=(2, 3, 4))  # Global average pooling over depth + height + width -> [1, 2048]
            patients_features[p_id] = feats.cpu()
            print(f"Processed patient {p_id}")

    if not patients_features:
        print("No patients processed, nothing to cluster.")
        return

    # Combine and cluster
    all_patient_features = torch.cat(list(patients_features.values()), dim=0)
    print(f"Final feature tensor shape: {all_patient_features.shape}")

    df_clusters = run_kmeans(patients_features, k=4)

    # Also perform tsne visualization here.
    tsne_plot_with_k_means_clustering(patients_features, n_components=3)

    os.makedirs("csv", exist_ok=True)
    out_path = os.path.join("csv", f"mri_clusters_{args.side}.csv")
    df_clusters.to_csv(out_path, index=False)
    print(f"Saved cluster assignments to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_root",
        type=str,
        default="/vol/miltank/projects/practical_wise2526/knee-osteoarthritis-severity/data/cleaned_images_baseline",
        help="Root folder containing 0.C.2 and 0.E.1 directories",
    )
    parser.add_argument(
        "--side",
        type=str,
        default="left",
        choices=["left", "right"],
        help="Knee side to process (default: left)",
    )
    parser.add_argument(
        "--weights_path",
        type=str,
        default="MedicalNet/pretrain/resnet_50.pth",
        help="Path to MedicalNet pretrained weights",
    )
    parser.add_argument(
        "--max_patients",
        type=int,
        default=None,
        help="Optional limit on number of patients to process",
    )
    args = parser.parse_args()

    main(args)
