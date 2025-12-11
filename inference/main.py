import argparse
import os
import sys

import numpy as np
import pandas as pd
import torch

from infer import run_inference
from cluster import run_kmeans, clusters_stats
from tsne_visualization import tsne_plot_with_clusters

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.append(project_root)


def load_features_npz(features_dir: str, side: str) -> tuple[np.ndarray, np.ndarray]:
    """
    Load features_{side}.npz and return (ids, features).
    """
    feat_path = os.path.join(features_dir, f"features_{side}.npz")
    if not os.path.exists(feat_path):
        raise FileNotFoundError(f"Features file not found for side={side}: {feat_path}")

    data = np.load(feat_path, allow_pickle=True)
    ids = data["ids"]
    features = data["features"]
    return ids, features


def run_clustering(features_dir: str, csv_dir: str, side: str, k: int = 5) -> None:
    """
    Load features for a side, run KMeans, and save mri_clusters_{side}.csv.
    """
    ids, features = load_features_npz(features_dir, side)

    # cluster.run_kmeans expects a dict of patient_id -> tensor/ndarray
    patients_features = {str(pid): features[i] for i, pid in enumerate(ids)}

    df_clusters = run_kmeans(patients_features, k=k)

    os.makedirs(csv_dir, exist_ok=True)
    out_path = os.path.join(csv_dir, f"mri_clusters_{side}.csv")
    df_clusters.to_csv(out_path, index=False)
    print(f"[{side}] Saved cluster assignments to {out_path}")
    return df_clusters


def run_stats(csv_dir: str, clinical_csv: str, side: str) -> None:
    """
    Load clinical and cluster CSVs, compute stats for a side, and print + save them.
    """
    clusters_path = os.path.join(csv_dir, f"mri_clusters_{side}.csv")
    if not os.path.exists(clusters_path):
        raise FileNotFoundError(
            f"Cluster CSV not found for side={side}: {clusters_path}"
        )

    df_clinical = pd.read_csv(clinical_csv)
    df_clusters = pd.read_csv(clusters_path)

    df_stats = clusters_stats(df_clusters, df_clinical, side=side)
    print(f"\n[{side}] Cluster stats:\n")
    print(df_stats.to_string(index=False))

    stats_path = os.path.join(csv_dir, f"cluster_stats_{side}.csv")
    df_stats.to_csv(stats_path, index=False)
    print(f"[{side}] Saved cluster stats to {stats_path}")


def run_tsne(features_dir: str, csv_dir: str, plots_dir: str, side: str,
             n_components: int = 3) -> None:
    """
    Load features and cluster assignments for a side,
    then run t-SNE visualization colored by existing clusters.
    """

    # Load features
    ids, features = load_features_npz(features_dir, side)

    # Load cluster CSV
    clusters_path = os.path.join(csv_dir, f"mri_clusters_{side}.csv")
    if not os.path.exists(clusters_path):
        raise FileNotFoundError(f"Cluster CSV not found for side={side}: {clusters_path}")

    df_clusters = pd.read_csv(clusters_path)

    # Build dict: patient_id → torch tensor
    patients_features = {
        str(pid): torch.from_numpy(features[i]).unsqueeze(0)
        for i, pid in enumerate(ids)
    }

    # Output path
    os.makedirs(plots_dir, exist_ok=True)
    out_path = os.path.join(plots_dir, f"tsne_{side}_{n_components}d.png")

    print(f"[{side}] Running t-SNE with existing clusters...")

    tsne_plot_with_clusters(
        patients_features=patients_features,
        df_clusters=df_clusters,
        n_components=n_components,
        output_path=out_path,
    )

    print(f"[{side}] Saved t-SNE plot to {out_path}")




def main() -> None:
    parser = argparse.ArgumentParser(
        description="Full MRI pipeline: inference, clustering, stats, t-SNE"
    )
    parser.add_argument(
        "--data_root",
        type=str,
        default="/vol/miltank/projects/practical_wise2526/knee-osteoarthritis-severity/data/cleaned_images_baseline",
        help="Root folder containing 0.C.2 and 0.E.1 directories",
    )
    parser.add_argument(
        "--weights_path",
        type=str,
        default=None,
        help="Path to model weights. If omitted, a model-specific default is used.",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        choices=["resnet50", "autoencoder","autoencoder_pain"],
        default="resnet50",
        help="Which feature extractor to use.",
    )
    parser.add_argument(
        "--side",
        type=str,
        choices=["left", "right", "both"],
        default="both",
        help="Which side(s) to process",
    )
    parser.add_argument(
        "--max_patients",
        type=int,
        default=None,
        help="Optional limit on number of patients to process.",
    )
    parser.add_argument(
        "--clinical_csv",
        type=str,
        default="csv/clinical00_cleaned.csv",
        help="Path to cleaned clinical CSV.",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=5,
        help="Number of clusters for KMeans.",
    )
    parser.add_argument(
        "--tsne_components",
        type=int,
        choices=[2, 3],
        default=3,
        help="Number of t-SNE dimensions.",
    )
    
    args = parser.parse_args()

    # Choose default weights if none provided
    if args.weights_path is None:
        if args.model_name == "resnet50":
            weights_path = "MedicalNet/pretrain/resnet_50.pth"
        else:  # autoencoder
            weights_path = "trained_knee_3d_autoencoder.pth"
    else:
        weights_path = args.weights_path

    print(f"Using model: {args.model_name}")
    print(f"Weights: {weights_path}")

    # Base results dir per model name
    results_dir = os.path.join("results", args.model_name)

    # Subdirs for different artifact types
    features_dir = os.path.join(results_dir, "features")
    csv_dir = os.path.join(results_dir, "csv")
    plots_dir = os.path.join(results_dir, "plots")

    os.makedirs(results_dir, exist_ok=True)

    if args.side == "both":
        sides = ["left", "right"]
    else:
        sides = [args.side]

    for side in sides:
        print(f"\n=== SIDE: {side} ===")

        # 1) Inference → save features_<side>.npz
        run_inference(
            data_root=args.data_root,
            side=side,
            weights_path=weights_path,
            max_patients=args.max_patients,
            features_dir=features_dir,
            model_name=args.model_name,
        )

        # 2) Clustering → mri_clusters_<side>.csv
        run_clustering(
            features_dir=features_dir,
            csv_dir=csv_dir,
            side=side,
            k=args.k,
        )

        # 3) Stats → cluster_stats_<side>.csv
        run_stats(
            csv_dir=csv_dir,
            clinical_csv=args.clinical_csv,
            side=side,
        )

        # 4) t-SNE → tsne_<side>_<tsne_components>d.png
        run_tsne(
            features_dir=features_dir,
            csv_dir=csv_dir,
            plots_dir=plots_dir,
            side=side,
            n_components=args.tsne_components,
        )



if __name__ == "__main__":
    main()
