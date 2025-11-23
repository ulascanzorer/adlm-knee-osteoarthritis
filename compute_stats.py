import argparse
import pandas as pd
from cluster import clusters_stats


def main(args):
    tabular_path = "csv/clinical00_cleaned.csv"
    clusters_path = f"csv/mri_clusters_{args.side}.csv"

    df_clinical = pd.read_csv(tabular_path)
    df_clusters = pd.read_csv(clusters_path)

    df_cluster_stats = clusters_stats(df_clusters, df_clinical, side=args.side)

    print(df_cluster_stats.to_string(index=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--side",
        type=str,
        choices=["left", "right"],
        default="left",
        help="Which knee side to compute stats for.",
    )
    args = parser.parse_args()
    main(args)