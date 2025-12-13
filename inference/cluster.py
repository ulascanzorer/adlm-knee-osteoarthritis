import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from score import (
    compute_surgery_percentages,
    compute_kl_distribution,
    compute_all_statistics
)


# Predefined number of clusters which corresponds to the KL grading
def run_kmeans(patients_features, k: int = 5) -> pd.DataFrame:
    """
    Perform K-Means clustering on latent features and return assignments.

    patients_features: dict[patient_id -> torch.Tensor or np.ndarray]
        Each value should be a 1D feature vector (e.g. shape (C,) or (1, C)).
    """
    patient_ids = list(patients_features.keys())

    # Convert all tensors/arrays to numpy and squeeze leading singleton dims
    features = np.stack(
        [
            (
                t.detach().cpu().numpy().squeeze()
                if hasattr(t, "detach")
                else np.asarray(t).squeeze()
            )
            for t in patients_features.values()
        ]
    )

    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(features)

    df_clusters = pd.DataFrame(
        {
            "ID": patient_ids,
            "cluster": labels,
        }
    )
    print(f" K-Means clustering complete. Found {k} clusters.")
    return df_clusters


def clusters_stats(
    df_clusters: pd.DataFrame,
    df_clinical: pd.DataFrame,
    side: str = "left"
) -> pd.DataFrame:
    """
    Merge clinical + cluster assignments and compute statistics per cluster.
    side: 'left' or 'right'
    """

    # Ensure IDs are aligned
    df_clusters["ID"] = df_clusters["ID"].astype(int)
    df_clinical["ID"] = df_clinical["ID"].astype(int)

    merged = df_clinical.merge(df_clusters, on="ID", how="inner")
    results = []

    for cluster_id, group in merged.groupby("cluster"):

        # Initialize base row
        row = {
            "cluster": cluster_id,
            "N_PATIENTS": len(group),
        }

        # Surgery summary (% of yes / no)
        surg_yes, surg_no = compute_surgery_percentages(group, side)
        row["SURGERY_PCT_YES"] = surg_yes
        row["SURGERY_PCT_NO"] = surg_no

        # KL distribution for this side
        kl_counts = compute_kl_distribution(group, side)

        prefix = f"KL_{side.upper()}"
        for grade in [0, 1, 2, 3, 4]:
            row[f"{prefix}_{grade}_PCT"] = kl_counts.loc[grade] / len(group)

        # ALL_L / ALL_R stats
        all_stats = compute_all_statistics(group, side)

        for subgroup, stats_dict in all_stats.items():
            for col, val in stats_dict.items():
                colname = f"{col}_MEAN"
                row[colname] = val

        # Save this cluster row
        results.append(row)

    return pd.DataFrame(results)
