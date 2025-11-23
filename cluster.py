import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

from score import (
    compute_symptoms_score,
    compute_structure_score,
    compute_surgery_percentages,
    compute_kl_distribution,
)


# Predefined number of clusters which corresponds to the KL grading
def run_kmeans(patients_features, k: int = 4) -> pd.DataFrame:
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
    print(f"✅ K-Means clustering complete. Found {k} clusters.")
    return df_clusters


def clusters_stats(df_clusters: pd.DataFrame,
                   df_clinical: pd.DataFrame,
                   side: str = "left") -> pd.DataFrame:
    """
    Join clinical and cluster data and compute summary stats per cluster.

    side: "left" or "right" — controls which side-specific clinical variables
    and KL distribution are used.
    """

    # Ensure IDs are comparable
    df_clusters["ID"] = df_clusters["ID"].astype(int)
    df_clinical["ID"] = df_clinical["ID"].astype(int)

    merged = df_clinical.merge(df_clusters, on="ID", how="inner")
    results = []

    for cluster_id, group in merged.groupby("cluster"):
        sym = compute_symptoms_score(group, side=side)
        struct = compute_structure_score(group, side=side)
        surgery_pct_yes, surgery_pct_no = compute_surgery_percentages(group, side=side)

        row = {
            "cluster": cluster_id,
            "SYMPTOMS_SCORE": sym,
            "STRUCTURE_SCORE": struct,
            "N_PATIENTS": len(group),
            "SURGERY_PCT_YES": surgery_pct_yes,
            "SURGERY_PCT_NO": surgery_pct_no,
        }

        # KL distribution for the requested side
        kl_counts = compute_kl_distribution(group, side=side)

        if side == "left":
            for grade in [0, 1, 2, 3, 4]:
                row[f"KL_LEFT_{grade}_PCT"] = kl_counts.loc[grade] / len(group)
        elif side == "right":
            for grade in [0, 1, 2, 3, 4]:
                row[f"KL_RIGHT_{grade}_PCT"] = kl_counts.loc[grade] / len(group)
        else:
            raise ValueError(f"Unknown side: {side}")

        results.append(row)

    return pd.DataFrame(results)