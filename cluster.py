import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

from score import (
    compute_symptoms_score,
    compute_structure_score,
    compute_surgery_score,
    compute_surgery_percentages,
    compute_kl_left_distribution,
    compute_kl_right_distribution,
)


# Predefined number of clusters which corresponds to the KL grading
def run_kmeans(patients_features, k=4):
    """
    Perform K-Means clustering on latent features and return assignments.
    """
    patient_ids = list(patients_features.keys())

    # Convert all tensors to numpy and squeeze leading singleton dims
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


def clusters_stats(df_clusters, df_clinical, side="left"):
    """
    Join clinical and cluster data and compute summary stats per cluster.
    side: "left" or "right" — controls which KL distribution is computed.
    """

    score_funcs = {
        "symptoms": compute_symptoms_score,
        "structure": compute_structure_score,
        "surgery": compute_surgery_score,
    }

    # Ensure IDs are comparable
    df_clusters["ID"] = df_clusters["ID"].astype(int)
    df_clinical["ID"] = df_clinical["ID"].astype(int)

    merged = df_clinical.merge(df_clusters, on="ID", how="inner")
    results = []

    for cluster_id, group in merged.groupby("cluster"):
        sym = score_funcs["symptoms"](group)
        struct = score_funcs["structure"](group)
        surg = score_funcs["surgery"](group)
        surgery_pct_yes, surgery_pct_no = compute_surgery_percentages(group)

        row = {
            "cluster": cluster_id,
            "SYMPTOMS_SCORE": sym,
            "STRUCTURE_SCORE": struct,
            "SURGERY_SCORE": surg,
            "N_PATIENTS": len(group),
            "SURGERY_PCT_YES": surgery_pct_yes,
            "SURGERY_PCT_NO": surgery_pct_no,
        }

        # Add KL distribution only for the requested side
        if side == "left":
            kl_counts = compute_kl_left_distribution(group)
            for grade in [0, 1, 2, 3, 4]:
                row[f"KL_LEFT_{grade}_PCT"] = kl_counts.loc[grade] / len(group)
        elif side == "right":
            kl_counts = compute_kl_right_distribution(group)
            for grade in [0, 1, 2, 3, 4]:
                row[f"KL_RIGHT_{grade}_PCT"] = kl_counts.loc[grade] / len(group)
        else:
            raise ValueError(f"Unknown side: {side}")

        results.append(row)

    return pd.DataFrame(results)