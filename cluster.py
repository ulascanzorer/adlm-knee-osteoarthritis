import torch
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from score import compute_symptoms_score, compute_structure_score, compute_surgery_score

    
#we need to find the optimal number of clusters k for k-means clustering
#should run the silhouette analysis to find the optimal k
#TODO


# Predefined number of clusters which corresponds to the KL grading
def run_kmeans(patients_features, k=4):
    """
    Perform K-Means clustering on latent features and return assignments.
    """
    patient_ids = list(patients_features.keys())
    features = np.stack([
            t.detach().cpu().numpy().squeeze()  # squeeze removes (1, 2048) → (2048,)
            for t in patients_features.values()
        ])

    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(features)

    df_clusters = pd.DataFrame({
        "ID": patient_ids,
        "cluster": labels
    })
    print(f"✅ K-Means clustering complete. Found {k} clusters.")
    return df_clusters



def clusters_stats(df_clusters, df_clinical):

    score_funcs = {
        "symptoms": compute_symptoms_score,
        "structure": compute_structure_score,
        "surgery": compute_surgery_score
    }
    df_clusters["ID"] = df_clusters["ID"].astype(int)
    df_clinical["ID"] = df_clinical["ID"].astype(int)
    
    merged = df_clinical.merge(df_clusters, on="ID", how="inner")
    results = []

    for cluster_id, group in merged.groupby("cluster"):
        sym = score_funcs["symptoms"](group)
        struct = score_funcs["structure"](group)
        surg = score_funcs["surgery"](group)
        results.append({
            "cluster": cluster_id,
            "SYMPTOMS_SCORE": sym,
            "STRUCTURE_SCORE": struct,
            "SURGERY_SCORE": surg,
            "N_PATIENTS": len(group)
        })

    return pd.DataFrame(results)
