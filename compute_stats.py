import pandas as pd
from cluster import clusters_stats

tabular_path = "csv/clinical00_cleaned.csv"
clusters_path = "csv/mri_clusters.csv"  

df_clinical = pd.read_csv(tabular_path)
df_clusters = pd.read_csv(clusters_path)

df_cluster_stats = clusters_stats(df_clusters, df_clinical)

print(df_cluster_stats.to_string(index=False))