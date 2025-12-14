import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import numpy as np

def tsne_plot_with_clusters(
    patients_features,
    df_clusters,
    n_components=2,
    output_path="tsne_plot.png"
):
    """
    Runs t-SNE on patient features and colors points using existing clusters.
    Supports 1D, 2D, and 3D.
    """
    # 1. Extract patient IDs and feature tensors
    patients_features = {int(pid): feat for pid, feat in patients_features.items()}
    patient_ids = list(patients_features.keys())

    # Convert feature tensors to numpy
    features = np.stack([
        t.detach().cpu().numpy().squeeze()
        for t in patients_features.values()
    ])

    # 2. Map cluster labels to patient IDs
    id_to_cluster = dict(zip(df_clusters["ID"], df_clusters["cluster"]))
    cluster_labels = np.array([id_to_cluster[pid] for pid in patient_ids])

    unique_clusters = np.sort(np.unique(cluster_labels))
    colors = plt.cm.tab10(np.linspace(0, 1, len(unique_clusters)))

    # 3. Run t-SNE
    print(f"Running t-SNE with n_components={n_components}...")
    tsne_model = TSNE(
        perplexity=30,           
        n_components=n_components,
        init="pca",
        max_iter=2500,
        random_state=23
    )
    tsne_values = tsne_model.fit_transform(features)


    if n_components == 1:
        plt.figure(figsize=(12, 5))
        
        # Add random "Jitter" to Y-axis so dots don't overlap
        y_jitter = np.random.normal(loc=0, scale=0.05, size=len(tsne_values))

        for cluster_id in unique_clusters:
            idx = np.where(cluster_labels == cluster_id)[0]
            plt.scatter(
                tsne_values[idx, 0],   # X is the 1D t-SNE value
                y_jitter[idx],         # Y is fake noise
                c=[colors[cluster_id]],
                s=50,
                alpha=0.7,
                label=f"Cluster {cluster_id}"
            )

        plt.title("t-SNE 1D Visualization")
        plt.xlabel("Decease Score")
        plt.yticks([])  
        plt.legend(title="Clusters")
        plt.grid(axis='x', alpha=0.3)
        
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()


    elif n_components == 2:
        plt.figure(figsize=(16, 9))
        for cluster_id in unique_clusters:
            idx = np.where(cluster_labels == cluster_id)[0]
            plt.scatter(
                tsne_values[idx, 0],
                tsne_values[idx, 1],
                c=[colors[cluster_id]],
                s=50,
                alpha=0.85,
                label=f"Cluster {cluster_id}"
            )
        plt.title("t-SNE 2D Visualization")
        plt.legend(title="Clusters")
        plt.grid(alpha=0.2)
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()

    elif n_components == 3:
        fig = plt.figure(figsize=(16, 9))
        ax = fig.add_subplot(111, projection="3d")
        for cluster_id in unique_clusters:
            idx = np.where(cluster_labels == cluster_id)[0]
            ax.scatter(
                tsne_values[idx, 0],
                tsne_values[idx, 1],
                tsne_values[idx, 2],
                c=[colors[cluster_id]],
                s=50,
                alpha=0.85,
                label=f"Cluster {cluster_id}"
            )
        ax.set_title("t-SNE 3D Visualization")
        ax.legend(title="Clusters")
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()

    else:
        raise ValueError("n_components must be 1, 2, or 3")