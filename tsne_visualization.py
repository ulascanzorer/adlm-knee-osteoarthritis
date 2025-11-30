import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import numpy as np

def tsne_plot(patients_features, n_components=2, output_path='tsne_plot.png'):
    """Creates a TSNE model and plots it in 2D or 3D"""
    patient_ids = list(patients_features.keys())
    features = np.stack([
        t.detach().cpu().numpy().squeeze() 
        for t in patients_features.values()
    ])
    tsne_model = TSNE(perplexity=20, n_components=n_components, init='pca', max_iter=2500, random_state=23)
    new_values = tsne_model.fit_transform(features)
    
    if n_components == 2:
        # 2D plot
        x = []
        y = []
        for value in new_values:
            x.append(value[0])
            y.append(value[1])
        
        plt.figure(figsize=(16, 9))
        plt.scatter(x, y, s=20)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
    elif n_components == 3:
        # 3D plot
        x = []
        y = []
        z = []
        for value in new_values:
            x.append(value[0])
            y.append(value[1])
            z.append(value[2])
        
        fig = plt.figure(figsize=(16, 9))
        ax = fig.add_subplot(111, projection='3d')
        
        ax.scatter(x, y, z, s=20)
        
        ax.set_xlabel('Component 1')
        ax.set_ylabel('Component 2')
        ax.set_zlabel('Component 3')
        
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    else:
        raise ValueError(f"n_components must be 2 or 3, got {n_components}")

def tsne_plot_with_k_means_clustering(patients_features, n_components=2, n_clusters=4, output_path='tsne_plot.png'):
    """Creates a TSNE model and plots it in 2D or 3D with clustering"""
    from sklearn.cluster import KMeans
    
    patient_ids = list(patients_features.keys())
    features = np.stack([
        t.detach().cpu().numpy().squeeze()  
        for t in patients_features.values()
    ])
    tsne_model = TSNE(perplexity=20, n_components=n_components, init='pca', max_iter=2500, random_state=23)
    new_values = tsne_model.fit_transform(features)
    
    # Perform clustering on the t-SNE results
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    cluster_labels = kmeans.fit_predict(new_values)
    
    # Create a color map
    colors = plt.cm.tab10(np.linspace(0, 1, n_clusters))
    
    if n_components == 2:
        # 2D plot
        x = []
        y = []
        for value in new_values:
            x.append(value[0])
            y.append(value[1])
        
        plt.figure(figsize=(16, 9))
        for i in range(len(x)):
            cluster_id = cluster_labels[i]
            plt.scatter(x[i], y[i], c=[colors[cluster_id]], s=20)
        plt.title(f't-SNE 2D Visualization ({n_clusters} clusters)')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
    elif n_components == 3:
        # 3D plot
        x = []
        y = []
        z = []
        for value in new_values:
            x.append(value[0])
            y.append(value[1])
            z.append(value[2])
        
        fig = plt.figure(figsize=(16, 9))
        ax = fig.add_subplot(111, projection='3d')
        
        for i in range(len(x)):
            cluster_id = cluster_labels[i]
            ax.scatter(x[i], y[i], z[i], c=[colors[cluster_id]], s=20)
        
        ax.set_xlabel('Component 1')
        ax.set_ylabel('Component 2')
        ax.set_zlabel('Component 3')
        ax.set_title(f't-SNE 3D Visualization ({n_clusters} clusters)')
        
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    else:
        raise ValueError(f"n_components must be 2 or 3, got {n_components}")