import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import numpy as np

def tsne_plot(patients_features):
    """Creates a TSNE model and plots it"""
    patient_ids = list(patients_features.keys())
    features = np.stack([
            t.detach().cpu().numpy().squeeze()  # squeeze removes (1, 2048) → (2048,)
            for t in patients_features.values()
        ])

    tsne_model = TSNE(perplexity=20, n_components=2, init='pca', max_iter=2500, random_state=23)
    new_values = tsne_model.fit_transform(features)

    x = []
    y = []
    for value in new_values:
        x.append(value[0])
        y.append(value[1])
        
    plt.figure(figsize=(16, 9)) 
    for i in range(len(x)):
        plt.scatter(x[i],y[i])
        plt.annotate(patient_ids[i],
                     xy=(x[i], y[i]),
                     xytext=(5, 2),
                     textcoords='offset points',
                     ha='right',
                     va='bottom')
    plt.show()

if __name__ == "__main__":
    # TODO: Insert an example code snippet to use tsne here.
    pass