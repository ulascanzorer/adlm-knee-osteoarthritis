import torch
import torch.nn as nn
from torchvision import models
from data import load_mri_folder
from cluster import run_kmeans, clusters_stats
import pandas as pd
# Path to your weights
weights_path = "RadImageNet_weights/ResNet50.pt"
tabular_path="data/clinical00_cleaned.csv"

# Load model
state_dict = torch.load(weights_path, map_location="cpu")
clean_state_dict = {k.replace("backbone.", ""): v for k, v in state_dict.items()}

model = models.resnet50(weights=None)
# missing, unexpected = model.load_state_dict(clean_state_dict, strict=False)
# print("Missing keys:", missing)
# print("Unexpected keys:", unexpected)
# Remove classification head
model = nn.Sequential(*list(model.children())[:-1])
model.eval()

# Process an entire folder of MRIs
data_folder = "data/mri_test/mri_only"
print(f"Loading MRIs from {data_folder}...")
patients_tensors = load_mri_folder(data_folder)

patients_features = {}

# Extract features for each MRI
features_list = []
with torch.no_grad():
    for p_id, imgs in patients_tensors.items():
        feats = model(imgs) # (n_mri, 2048, 1, 1)
        print(f"model output", feats.shape)
        #torch.view reshapes the tensor to have the desired shape with the same number of elements
        #so putting -1 lets him infer the other dimension size automatically
        feats = feats.view(feats.size(0), -1) # (n_mri, 2048)
        print(f"output reshaped",feats.shape)
        features_avg=feats.mean(dim=0, keepdim=True)
        print(f"patient features averaged dims", features_avg.shape)
        patients_features[p_id] = features_avg  # (1, 2048)        
        print(f"Processing {len(patients_tensors)} patients")

all_patient_features = torch.cat(list(patients_features.values()), dim=0)
print(f"Final feature tensor shape: {all_patient_features.shape}") #(num_patients, 2048)


#now we can run k-means clustering on the extracted features
df_clusters = run_kmeans(patients_features, k=4)
#then we can compute the cluster statistics
df_clinical = pd.read_csv(tabular_path)
df_cluster_stats = clusters_stats(df_clusters, df_clinical)