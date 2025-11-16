from torchvision import transforms
from PIL import Image
import numpy as np
import os
import torch
from collections import defaultdict

transform = transforms.Compose([
    transforms.Resize((224, 224)),     # standard RadImageNet input size
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]) #normalisation of the ImageNet dataset
                                                    # as RadImageNet is based on ImageNet
])

def preprocess_mri_image(img_path):
    img = Image.open(img_path).convert('L')  # makes sure you are working with grayscale
                                             # in case the image has an alpha channel
    img = np.stack((np.array(img),)*3, axis=-1)  # now the image has one channel
                                                 # radimagenet takes 3 channel images as input so we triplicate the input on
                                                 # all 3 channels
    img = Image.fromarray(img.astype(np.uint8)) #we cast the object to uint values in the [0-255] range
                                                # and we conveer the array back to a pil image
    return transform(img).unsqueeze(0) #finally we can apply the transformations we need
    # i have a [1,3,224,224] tensor now

def load_mri_folder(folder_path):
    """
    Recursively loads all MRI images from a folder structure like:
    data/mri_only/mri_only/{subset}/{patient_id}/...
    """
    patient_tensors = defaultdict(list)

    for root, _, files in os.walk(folder_path):
        #os.walk traverses all the subdirectories in folder_path
        for f in files:
            path = os.path.join(root, f)
            # Extract patient ID = last numeric folder in the path
            parts = path.split(os.sep)
            patient_id = next((p for p in parts if p.isdigit()), None)
            try:
                tensor = preprocess_mri_image(path)
                patient_tensors[patient_id].append(tensor)
            except Exception as e:
                print(f"Skipping {path} due to error: {e}")

                
    #so now i have a dictionary where keys are patient ids and values are lists of tensors
    #"9000099": [tensor_1, tensor_2, tensor_3],
    #"9000456": [tensor_4, tensor_5],
    for p_id in list(patient_tensors.keys()):
        patient_tensors[p_id] = torch.cat(patient_tensors[p_id], dim=0)
    #once i am done with this loop i have a dictionary where keys are patient ids and values are
    #"9000099": tensor of shape [n,3,224,224] where n is the number of MRIs for that patient
    
    return patient_tensors