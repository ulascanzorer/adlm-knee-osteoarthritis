import os
import tarfile
import tempfile
import shutil
import numpy as np
import torch
import pydicom
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

# standard normalization for MedicalNet input
transform_2d = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])  # 3D MedicalNet models expect normalized grayscale volumes
])

def reconstruct_mri_from_tar(tar_path):
    """
    Opens a .tar.gz MRI archive, extracts all DICOM slices, reconstructs
    the 3D volume, normalizes intensities, resizes slices to 224×224,
    and returns a torch tensor of shape [1, D, 224, 224].
    """

    temp_dir = tempfile.mkdtemp()
    try:
        #Extract DICOMs
        with tarfile.open(tar_path, "r:gz") as tar:
            tar.extractall(path=temp_dir)

        # Collect DICOM files
        dicom_files = []
        for root, _, files in os.walk(temp_dir):
            for f in files:
                dicom_files.append(os.path.join(root, f))
        # Read and sort by InstanceNumber / SliceLocation
        dicoms = [pydicom.dcmread(f) for f in dicom_files]
        dicoms.sort(key=lambda d: getattr(d, "InstanceNumber", getattr(d, "SliceLocation", 0)))

        # Stack into 3D volume (D, H, W) where D is the number of slices and H,W are height and width of each slice
        volume = np.stack([d.pixel_array for d in dicoms], axis=0).astype(np.float32)

        # Normalize intensities these make sure every 
        # pixel value is in [0, 1] range and every slice matches the model’s expected resolution.
        volume -= volume.min()
        if volume.max() > 0:
            volume /= volume.max()

        # Resize each slice to 224×224
        resized_slices = []
        for i in range(volume.shape[0]):
            img = Image.fromarray((volume[i] * 255).astype(np.uint8))
            img = transform_2d.transforms[0](img)  # only resize here
            resized_slices.append(np.array(img))
        volume = np.stack(resized_slices, axis=0)

        # Convert to torch tensor [1, D, 224, 224]
        volume = torch.tensor(volume, dtype=torch.float32)
        volume = volume.unsqueeze(0)  # add channel dim

        # Normalize to [-1, 1] for consistency with MedicalNet
        volume = (volume - 0.5) / 0.5
        return volume  # [1, D, 224, 224]

    except Exception as e:
        print(f"Error reconstructing {tar_path}: {e}")
        return None

    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)



def iter_mri_dataset(dataset_root, side="left", max_patients=None):
    """
    Generator that yields (patient_id, tensor[1, D, 224, 224])
    with a progress bar showing how many patients are being loaded.
    """

    # Get subset directories
    subset_dirs = [
        d for d in os.listdir(dataset_root)
        if os.path.isdir(os.path.join(dataset_root, d))
    ]

    # First collect ALL patient paths (so tqdm knows total length)
    all_patients = []
    for subset in subset_dirs:
        subset_path = os.path.join(dataset_root, subset)
        for patient_id in os.listdir(subset_path):
            all_patients.append((subset, patient_id))

    # If max_patients requested, slice the list
    if max_patients is not None:
        all_patients = all_patients[:max_patients]

    # Build progress bar
    pbar = tqdm(all_patients, desc=f"Loading {side} MRI volumes", unit="patient")

    count = 0

    # Iterate through patients with a progress bar
    for subset, patient_id in pbar:
        subset_path = os.path.join(dataset_root, subset)
        patient_path = os.path.join(subset_path, patient_id)
        mri_side_dir = os.path.join(patient_path, "mri", side)

        if not os.path.isdir(mri_side_dir):
            continue

        tar_files = [f for f in os.listdir(mri_side_dir) if f.endswith(".tar.gz")]
        if not tar_files:
            continue

        tar_path = os.path.join(mri_side_dir, tar_files[0])
        tensor = reconstruct_mri_from_tar(tar_path)

        if tensor is not None:
            count += 1

            # print shape of first valid tensor
            if count == 1:
                print(f"This is the shape of the tensor we extract from the MRI image: {tensor.shape}")

            yield patient_id, tensor


def load_single_patient_mri(dataset_root, patient_id, side="left"):
    """A function which returns the specific patient mri as a tensor. Used in our custom Pytorch dataset."""
    subset_dirs = [
        d for d in os.listdir(dataset_root)
        if os.path.isdir(os.path.join(dataset_root, d))
    ]

    for subset in subset_dirs:
        subset_path = os.path.join(dataset_root, subset)
        patient_path = os.path.join(subset_path, patient_id)
        mri_side_dir = os.path.join(patient_path, "mri", side)

        if not os.path.isdir(mri_side_dir):
            continue

        tar_files = [f for f in os.listdir(mri_side_dir) if f.endswith(".tar.gz")]
        if not tar_files:
            continue

        tar_path = os.path.join(mri_side_dir, tar_files[0])
        tensor = reconstruct_mri_from_tar(tar_path)

        if tensor is not None:
            return tensor
        