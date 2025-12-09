"""
visualize_mri.py

Loads a knee MRI (DICOM .tar.gz), normalizes and pads each slice to 224×224,
builds a 3D tensor for 3D CNN inference, and visualizes original vs padded
volumes side-by-side using vedo.
"""

import os
import tarfile
import tempfile
import pydicom
import numpy as np
from PIL import Image
from torchvision import transforms
from vedo import Volume, Plotter
import torch


# -----------------------------------------------------------------------------
# Utility functions
# -----------------------------------------------------------------------------
def extract_dicom_from_tar(tar_path):
    """Extracts DICOM files from a .tar.gz into a temporary directory."""
    temp_dir = tempfile.mkdtemp()
    with tarfile.open(tar_path, "r:gz") as tar:
        tar.extractall(path=temp_dir)
    return temp_dir


def load_dicom_volume(dicom_dir):
    """Reads all DICOMs in a directory and returns a 3D numpy array (D, H, W)."""
    dicom_files = []
    for root, _, files in os.walk(dicom_dir):
        for f in files:
            dicom_files.append(os.path.join(root, f))

    dicoms = [pydicom.dcmread(f) for f in dicom_files]
    dicoms.sort(key=lambda d: getattr(d, "InstanceNumber", getattr(d, "SliceLocation", 0)))
    volume = np.stack([d.pixel_array for d in dicoms], axis=0).astype(np.float32)

    # Normalize to [0, 1]
    volume -= volume.min()
    if volume.max() > 0:
        volume /= volume.max()

    return volume


def _resize_with_padding(img, target_size, fill_value):
    """Resize keeping aspect ratio, then pad to square."""
    w, h = img.size
    scale = target_size / max(w, h)
    new_w, new_h = int(w * scale), int(h * scale)
    img = img.resize((new_w, new_h), Image.BICUBIC)

    pad_left = (target_size - new_w) // 2
    pad_top = (target_size - new_h) // 2
    pad_right = target_size - new_w - pad_left
    pad_bottom = target_size - new_h - pad_top

    return transforms.functional.pad(img, (pad_left, pad_top, pad_right, pad_bottom), fill=fill_value)


def resize_with_padding(target_size=224, fill_value=0):
    """Returns a transform pipeline for padding resize."""
    return transforms.Compose([
        transforms.Lambda(lambda img: _resize_with_padding(img, target_size, fill_value)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])


def preprocess_volume(volume_np, target_size=224):
    """Applies resize_with_padding to all slices."""
    transform_2d = resize_with_padding(target_size=target_size, fill_value=0)
    processed_slices = []
    for i in range(volume_np.shape[0]):
        img = Image.fromarray((volume_np[i] * 255).astype(np.uint8))
        tensor_slice = transform_2d(img)
        processed_slices.append(tensor_slice.numpy())
    volume = np.stack(processed_slices, axis=0)
    return volume


def make_tensor(volume_np):
    """Converts to torch tensor shaped (1, 1, D, H, W)."""
    volume_np = np.transpose(volume_np, (1, 0, 2, 3))  # (1, D, H, W)
    volume_tensor = torch.from_numpy(volume_np).unsqueeze(0).float()
    return volume_tensor


# -----------------------------------------------------------------------------
# Visualization
# -----------------------------------------------------------------------------
def visualize_volumes(volume_original, volume_padded):
    """Show both original and padded volumes side-by-side with vedo."""
    vol_original = Volume(volume_original)
    vol_padded = Volume(volume_padded.squeeze())

    plt = Plotter(shape=(1, 2), sharecam=False)
    plt.at(0).show(vol_original, "Original 384×384 Volume")
    plt.at(1).show(vol_padded, "Padded 224×224 Volume")
    plt.interactive().close()


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main():
    tar_path = "/Users/filippofocaccia/Desktop/adlm-knee-osteoarthritis/baseline/0.C.2/9000798/mri/left/10249506.tar.gz"

    temp_dir = extract_dicom_from_tar(tar_path)
    volume = load_dicom_volume(temp_dir)
    print("Original volume shape:", volume.shape)

    volume_padded = preprocess_volume(volume, target_size=224)
    print("After padding:", volume_padded.shape)

    volume_tensor = make_tensor(volume_padded)
    print("Tensor shape:", volume_tensor.shape)

    visualize_volumes(volume, volume_padded)


if __name__ == "__main__":
    main()
