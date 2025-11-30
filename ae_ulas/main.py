# train.py
from model import Autoencoder3D
from train import train_autoencoder
from dataset import KneeMRIDataset
import torch
from torch.utils.data import DataLoader
import argparse


def get_device():
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"

def main():
    parser = argparse.ArgumentParser(
        description="Autoencoder training, using patient MRI images"
    )

    parser.add_argument(
        "--side",
        type=str,
        choices=["left", "right", "both"],
        default="left",
        help="Which side(s) to process",
    )

    parser.add_argument(
        "--max_patients",
        type=int,
        default=None,
        help="Optional limit on number of patients to process.",
    )

    args = parser.parse_args()

    print("Beginning running our program...")

    data_root = "/vol/miltank/projects/practical_wise2526/knee-osteoarthritis-severity/data/cleaned_images_baseline"
    device = get_device()
    print(f"Using device: {device}")

    dataset = KneeMRIDataset(data_root, side=args.side, max_patients=args.max_patients)
    len_dataset = len(dataset)

    if len_dataset == 0:
        print("No volumes found in dataset. Check data_root and structure.")
        return

    print(f"Created the dataset, this is the number of samples we have: {len_dataset}")

    # GPU-friendly DataLoader config
    is_cuda = (device == "cuda")
    loader = DataLoader(
        dataset,
        batch_size=3,
        shuffle=True,
        num_workers=4 if is_cuda else 0,
        pin_memory=is_cuda,
        persistent_workers=is_cuda,
    )

    model = Autoencoder3D(in_channels=1, latent_channels=64)

    trained = train_autoencoder(
        model,
        loader,
        num_epochs=5,
        lr=1e-4,
        device=device,
        use_amp=is_cuda,
    )

    torch.save(trained.state_dict(), "trained_knee_3d_autoencoder.pth")
    print("Model saved to trained_knee_3d_autoencoder.pth")


if __name__ == "__main__":
    main()
