# train.py
from model import Autoencoder3D
from train_f import train_autoencoder
from data_t import KneeMRIDataset
import torch
from torch.utils.data import DataLoader
from save import save_reconstruction, log_reconstruction_to_wandb
import wandb

def get_device():
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def main():
    data_root = "/vol/miltank/projects/practical_wise2526/knee-osteoarthritis-severity/data/cleaned_images_baseline"
    device = get_device()
    print(f"Using device: {device}")

    dataset = KneeMRIDataset(data_root, side='left', max_patients=None)

    if len(dataset) == 0:
        print("No volumes found in dataset. Check data_root and structure.")
        return

    # GPU-friendly DataLoader config
    is_cuda = (device == "cuda")
    loader = DataLoader(
        dataset,
        batch_size=2,
        shuffle=True,
        num_workers=4 if is_cuda else 0,
        pin_memory=is_cuda,
        persistent_workers=is_cuda,
    )

    model = Autoencoder3D(in_channels=1, latent_channels=64)

    wandb.init(
    project="knee-mri-autoencoder",
    config={
        "learning_rate": 1e-4,
        "epochs": 50,
        "batch_size": 1,
        "latent_channels": 64,
        "img_depth": 160,
        "img_res": "224x224",
    }
)



    trained = train_autoencoder(
        model,
        loader,
        num_epochs=50,
        lr=1e-4,
        device=device,
        use_amp=is_cuda,
    )

    torch.save(trained.state_dict(), "trained_knee_3d_autoencoder.pth")
    print("Model saved to trained_knee_3d_autoencoder.pth")
    save_reconstruction(trained, dataset, device)
    log_reconstruction_to_wandb(trained, dataset, device, num_samples=5)
    wandb.finish()





if __name__ == "__main__":
    main()
