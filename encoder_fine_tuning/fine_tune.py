from ../ae_filippo.model import Autoencoder3D, Encoder3D
import torch
from dataset import PatientAndTabularDataset
from model import EncoderAndRegression, RegressionHead

def load_encoder(weights_path: str, device: torch.device) -> Encoder3D:
    ae = Autoencoder3D()
    state_dict = torch.load(weights_path, map_location=device)
    ae.load_state_dict(state_dict)
    ae = ae.to(device)
    ae.eval()

    return ae.encoder


def get_device():
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def main():
    data_root = "/vol/miltank/projects/practical_wise2526/knee-osteoarthritis-severity/data/cleaned_images_baseline"
    tabular_variable = "P01KSURGL"  # TODO: Change this to be the tabular variable indicating subjective pain level or something that makes more sense.
    clinical_csv_path = "./csv/clinical00_cleaned.csv"  # TODO: Also adjust this if needed.
    weight_path = "./50_ae.pth"
    device = get_device()
    print(f"Using device: {device}")

    # Load train + validation sets
    train_set = PatientAndTabularDataset(data_root, clinical_csv_path, tabular_variable, side="left", split="train")
    val_set   = PatientAndTabularDataset(data_root, clinical_csv_path, tabular_variable, side="left", split="val")

    if len(train_set) == 0:
        print("ERROR: No training samples found! Check dataset path and splits.")
        return


    # Dataloaders

    is_cuda = device == "cuda"

    train_loader = DataLoader(
        train_set,
        batch_size=4,
        shuffle=True,
        num_workers=4 if is_cuda else 0,
        pin_memory=is_cuda,
        persistent_workers=is_cuda,
    )

    val_loader = DataLoader(
        val_set,
        batch_size=1,
        shuffle=False,
        num_workers=2 if is_cuda else 0,
        pin_memory=is_cuda,
        persistent_workers=is_cuda and len(val_set) > 0,
    )

    # Load the pre-trained encoder.
    pre_trained_encoder = load_encoder(weight_path, device)

    # Create the combined model.
    model = EncoderAndRegression(pre_trained_encoder, RegressionHead().to(device))

    # TODO: Change the code below and train the combined model in a supervised fashion using the new dataset!

    # Train with validation support
    num_epochs=50
    trained_model = train_autoencoder(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=num_epochs,
        lr=1e-4,
        device=device,
        use_amp=is_cuda,
        use_wandb=False,  # change to True if using wandb
    )

    # Save final model
    
    torch.save(trained_model.state_dict(), "weights_ae/"+str(num_epochs)+"_ae.pth")
    print("Model saved to trained_knee_3d_autoencoder.pth")

    # Save example reconstruction
    if len(val_set) > 0:
        save_reconstruction(trained_model, val_set, device)
    else:
        save_reconstruction(trained_model, train_set, device)


if __name__ == "__main__":
    main()