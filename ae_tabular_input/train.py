import os
import sys

import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader
from tqdm import tqdm


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)


from ae_tabular_input.dataset import KneeMRITabularDataset
from ae_tabular_input.model import build_tabular_input_ae
from ae_filippo.save import save_reconstruction
from ae_filippo.data_t import KneeMRIDataset as PlainMRIDataset


def get_device():
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"

def train_tabular_input_autoencoder(
    model,
    train_loader,
    val_loader=None,
    num_epochs=50,
    lr=1e-4,
    lambda_tabular=0.5,
    device="cuda",
    use_amp=True,
    recon_dataset=None,
    recon_outdir: str | None = None,
    phase_name="train"
):
    recon_criterion = nn.MSELoss()
    womac_criterion = nn.MSELoss()
    jsn_criterion = nn.CrossEntropyLoss()
    surgery_criterion = nn.BCEWithLogitsLoss()

    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
    
    scaler = GradScaler(enabled=use_amp)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3
    )

    best_val_loss = float("inf")

    for epoch in range(1, num_epochs + 1):
        # ----------------- TRAIN -----------------
        model.train()
        train_recon_sum = 0.0
        train_womac_sum = 0.0
        train_jsn_sum = 0.0
        train_surgery_sum = 0.0

        pbar = tqdm(train_loader, desc=f"[{phase_name}] Epoch {epoch}/{num_epochs}")

        for x, y in pbar:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            optimizer.zero_grad()

            with autocast(enabled=use_amp):
                x_hat, womac_pred, jsn_pred, surgery_pred, _ = model(x=x, tabular_variables=y)
                
                # Unpack targets (Assumes order: WOMAC, JSN, Surgery)
                target_womac = y[:, 0:1]
                target_jsn = (y[:, 1] * 3).round().clamp(0, 3).long()   # This is so that the JSN is in the range [0, 3]
                target_surgery = y[:, 2:3]

                loss_recon = recon_criterion(x_hat, x)
                loss_womac = womac_criterion(womac_pred, target_womac)
                loss_jsn = jsn_criterion(jsn_pred, target_jsn)
                loss_surgery = surgery_criterion(surgery_pred, target_surgery)
                
                # Weighted Sum
                loss = loss_recon + lambda_tabular * (loss_womac + loss_jsn + loss_surgery)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            train_recon_sum += loss_recon.item()
            train_womac_sum += loss_womac.item()
            train_jsn_sum += loss_jsn.item()
            train_surgery_sum += loss_surgery.item()

        train_recon = train_recon_sum / len(train_loader)
        train_womac = train_womac_sum / len(train_loader)
        train_jsn = train_jsn_sum / len(train_loader)
        train_surgery = train_surgery_sum / len(train_loader)

        print(
            f"[{phase_name} Epoch {epoch}] Train: recon={train_recon:.6f}, "
            f"womac={train_womac:.6f}, jsn={train_jsn:.6f}, surgery={train_surgery:.6f}"
        )

        # ----------------- VALIDATION -----------------
        if val_loader is not None and len(val_loader) > 0:
            model.eval()
            val_recon_sum = 0.0
            val_womac_sum = 0.0
            val_jsn_sum = 0.0
            val_surgery_sum = 0.0

            with torch.no_grad():
                for x, y in val_loader:
                    x = x.to(device, non_blocking=True)
                    y = y.to(device, non_blocking=True)

                    with autocast(enabled=use_amp):
                        x_hat, womac_pred, jsn_pred, surgery_pred, _ = model(x=x, tabular_variables=y)

                        # Unpack targets
                        target_womac = y[:, 0:1]
                        target_jsn = (y[:, 1] * 3).round().long()
                        target_surgery = y[:, 2:3]

                        loss_recon = recon_criterion(x_hat, x)
                        loss_womac = womac_criterion(womac_pred, target_womac)
                        loss_jsn = jsn_criterion(jsn_pred, target_jsn)
                        loss_surgery = surgery_criterion(surgery_pred, target_surgery)

                    val_recon_sum += loss_recon.item()
                    val_womac_sum += loss_womac.item()
                    val_jsn_sum += loss_jsn.item()
                    val_surgery_sum += loss_surgery.item()

            val_recon = val_recon_sum / len(val_loader)
            val_womac = val_womac_sum / len(val_loader)
            val_jsn = val_jsn_sum / len(val_loader)
            val_surgery = val_surgery_sum / len(val_loader)

            print(
                f"[{phase_name} Epoch {epoch}] Val:   recon={val_recon:.6f}, "
                f"womac={val_womac:.6f}, jsn={val_jsn:.6f}, surgery={val_surgery:.6f}"
            )

            scheduler.step(val_recon)

            # Save best model for this phase
            if val_recon < best_val_loss:
                best_val_loss = val_recon
                best_path = os.path.join("weights_tabular_input_ae", f"best_tabular_input_ae_{phase_name}.pth")
                os.makedirs("weights_tabular_input_ae", exist_ok=True)
                torch.save(model.state_dict(), best_path)
                print(f"New best model saved to {best_path}")
        else:
            scheduler.step(train_recon)

    return model


def main():
    data_root = "/vol/miltank/projects/practical_wise2526/knee-osteoarthritis-severity/data/cleaned_images_baseline"
    clinical_csv_path = "./csv/clinical00_cleaned.csv"
    tabular_variables = ["V00WOMTSL", "V00XRJSM_L", "V99ELKVSAF"]  # WOMAC score, JSN, and surgery variables for the LEFT side for now.
    side = "left"
    

    # Use ALL patients
    max_patients = None 
    
    batch_size = 8
    num_workers = 8  
    lambda_tabular = 0.5    # Weight for tabular input loss.   
    pretrained_ae_path = None   # Let's train everything from scratch.
    
    # Output Paths
    out_dir = "weights_tabular_input_ae"
    recon_outdir = "recon_samples_tabular_input"
    os.makedirs(out_dir, exist_ok=True)

    device = get_device()
    print(f"Using device: {device}")
    is_cuda = device == "cuda"
    
    if is_cuda:
        try:
            print(f"GPU Name: {torch.cuda.get_device_name(0)}")
        except:
            pass

    # ----------------- LOAD DATA -----------------
    print(f"Loading datasets (Max patients: {max_patients if max_patients else 'ALL'})...")
    
    train_set = KneeMRITabularDataset(
        root=data_root, 
        clinical_csv_path=clinical_csv_path, 
        tabular_variables=tabular_variables, 
        side=side, 
        split="train", 
        max_patients=max_patients
    )
    
    val_set = KneeMRITabularDataset(
        root=data_root, 
        clinical_csv_path=clinical_csv_path, 
        tabular_variables=tabular_variables, 
        side=side, 
        split="val", 
        max_patients=max_patients
    )
    
    if len(train_set) == 0:
        print("ERROR: No training samples found! Check dataset path.")
        return

    # Optimized DataLoaders
    train_loader = DataLoader(
        train_set, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=num_workers if is_cuda else 0, 
        pin_memory=is_cuda,
    )
    
    val_loader = DataLoader(
        val_set, 
        batch_size=1, 
        shuffle=False, 
        num_workers=2 if is_cuda else 0, 
        pin_memory=is_cuda
    )
    
    # Dataset just for saving reconstruction images
    recon_set = PlainMRIDataset(root=data_root, side=side, split="val", max_patients=max_patients)

    # ----------------- BUILD MODEL -----------------
    print("Building model...")
    model = build_tabular_input_ae(
        pretrained_ae_path=pretrained_ae_path,
        device=torch.device(device),
        in_channels=1,
        latent_channels=64,
        num_tabular_inputs=len(tabular_variables),
    )

    num_epochs = 50  # NOTE: Can play with this.

        
    model = train_tabular_input_autoencoder(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=num_epochs,       
        lr=1e-4,
        lambda_tabular=lambda_tabular,        
        device=device,
        use_amp=is_cuda,
        recon_dataset=recon_set,
        recon_outdir=recon_outdir,
        phase_name="training_phase"
    )


    # ----------------- FINISH -----------------
    final_path = os.path.join(out_dir, f"final_tabular_input_ae_{num_epochs}_epochs.pth")
    torch.save(model.state_dict(), final_path)
    print(f"\nDONE. Final model saved to {final_path}")

if __name__ == "__main__":
    main()