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
from ae_filippo.data_t import KneeMRIDataset as PlainMRIDataset


def get_device():
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


# -------------------------
# Masked loss helpers
# -------------------------

def masked_mse(pred, target, mask):
    # pred, target: (B, 1)
    # mask: (B, 1)
    diff = (pred - target) ** 2
    diff = diff * mask
    return diff.sum() / (mask.sum() + 1e-8)


def masked_bce(logits, target, mask):
    # logits, target, mask: (B, 1)
    loss = nn.functional.binary_cross_entropy_with_logits(
        logits, target, reduction="none"
    )
    loss = loss * mask
    return loss.sum() / (mask.sum() + 1e-8)


def masked_ce(logits, target, mask):
    # logits: (B, C), target: (B,), mask: (B,)
    loss = nn.functional.cross_entropy(logits, target, reduction="none")
    loss = loss * mask
    return loss.sum() / (mask.sum() + 1e-8)


def apply_tabular_dropout(
    tab_x: torch.Tensor,
    tab_mask: torch.Tensor,
    p: float,
    full_modality_p: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Simulate missing tabular inputs during training.

    - With probability `full_modality_p`, drop ALL tabular inputs for the sample.
    - Otherwise, independently drop each present feature with probability `p`.

    Assumes missing features already have tab_mask=0 and tab_x=0.
    """
    if p <= 0.0 and full_modality_p <= 0.0:
        return tab_x, tab_mask

    # Full modality dropout per sample (B, 1)
    if full_modality_p > 0.0:
        drop_all = (torch.rand(tab_mask.size(0), 1, device=tab_mask.device) < full_modality_p).float()
        tab_mask = tab_mask * (1.0 - drop_all)
        tab_x = tab_x * tab_mask

    # Feature dropout (only affects currently-available features)
    if p > 0.0:
        drop_feat = (torch.rand_like(tab_mask) < p).float()
        drop_feat = drop_feat * tab_mask  # only drop features that are present
        tab_mask = tab_mask * (1.0 - drop_feat)
        tab_x = tab_x * tab_mask

    return tab_x, tab_mask


def train_tabular_input_autoencoder(
    model,
    train_loader,
    val_loader=None,
    num_epochs=50,
    lr=1e-4,
    lambda_tabular=0.5,
    device="cuda",
    use_amp=True,
    phase_name="train",
    tabular_dropout_p: float = 0.2,
    tabular_dropout_full_p: float = 0.0,
):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scaler = GradScaler(enabled=use_amp)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3
    )

    best_val_loss = float("inf")

    for epoch in range(1, num_epochs + 1):
        model.train()
        recon_sum = womac_sum = jsn_sum = surg_sum = 0.0

        pbar = tqdm(train_loader, desc=f"[{phase_name}] Epoch {epoch}/{num_epochs}")

        for x, tab_x, tab_mask, y, y_mask in pbar:
            x = x.to(device)
            tab_x = tab_x.to(device)
            tab_mask = tab_mask.to(device)
            y = y.to(device)
            y_mask = y_mask.to(device)

            # simulate missing tabular inputs (TRAIN ONLY)
            tab_x_do, tab_mask_do = apply_tabular_dropout(
                tab_x, tab_mask, p=tabular_dropout_p, full_modality_p=tabular_dropout_full_p
            )

            optimizer.zero_grad()

            with autocast(enabled=use_amp):
                x_hat, womac_pred, jsn_pred, surg_pred, _ = model(
                    x, tab_x_do, tab_mask_do
                )

                target_womac = y[:, 0:1]
                target_jsn = (y[:, 1] * 3).round().clamp(0, 3).long()
                target_surg = y[:, 2:3]

                mask_womac = y_mask[:, 0:1]
                mask_jsn = y_mask[:, 1]
                mask_surg = y_mask[:, 2:3]

                loss_recon = nn.functional.mse_loss(x_hat, x)
                loss_womac = masked_mse(womac_pred, target_womac, mask_womac)
                loss_jsn = masked_ce(jsn_pred, target_jsn, mask_jsn)
                loss_surg = masked_bce(surg_pred, target_surg, mask_surg)

                loss = loss_recon + lambda_tabular * (loss_womac + loss_jsn + loss_surg)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            recon_sum += loss_recon.item()
            womac_sum += loss_womac.item()
            jsn_sum += loss_jsn.item()
            surg_sum += loss_surg.item()

        n = len(train_loader)
        print(
            f"[{phase_name} Epoch {epoch}] "
            f"Train: recon={recon_sum/n:.6f}, "
            f"womac={womac_sum/n:.6f}, "
            f"jsn={jsn_sum/n:.6f}, "
            f"surgery={surg_sum/n:.6f}"
        )

        # ----------------- VALIDATION -----------------
        if val_loader is not None:
            model.eval()
            recon_sum = womac_sum = jsn_sum = surg_sum = 0.0

            with torch.no_grad():
                for x, tab_x, tab_mask, y, y_mask in val_loader:
                    x = x.to(device)
                    tab_x = tab_x.to(device)
                    tab_mask = tab_mask.to(device)
                    y = y.to(device)
                    y_mask = y_mask.to(device)

                    with autocast(enabled=use_amp):
                        x_hat, womac_pred, jsn_pred, surg_pred, _ = model(
                            x, tab_x, tab_mask
                        )

                        target_womac = y[:, 0:1]
                        target_jsn = (y[:, 1] * 3).round().clamp(0, 3).long()
                        target_surg = y[:, 2:3]

                        mask_womac = y_mask[:, 0:1]
                        mask_jsn = y_mask[:, 1]
                        mask_surg = y_mask[:, 2:3]

                        loss_recon = nn.functional.mse_loss(x_hat, x)
                        loss_womac = masked_mse(womac_pred, target_womac, mask_womac)
                        loss_jsn = masked_ce(jsn_pred, target_jsn, mask_jsn)
                        loss_surg = masked_bce(surg_pred, target_surg, mask_surg)

                    recon_sum += loss_recon.item()
                    womac_sum += loss_womac.item()
                    jsn_sum += loss_jsn.item()
                    surg_sum += loss_surg.item()

            n = len(val_loader)
            val_recon = recon_sum / n

            print(
                f"[{phase_name} Epoch {epoch}] "
                f"Val:   recon={val_recon:.6f}, "
                f"womac={womac_sum/n:.6f}, "
                f"jsn={jsn_sum/n:.6f}, "
                f"surgery={surg_sum/n:.6f}"
            )

            scheduler.step(val_recon)

            if val_recon < best_val_loss:
                best_val_loss = val_recon
                os.makedirs("weights_tabular_input_ae", exist_ok=True)
                path = f"weights_tabular_input_ae/best_tabular_input_ae_{phase_name}.pth"
                torch.save(model.state_dict(), path)
                print(f"New best model saved to {path}")

    return model


def main():
    data_root = "/vol/miltank/projects/practical_wise2526/knee-osteoarthritis-severity/data/cleaned_images_baseline"
    clinical_csv_path = "./csv/clinical00_cleaned.csv"
    side = "left"

    batch_size = 8
    num_workers = 8
    num_epochs = 50
    lambda_tabular = 0.5

    # simulate missing tabular inputs
    tabular_dropout_p = 0.2          # per-feature drop prob
    tabular_dropout_full_p = 0.0   

    device = get_device()
    is_cuda = device == "cuda"

    train_set = KneeMRITabularDataset(
        root=data_root,
        clinical_csv_path=clinical_csv_path,
        side=side,
        split="train",
    )

    val_set = KneeMRITabularDataset(
        root=data_root,
        clinical_csv_path=clinical_csv_path,
        side=side,
        split="val",
    )

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
        pin_memory=is_cuda,
    )

    model = build_tabular_input_ae(
        pretrained_ae_path=None,
        device=torch.device(device),
        in_channels=1,
        latent_channels=64,
        num_tabular_inputs=len(train_set.tab_inputs),
    )

    train_tabular_input_autoencoder(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=num_epochs,
        lambda_tabular=lambda_tabular,
        device=device,
        use_amp=is_cuda,
        phase_name="training_phase",
        tabular_dropout_p=tabular_dropout_p,
        tabular_dropout_full_p=tabular_dropout_full_p,
    )

    os.makedirs("weights_tabular_input_ae", exist_ok=True)
    torch.save(
        model.state_dict(),
        f"weights_tabular_input_ae/final_tabular_input_ae_{num_epochs}_epochs.pth",
    )


if __name__ == "__main__":
    main()
