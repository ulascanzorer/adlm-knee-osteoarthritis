# train_f.py
import torch
from tqdm import tqdm
from torch.amp import autocast, GradScaler
import wandb



def train_autoencoder(
        model,
        dataloader,
        num_epochs=50,
        lr=1e-4,
        device="cuda",
        use_amp=True,
    ):

    device = torch.device(device)
    model = model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = torch.nn.MSELoss()

    # Enable AMP only on CUDA
    use_amp = use_amp and (device.type == "cuda") and torch.cuda.is_available()
    scaler = GradScaler(enabled=use_amp)

    if len(dataloader) == 0:
        print("Dataloader is empty. No training will be performed.")
        return model

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}", unit="batch")

        for vol in pbar:
            # vol: (B, 1, D, H, W)
            vol = vol.to(device, non_blocking=True).float()

            opt.zero_grad()

            if use_amp:
                with autocast("cuda"):
                    recon, z = model(vol)
                    loss = loss_fn(recon, vol)
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()
            else:
                recon, z = model(vol)
                loss = loss_fn(recon, vol)
                loss.backward()
                opt.step()

            running_loss += loss.item()
            pbar.set_postfix({"batch_loss": loss.item()})
            


        epoch_loss = running_loss / len(dataloader)

        print(f"Epoch {epoch+1}/{num_epochs} - Loss: {epoch_loss:.6f}")

    return model
