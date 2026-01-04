import torch
import torch.nn as nn
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from ae_filippo.model import Autoencoder3D


class TabularHead(nn.Module):
    """
    Simple regression head on top of the bottleneck z. This is used to predict a tabular variable from the latent space.

    We do global average pooling over (D, H, W) and then a Linear layer.
    Input z: (B, C, D', H', W')
    Output:  (B, num_outputs) - tabular variable
    """

    def __init__(self, latent_channels: int, num_outputs: int = 1):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool3d(1)  # -> (B, C, 1, 1, 1)
        self.fc = nn.Linear(latent_channels, num_outputs)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        # z: (B, C, D', H', W')
        pooled = self.pool(z)  # (B, C, 1, 1, 1)
        pooled = pooled.view(pooled.size(0), -1)  # (B, C)
        out = self.fc(pooled)  # (B, num_outputs)
        return out

class TabularEncoder(nn.Module):
    """
    Tabular Encoder with minimal architecture. Takes a scalar tabular variable as input and outputs a latent vector.
        
    Args:
        tabular_latent_dim (int): Dimension of the tabular latent space (default: 4)
        hidden_dim (int): Dimension of the single hidden layer (default: 32)
    """
    
    def __init__(self, tabular_latent_dim=4, hidden_dim=32):
        super(TabularEncoder, self).__init__()
        
        self.encoder = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, tabular_latent_dim)
        )
        
        # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, tabular_var):
        """
        Forward pass through the tabular encoder.
        
        Args:
            tabular_var: Tensor of shape (batch_size,) or (batch_size, 1)
        
        Returns:
            tabular_latent: Tensor of shape (batch_size, tabular_latent_dim)
        """
        if tabular_var.dim() == 1:
            tabular_var = tabular_var.unsqueeze(-1)
        
        return self.encoder(tabular_var)


class AutoencoderWithTabularInput(nn.Module):
    """
    - reconstructs the MRI volume
    - predicts tabular variable from the bottleneck
    """

    def __init__(
        self,
        base_ae: Autoencoder3D,
        latent_channels: int = 64,
        num_tabular_outputs: int = 1,
    ):
        super().__init__()
        self.encoder = base_ae.encoder
        self.decoder = base_ae.decoder
        self.tabular_encoder = TabularEncoder()
        self.tabular_predictor = TabularHead(latent_channels, num_tabular_outputs)

        # Project concatenated features back to decoder's expected channels
        self.channel_projection = nn.Conv3d(
            in_channels=latent_channels + 4,  # 68 by default
            out_channels=latent_channels,  # 64 by default
            kernel_size=1,
            padding=0
        )

    def forward(self, x: torch.Tensor, tabular_var: torch.Tensor):
        """
        Args:
            x: (B, 1, D, 224, 224), normalized to [-1, 1]

        Returns:
            x_hat:     reconstructed volume, same shape as x
            tabular_pred: (B, num_tabular_outputs)
            z:         latent tensor (B, C, D', H', W')
        """
        encoded_mri = self.encoder(x)
        encoded_tabular = self.tabular_encoder(tabular_var)

        # Get spatial dimensions from encoded MRI.
        B, C, D, H, W = encoded_mri.shape
        
        # Reshape and broadcast tabular variable to match spatial dimensions.
        # (B, tabular_dim) -> (B, tabular_dim, 1, 1, 1) -> (B, tabular_dim, D', H', W')
        encoded_tabular = encoded_tabular.view(B, -1, 1, 1, 1)
        encoded_tabular = encoded_tabular.expand(-1, -1, D, H, W)
        
        # Now concatenate along channel dimension.
        z = torch.cat([encoded_mri, encoded_tabular], dim=1)  # (B, C+tabular_dim, D', H', W')

        # Project back to 64 channels for decoder
        z = self.channel_projection(z)  # (B, 64, D', H', W')
        x_hat = self.decoder(z)
        tabular_pred = self.tabular_predictor(z)
        return x_hat, tabular_pred, z


def build_tabular_input_ae(
    pretrained_ae_path: str | None,
    device: torch.device,
    in_channels: int = 1,
    latent_channels: int = 64,
    num_tabular_outputs: int = 1,
) -> AutoencoderWithTabularInput:

    base_ae = Autoencoder3D(in_channels=in_channels, latent_channels=latent_channels)

    if pretrained_ae_path is not None:
        state = torch.load(pretrained_ae_path, map_location=device)
        base_ae.load_state_dict(state)
        print(f"[build_tabular_input_ae] Loaded pretrained AE from {pretrained_ae_path}")

    base_ae.to(device)

    model = AutoencoderWithTabularInput(
        base_ae=base_ae,
        latent_channels=latent_channels,
        num_tabular_outputs=num_tabular_outputs,
    ).to(device)

    return model
