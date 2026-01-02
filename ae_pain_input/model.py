import torch
import torch.nn as nn
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from ae_filippo.model import Autoencoder3D


class PainHead(nn.Module):
    """
    Simple regression head on top of the bottleneck z.

    We do global average pooling over (D, H, W) and then a Linear layer.
    Input z: (B, C, D', H', W')
    Output:  (B, num_outputs) - KOOS Pain score
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

class PainEncoder(nn.Module):
    """
    Pain Encoder with minimal architecture.
        
    Args:
        pain_latent_dim (int): Dimension of the pain latent space (default: 4)
        hidden_dim (int): Dimension of the single hidden layer (default: 32)
    """
    
    def __init__(self, pain_latent_dim=4, hidden_dim=32):
        super(PainEncoder, self).__init__()
        
        self.encoder = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, pain_latent_dim)
        )
        
        # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, pain_level):
        """
        Forward pass through the pain encoder.
        
        Args:
            pain_level: Tensor of shape (batch_size,) or (batch_size, 1)
        
        Returns:
            pain_latent: Tensor of shape (batch_size, pain_latent_dim)
        """
        if pain_level.dim() == 1:
            pain_level = pain_level.unsqueeze(-1)
        
        return self.encoder(pain_level)


class AutoencoderWithPainInput(nn.Module):
    """
    - reconstructs the MRI volume
    - predicts pain from the bottleneck
    """

    def __init__(
        self,
        base_ae: Autoencoder3D,
        latent_channels: int = 64,
        num_pain_outputs: int = 1,
    ):
        super().__init__()
        self.encoder = base_ae.encoder
        self.decoder = base_ae.decoder
        self.pain_encoder = PainEncoder()
        self.pain_predictor = PainHead(latent_channels, num_pain_outputs)

        # Project concatenated features back to decoder's expected channels
        self.channel_projection = nn.Conv3d(
            in_channels=latent_channels + 4,  # 68 by default
            out_channels=latent_channels,  # 64 by default
            kernel_size=1,
            padding=0
        )

    def forward(self, x: torch.Tensor, pain_level: torch.Tensor):
        """
        Args:
            x: (B, 1, D, 224, 224), normalized to [-1, 1]

        Returns:
            x_hat:     reconstructed volume, same shape as x
            pain_pred: (B, num_pain_outputs)
            z:         latent tensor (B, C, D', H', W')
        """
        encoded_mri = self.encoder(x)
        encoded_pain = self.pain_encoder(pain_level)

        # Get spatial dimensions from encoded MRI.
        B, C, D, H, W = encoded_mri.shape
        
        # Reshape and broadcast pain to match spatial dimensions.
        # (B, pain_dim) -> (B, pain_dim, 1, 1, 1) -> (B, pain_dim, D', H', W')
        encoded_pain = encoded_pain.view(B, -1, 1, 1, 1)
        encoded_pain = encoded_pain.expand(-1, -1, D, H, W)
        
        # Now concatenate along channel dimension.
        z = torch.cat([encoded_mri, encoded_pain], dim=1)  # (B, C+pain_dim, D', H', W')

        # Project back to 64 channels for decoder
        z = self.channel_projection(z)  # (B, 64, D', H', W')
        x_hat = self.decoder(z)
        pain_pred = self.pain_predictor(z)
        return x_hat, pain_pred, z


def build_pain_input_ae(
    pretrained_ae_path: str | None,
    device: torch.device,
    in_channels: int = 1,
    latent_channels: int = 64,
    num_pain_outputs: int = 1,
) -> AutoencoderWithPainInput:

    base_ae = Autoencoder3D(in_channels=in_channels, latent_channels=latent_channels)

    if pretrained_ae_path is not None:
        state = torch.load(pretrained_ae_path, map_location=device)
        base_ae.load_state_dict(state)
        print(f"[build_pain_ae] Loaded pretrained AE from {pretrained_ae_path}")

    base_ae.to(device)

    model = AutoencoderWithPainInput(
        base_ae=base_ae,
        latent_channels=latent_channels,
        num_pain_outputs=num_pain_outputs,
    ).to(device)

    return model
