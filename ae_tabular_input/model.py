import torch
import torch.nn as nn
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from ae_filippo.model import Autoencoder3D


class WOMACHead(nn.Module):
    """
    Simple regression head on top of the bottleneck z. This is used to predict the WOMAC score from the latent space.

    We do global average pooling over (D, H, W) and then a Linear layer.
    Input z: (B, C, D', H', W')
    Output:  (B, 1) - WOMAC score
    """

    def __init__(self, latent_channels: int):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool3d(1)  # -> (B, C, 1, 1, 1)
        self.fc = nn.Linear(latent_channels, 1)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        # z: (B, C, D', H', W')
        pooled = self.pool(z)  # (B, C, 1, 1, 1)
        pooled = pooled.view(pooled.size(0), -1)  # (B, C)
        out = self.fc(pooled)  # (B, 1)
        return out


class JSNHead(nn.Module):
    """
    Classification head for Joint Space Narrowing (JSN) on top of the bottleneck z.
    Predicts class probabilities for 0, 1, 2, 3.

    Input z: (B, C, D', H', W')
    Output:  (B, 4) - logits for JSN classes
    """

    def __init__(self, latent_channels: int, num_classes: int = 4):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool3d(1)  # -> (B, C, 1, 1, 1)
        self.fc = nn.Linear(latent_channels, num_classes)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        # z: (B, C, D', H', W')
        pooled = self.pool(z)  # (B, C, 1, 1, 1)
        pooled = pooled.view(pooled.size(0), -1)  # (B, C)
        out = self.fc(pooled)  # (B, 4)
        return out

    def predict_probs(self, z: torch.Tensor) -> torch.Tensor:
        """
        Returns class probabilities for the given latent representation z.
        output: (B, 4)
        """
        logits = self.forward(z)
        probs = torch.softmax(logits, dim=1)
        return probs


class SurgeryHead(nn.Module):
    """
    Binary classification head for surgery prediction.
    Predicts probability of surgery (class 1) vs no surgery (class 0).

    Input z: (B, C, D', H', W')
    Output:  (B, 1) - logits for surgery (use sigmoid for prob)
    """

    def __init__(self, latent_channels: int):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Linear(latent_channels, 1)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        pooled = self.pool(z)
        pooled = pooled.view(pooled.size(0), -1)
        out = self.fc(pooled)
        return out

    def predict_probs(self, z: torch.Tensor) -> torch.Tensor:
        """
        Returns probability of surgery (class 1).
        output: (B, 1)
        """
        logits = self.forward(z)
        probs = torch.sigmoid(logits)
        return probs


class TabularEncoder(nn.Module):
    """
    Tabular Encoder with minimal architecture. Takes a scalar tabular variable as input and outputs a latent vector.
        
    Args:
        num_tabular_input (int): Number of tabular inputs (default: 1)
        tabular_latent_dim (int): Dimension of the tabular latent space (default: 4)
        hidden_dim (int): Dimension of the single hidden layer (default: 32)
    """
    
    def __init__(self, num_tabular_input=1, tabular_latent_dim=4, hidden_dim=32):
        super(TabularEncoder, self).__init__()
        
        self.encoder = nn.Sequential(
            nn.Linear(num_tabular_input, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, tabular_latent_dim)
        )
        
        # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, tabular_variables):
        """
        Forward pass through the tabular encoder.
        
        Args:
            tabular_variables: Tensor of shape (batch_size, num_tabular_input)
        
        Returns:
            tabular_latent: Tensor of shape (batch_size, tabular_latent_dim)
        """
        if tabular_variables.dim() == 1:
            tabular_variables = tabular_variables.unsqueeze(-1)
        
        return self.encoder(tabular_variables)


class AutoencoderWithTabularInput(nn.Module):
    """
    - reconstructs the MRI volume
    - predicts tabular variable from the bottleneck
    """

    def __init__(
        self,
        base_ae: Autoencoder3D,
        latent_channels: int = 64,
        num_tabular_inputs: int = 1,
        tabular_latent_dim: int = 4,
    ):
        super().__init__()
        self.encoder = base_ae.encoder
        self.decoder = base_ae.decoder

        self.num_tabular_inputs = num_tabular_inputs
        self.tabular_latent_dim = tabular_latent_dim

        self.tabular_encoder = TabularEncoder(num_tabular_input=self.num_tabular_inputs, tabular_latent_dim=self.tabular_latent_dim)

        # Different heads for predicting WOMAC score, joint space narrowing, and the binary surgery variable.
        self.womac_predictor = WOMACHead(latent_channels)
        self.jsn_predictor = JSNHead(latent_channels)
        self.surgery_predictor = SurgeryHead(latent_channels)


        # Project concatenated features back to decoder's expected channels. This way the model can hopefully learn how to mix the encoding of the MRI and the encoding of the tabular input.
        self.channel_projection = nn.Conv3d(
            in_channels=latent_channels + self.tabular_latent_dim,
            out_channels=latent_channels,
            kernel_size=1,
            padding=0
        )

    def forward(self, x: torch.Tensor, tabular_variables: torch.Tensor):
        """
        Args:
            x: (B, 1, D, 224, 224), normalized to [-1, 1]

        Returns:
            x_hat:     reconstructed volume, same shape as x
            womac_score: (B, 1)
            jsn: (B, 1)
            surgery: (B, 1)
            z:         latent tensor (B, C, D', H', W')
        """
        encoded_mri = self.encoder(x)
        encoded_tabular = self.tabular_encoder(tabular_variables)   # (B, tabular_dim)

        # Get spatial dimensions from encoded MRI.
        B, C, D, H, W = encoded_mri.shape
        
        # Reshape and broadcast tabular variable to match spatial dimensions.
        # (B, tabular_dim) -> (B, tabular_dim, 1, 1, 1) -> (B, tabular_dim, D', H', W')
        encoded_tabular = encoded_tabular.view(B, -1, 1, 1, 1)
        encoded_tabular = encoded_tabular.expand(-1, -1, D, H, W)
        
        # Now concatenate along channel dimension.
        z = torch.cat([encoded_mri, encoded_tabular], dim=1)  # (B, C+tabular_dim, D', H', W')

        # Project back to 64 channels for decoder.
        z = self.channel_projection(z)  # (B, 64, D', H', W')
        x_hat = self.decoder(z)
        
        womac_pred = self.womac_predictor(z)
        jsn_pred = self.jsn_predictor(z)
        surgery_pred = self.surgery_predictor(z)
        
        return x_hat, womac_pred, jsn_pred, surgery_pred, z


def build_tabular_input_ae(
    pretrained_ae_path: str | None,
    device: torch.device,
    in_channels: int = 1,
    latent_channels: int = 64,
    num_tabular_inputs: int = 1,
    tabular_latent_dim: int = 4,
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
        num_tabular_inputs=num_tabular_inputs,
        tabular_latent_dim=tabular_latent_dim,
    ).to(device)

    return model
