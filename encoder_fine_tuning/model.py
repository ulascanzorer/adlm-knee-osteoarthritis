import torch
import torch.nn as nn
from ../ae_filippo.model import Encoder3D

class RegressionHead(nn.Module):
    def __init__(self):
        super(RegressionHead, self).__init__()
        self.fc = nn.Linear(2048, 1)
    
    def forward(self, x):
        return self.fc(x)


class EncoderAndRegression(nn.Module):
    def __init__(self, encoder: Encoder3D, regression: RegressionHead):
        super().__init__()
        self.encoder = encoder
        self.regression = regression

    def forward(self, x):
        z = self.encoder(x)
        pred = self.regression(z)
        return pred