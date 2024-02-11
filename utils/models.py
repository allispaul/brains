"""Model architectures."""
from torch import nn
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights

class Spectrogram_EfficientNet(nn.Module):
    """An EfficientNetB0 vision model for the spectrogram data."""
    def __init__(self):
        super().__init__()
        self.efficientnet = efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)
        # freeze pretrained layers besides classifier
        for param in self.efficientnet.features.parameters():
            param.requires_grad = False
        # replace classifier with one of appropriate shape
        self.efficientnet.classifier = nn.Linear(1280, 6)
        
    def forward(self, x):
        # Convert grayscale images, [batch, H, W], to RGB images, [batch, 3, H, W]
        return self.efficientnet(x.unsqueeze(1).repeat(1, 3, 1, 1))