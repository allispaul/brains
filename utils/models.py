"""Model architectures."""
import torch
from torch import nn
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
from torchvision import transforms
import torchvision.transforms.v2.functional as F

class Spectrogram_EfficientNet(nn.Module):
    """An EfficientNetB0 vision model for the spectrogram data.
    
    Parameters:
      frozen (default True): Whether to freeze the model's pretrained layers.
    """
    def __init__(self, frozen=True):
        super().__init__()
        self.preprocessor = SpectrogramPreprocessor()
        self.efficientnet = efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)
        if frozen:
            # freeze pretrained layers besides classifier
            for param in self.efficientnet.features.parameters():
                param.requires_grad = False
        # replace classifier with one of appropriate shape
        self.efficientnet.classifier = nn.Linear(1280, 6)
        # make model output log-probabilities
        self.activation = nn.LogSoftmax(dim=1)
        
    def forward(self, x):
        return self.activation(self.efficientnet(self.preprocessor(x)))
    
class SpectrogramPreprocessor(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, x):
        x = x.unsqueeze(1).repeat(1,3,1,1)  # convert to RGB
        # EffnetB0 expects size [224, 224] images. But I won't center-crop, since
        # this leaves out too much data from LL and RP.
        x = F.resize(x, (224, 224), interpolation=transforms.InterpolationMode.BICUBIC)
        x.clamp_(1e-4, 1e7)  # avoid log(0) errors
        x.log_()  # log-scale
        # scale to [0, 1]
        x = (x - torch.amin(x, dim=(-1,-2), keepdim=True)) / torch.amax(x, dim=(-1,-2), keepdim=True)
        # normalization from EffnetB0 transforms
        x = F.normalize(x, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        return x
        
    