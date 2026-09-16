import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torchvision.models import EfficientNet_B0_Weights

class FFTFrequencyBranch(nn.Module):
    """
    2D Fast Fourier Transform Magnitude Spectrum Branch
    Extracts periodic high-frequency noise & grid artifacts characteristic of generative models.
    """
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(2, 2)

        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, fft_mag):
        x = F.relu(self.bn1(self.conv1(fft_mag)))
        x = self.pool1(x)
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.global_pool(x)
        return x.view(x.size(0), -1)  # 64 dims


class EfficientNetDetector(nn.Module):
    """
    Hybrid EfficientNet-B0 + 2D FFT Frequency Spectral Neural Network
    Combines ImageNet pretrained deep spatial features with spectral forensic features.
    """
    def __init__(self, pretrained=True):
        super().__init__()

        # Load EfficientNet-B0 Backbone
        if pretrained:
            try:
                weights = EfficientNet_B0_Weights.DEFAULT
                self.backbone = models.efficientnet_b0(weights=weights)
            except Exception as e:
                print(f"Loading pretrained weights failed ({e}). Initializing standard EfficientNet-B0 backbone.")
                self.backbone = models.efficientnet_b0(weights=None)
        else:
            self.backbone = models.efficientnet_b0(weights=None)

        # Separate feature extractor from classifier
        self.spatial_features = self.backbone.features
        self.spatial_pool = self.backbone.avgpool  # Outputs 1280 features

        # Frequency Branch
        self.frequency_branch = FFTFrequencyBranch()

        # Fusion Classifier Head (1280 + 64 = 1344 features)
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.4, inplace=False),
            nn.Linear(1344, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(p=0.3, inplace=False),
            nn.Linear(128, 2)  # [Real, AI] logits
        )

    def forward(self, x, fft_mag=None):
        # Extract Spatial Features
        s_feat = self.spatial_features(x)
        s_feat = self.spatial_pool(s_feat)
        s_feat = torch.flatten(s_feat, 1)  # (B, 1280)

        # Extract Frequency Features
        if fft_mag is not None:
            f_feat = self.frequency_branch(fft_mag)
        else:
            # Fallback zero tensor if fft_mag not provided
            f_feat = torch.zeros((x.size(0), 64), device=x.device)

        # Fuse Streams
        fused = torch.cat((s_feat, f_feat), dim=1)  # (B, 1344)
        logits = self.classifier(fused)
        return logits


def get_efficientnet_model(model_path=None, pretrained=True):
    """
    Factory function for initializing EfficientNetDetector model.
    Loads trained weights if present at model_path.
    """
    model = EfficientNetDetector(pretrained=pretrained)
    model.eval()

    if model_path and os.path.exists(model_path):
        try:
            state_dict = torch.load(model_path, map_location=torch.device('cpu'))
            model.load_state_dict(state_dict)
            print(f"Loaded trained EfficientNet weights from: {model_path}")
        except Exception as e:
            print(f"Warning: Could not load weights from {model_path}: {e}")

    return model
