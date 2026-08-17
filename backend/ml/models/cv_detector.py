import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class SpatialStream(nn.Module):
    """
    Spatial Feature Extractor CNN
    Captures edge discontinuities, color blending anomalies, and spatial artifacts.
    """
    def __init__(self):
        super(SpatialStream, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(2, 2)  # 224 -> 112

        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(2, 2)  # 112 -> 56

        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(2, 2)  # 56 -> 28

        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.bn4 = nn.BatchNorm2d(256)
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool1(x)
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool2(x)
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.pool3(x)
        x = F.relu(self.bn4(self.conv4(x)))
        x = self.global_pool(x)
        return x.view(x.size(0), -1)  # 256 vector


class FrequencyStream(nn.Module):
    """
    Frequency Domain Spectrum Extractor
    Analyzes 2D Fast Fourier Transform (FFT) magnitude spectrum for periodic high-frequency grid artifacts.
    """
    def __init__(self):
        super(FrequencyStream, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(2, 2)

        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, fft_magnitude):
        x = F.relu(self.bn1(self.conv1(fft_magnitude)))
        x = self.pool1(x)
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.global_pool(x)
        return x.view(x.size(0), -1)  # 64 vector


class SpatialFrequencyDetectorNet(nn.Module):
    """
    Dual-Stream Vision Neural Network for AI Image & Video Detection
    Combines Spatial Convolutional Features with FFT Magnitude Frequency Features.
    """
    def __init__(self):
        super(SpatialFrequencyDetectorNet, self).__init__()
        self.spatial_branch = SpatialStream()
        self.frequency_branch = FrequencyStream()

        # Fusion Classifier Head (256 + 64 = 320 input features)
        self.fc1 = nn.Linear(320, 128)
        self.dropout = nn.Dropout(0.4)
        self.fc2 = nn.Linear(128, 2)  # [Real, AI] logits

    def forward(self, x, fft_mag):
        spatial_feat = self.spatial_branch(x)
        freq_feat = self.frequency_branch(fft_mag)

        fused = torch.cat((spatial_feat, freq_feat), dim=1)
        out = F.relu(self.fc1(fused))
        out = self.dropout(out)
        logits = self.fc2(out)
        return logits


def get_detector_model(model_path=None):
    """
    Factory function initializing the PyTorch model.
    Loads trained weights if present at model_path.
    """
    model = SpatialFrequencyDetectorNet()
    model.eval()

    if model_path and os.path.exists(model_path):
        try:
            state_dict = torch.load(model_path, map_location=torch.device('cpu'))
            model.load_state_dict(state_dict)
            print(f"Loaded trained vision detector model from {model_path}")
        except Exception as e:
            print(f"Could not load weights from {model_path}: {e}. Running with initialized model weights.")

    return model
