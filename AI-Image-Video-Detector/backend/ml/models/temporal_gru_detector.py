import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torchvision.models import EfficientNet_B0_Weights

class SpatialFeatureExtractor(nn.Module):
    """
    Extracts 1280-dimensional spatial feature representations from individual video frames
    using ImageNet-pretrained EfficientNet-B0 backbone.
    """
    def __init__(self, pretrained=True):
        super().__init__()
        if pretrained:
            try:
                weights = EfficientNet_B0_Weights.DEFAULT
                self.backbone = models.efficientnet_b0(weights=weights)
            except Exception:
                self.backbone = models.efficientnet_b0(weights=None)
        else:
            self.backbone = models.efficientnet_b0(weights=None)

        self.features = self.backbone.features
        self.avgpool = self.backbone.avgpool  # Outputs 1280 dimensions

    def forward(self, x):
        # x: (B, 3, 224, 224)
        feat = self.features(x)
        feat = self.avgpool(feat)
        return torch.flatten(feat, 1)  # (B, 1280)


class GRUTemporalModel(nn.Module):
    """
    Dual-Branch Temporal Sequence Neural Network for Video AI Detection.
    Processes sequences of spatial feature vectors through a 2-Layer Gated Recurrent Unit (GRU)
    to capture inter-frame temporal inconsistencies, unnatural warping, and flickering.
    """
    def __init__(
        self,
        feature_dim=1280,
        hidden_dim=256,
        num_layers=2,
        dropout=0.3,
        num_classes=2,
        pretrained_spatial=True
    ):
        super().__init__()
        self.spatial_extractor = SpatialFeatureExtractor(pretrained=pretrained_spatial)
        
        self.gru = nn.GRU(
            input_size=feature_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )

        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(128, num_classes)
        )

    def forward(self, frame_sequence, precomputed_embeddings=None):
        """
        Args:
            frame_sequence: (B, T, 3, 224, 224) video frame sequence tensor.
            precomputed_embeddings: Optional (B, T, 1280) feature tensor.
        Returns:
            logits: (B, 2) classification logits [REAL, AI_GENERATED].
        """
        if precomputed_embeddings is not None:
            embeddings = precomputed_embeddings
        else:
            b, t, c, h, w = frame_sequence.shape
            # Reshape (B, T, C, H, W) -> (B*T, C, H, W) for parallel spatial feature extraction
            frames_flat = frame_sequence.view(b * t, c, h, w)
            spatial_feats = self.spatial_extractor(frames_flat)  # (B*T, 1280)
            embeddings = spatial_feats.view(b, t, -1)  # (B, T, 1280)

        # Pass through GRU sequence layer
        gru_out, _ = self.gru(embeddings)  # gru_out: (B, T, 256)
        
        # Take the final sequence step state
        last_temporal_state = gru_out[:, -1, :]  # (B, 256)
        
        logits = self.classifier(last_temporal_state)  # (B, 2)
        return logits


def get_video_temporal_model(model_path=None, pretrained_spatial=True):
    model = GRUTemporalModel(pretrained_spatial=pretrained_spatial)
    model.eval()

    if model_path and os.path.exists(model_path):
        try:
            state_dict = torch.load(model_path, map_location=torch.device('cpu'))
            model.load_state_dict(state_dict)
            print(f"Loaded trained GRU temporal model from: {model_path}")
        except Exception as e:
            print(f"Warning: Could not load GRU model weights from {model_path}: {e}")

    return model
