import os
import sys
import json
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

try:
    from backend.services.frame_extractor import FrameExtractor
    from backend.ml.models.efficientnet_detector import get_efficientnet_model
    from backend.ml.models.temporal_gru_detector import get_video_temporal_model
    from backend.ml.datasets.dataset import get_transforms, compute_fft_tensor
except ImportError:
    from services.frame_extractor import FrameExtractor
    from ml.models.efficientnet_detector import get_efficientnet_model
    from ml.models.temporal_gru_detector import get_video_temporal_model
    from ml.datasets.dataset import get_transforms, compute_fft_tensor

class VideoDetectorService:
    def __init__(self, model_path=None):
        base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
        models_dir = os.path.join(base_dir, 'ml', 'models')

        spatial_ckpt = os.path.abspath(os.path.join(models_dir, 'best_model.pth'))
        temporal_ckpt = os.path.abspath(os.path.join(models_dir, 'video_temporal_model.pth'))
        fusion_cfg_path = os.path.abspath(os.path.join(models_dir, 'fusion_config.json'))

        if not os.path.exists(spatial_ckpt):
            raise FileNotFoundError(f"Spatial model checkpoint not found at: '{spatial_ckpt}'. Ensure best_model.pth is committed and deployed.")
        if not os.path.exists(temporal_ckpt):
            raise FileNotFoundError(f"Temporal model checkpoint not found at: '{temporal_ckpt}'. Ensure video_temporal_model.pth is committed and deployed.")

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.frame_extractor = FrameExtractor(target_sample_fps=1.0, max_frames=16)

        self.spatial_model = get_efficientnet_model(spatial_ckpt, pretrained=False).to(self.device)
        self.temporal_model = get_video_temporal_model(temporal_ckpt, pretrained_spatial=False).to(self.device)

        self.spatial_model.eval()
        self.temporal_model.eval()

        self.transforms = get_transforms(is_training=False)

        self.w_spatial = 0.50
        self.w_temporal = 0.50
        self.threshold = 0.30
        self.unc_lower = 0.25
        self.unc_upper = 0.35
        self.disagree_thresh = 0.40

        if os.path.exists(fusion_cfg_path):
            try:
                with open(fusion_cfg_path, 'r') as f:
                    cfg = json.load(f)
                    self.w_spatial = float(cfg.get('spatial_weight', 0.50))
                    self.w_temporal = float(cfg.get('temporal_weight', 0.50))
                    self.threshold = float(cfg.get('classification_threshold', 0.30))
                    self.unc_lower = float(cfg.get('uncertainty_lower', self.threshold - 0.05))
                    self.unc_upper = float(cfg.get('uncertainty_upper', self.threshold + 0.05))
                    self.disagree_thresh = float(cfg.get('disagreement_threshold', 0.40))
            except Exception:
                pass

    def predict_video(self, video_path):
        frames, metadata = self.frame_extractor.extract_frames(video_path)
        if not frames:
            raise ValueError(f"No frames could be extracted from video: {video_path}")

        # Ensure exact 16-frame sequence length for GRU temporal model
        original_count = len(frames)
        sample_frames = list(frames)
        while len(sample_frames) < 16:
            sample_frames.append(sample_frames[-1])
        sample_frames = sample_frames[:16]

        frame_tensors = []
        spatial_scores = []
        suspicious_timestamps = []
        fps = metadata.get('fps', 30.0)

        for i, frame in enumerate(sample_frames):
            import cv2
            img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(img_rgb)

            rgb_t = self.transforms(pil_img)
            fft_t = compute_fft_tensor(pil_img)

            frame_tensors.append(rgb_t)

            with torch.no_grad():
                s_out = self.spatial_model(rgb_t.unsqueeze(0).to(self.device), fft_t.unsqueeze(0).to(self.device))
                s_prob = F.softmax(s_out, dim=1).squeeze(0)[1].item()
                spatial_scores.append(s_prob)
                if s_prob >= self.threshold:
                    ts_seconds = round(i / (fps / 2.0) if fps > 0 else float(i), 2)
                    suspicious_timestamps.append(ts_seconds)

        seq_tensor = torch.stack(frame_tensors, dim=0).unsqueeze(0).to(self.device)  # (1, 16, 3, 224, 224)

        with torch.no_grad():
            t_out = self.temporal_model(seq_tensor)
            temporal_ai_prob = F.softmax(t_out, dim=1).squeeze(0)[1].item()

        spatial_ai_prob = float(np.mean(spatial_scores))
        fused_ai_prob = self.w_spatial * spatial_ai_prob + self.w_temporal * temporal_ai_prob
        fused_real_prob = float(1.0 - fused_ai_prob)

        disagreement = abs(spatial_ai_prob - temporal_ai_prob) >= self.disagree_thresh

        if disagreement or (self.unc_lower <= fused_ai_prob <= self.unc_upper):
            raw_result = 'UNCERTAIN'
            label = 'Uncertain'
            confidence_category = "Low Confidence"
            confidence_val = round(max(fused_ai_prob, fused_real_prob) * 100.0, 2)
        elif fused_ai_prob >= self.threshold:
            raw_result = 'AI-Generated'
            label = 'Likely AI-generated'
            confidence_val = round(fused_ai_prob * 100.0, 2)
            confidence_category = "High Confidence" if fused_ai_prob >= 0.80 else "Moderate Confidence"
        else:
            raw_result = 'Real'
            label = 'Likely real'
            confidence_val = round(fused_real_prob * 100.0, 2)
            confidence_category = "High Confidence" if fused_real_prob >= 0.80 else "Moderate Confidence"

        return {
            'type': 'video',
            'result': raw_result,
            'label': label,
            'ai_probability': round(fused_ai_prob * 100.0, 2),
            'real_probability': round(fused_real_prob * 100.0, 2),
            'confidence': confidence_val,
            'confidence_category': confidence_category,
            'spatial_probability': round(spatial_ai_prob * 100.0, 2),
            'temporal_probability': round(temporal_ai_prob * 100.0, 2),
            'frames_analyzed': len(sample_frames),
            'sequences_analyzed': 1,
            'suspicious_timestamps': suspicious_timestamps,
            'video_metadata': metadata
        }
