import os
import sys
import json
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

try:
    from backend.ml.models.efficientnet_detector import get_efficientnet_model
    from backend.ml.models.temporal_gru_detector import get_video_temporal_model
    from backend.ml.datasets.dataset import get_transforms, compute_fft_tensor
    from backend.services.frame_extractor import FrameExtractor
except ImportError:
    from ml.models.efficientnet_detector import get_efficientnet_model
    from ml.models.temporal_gru_detector import get_video_temporal_model
    from ml.datasets.dataset import get_transforms, compute_fft_tensor
    from services.frame_extractor import FrameExtractor

def predict_single_video(video_path):
    if not os.path.exists(video_path):
        print(f"Error: Video file does not exist at {video_path}")
        sys.exit(1)

    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
    models_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'models'))
    spatial_ckpt_path = os.path.join(models_dir, 'best_model.pth')
    temporal_ckpt_path = os.path.join(models_dir, 'video_temporal_model.pth')
    fusion_cfg_path = os.path.join(models_dir, 'fusion_config.json')

    w_spatial = 0.50
    w_temporal = 0.50
    thresh = 0.30
    unc_lower = 0.25
    unc_upper = 0.35
    disagree_thresh = 0.40

    if os.path.exists(fusion_cfg_path):
        try:
            with open(fusion_cfg_path, 'r') as f:
                cfg = json.load(f)
                w_spatial = float(cfg.get('spatial_weight', 0.50))
                w_temporal = float(cfg.get('temporal_weight', 0.50))
                thresh = float(cfg.get('classification_threshold', 0.30))
                unc_lower = float(cfg.get('uncertainty_lower', thresh - 0.05))
                unc_upper = float(cfg.get('uncertainty_upper', thresh + 0.05))
                disagree_thresh = float(cfg.get('disagreement_threshold', 0.40))
        except Exception:
            pass

    # Extract Frames
    extractor = FrameExtractor(target_sample_fps=1.0, max_frames=16)
    frames, metadata = extractor.extract_frames(video_path)

    if len(frames) == 0:
        print(f"Error: Could not extract frames from video {video_path}")
        sys.exit(1)

    # Pad or sample exact 16 frames
    while len(frames) < 16:
        frames.append(frames[-1])
    frames = frames[:16]

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    spatial_model = get_efficientnet_model(spatial_ckpt_path, pretrained=False).to(device)
    temporal_model = get_video_temporal_model(temporal_ckpt_path, pretrained_spatial=False).to(device)

    spatial_model.eval()
    temporal_model.eval()

    transforms = get_transforms(is_training=False)

    frame_tensors = []
    fft_tensors = []
    spatial_scores = []
    suspicious_timestamps = []
    fps = metadata.get('fps', 30.0)

    for i, frame in enumerate(frames):
        import cv2
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(img_rgb)
        
        rgb_t = transforms(pil_img)
        fft_t = compute_fft_tensor(pil_img)
        
        frame_tensors.append(rgb_t)
        fft_tensors.append(fft_t)

        with torch.no_grad():
            s_out = spatial_model(rgb_t.unsqueeze(0).to(device), fft_t.unsqueeze(0).to(device))
            s_prob = F.softmax(s_out, dim=1).squeeze(0)[1].item()
            spatial_scores.append(s_prob)
            if s_prob >= thresh:
                ts_seconds = round(i / (fps / 2.0) if fps > 0 else float(i), 2)
                suspicious_timestamps.append(ts_seconds)

    seq_tensor = torch.stack(frame_tensors, dim=0).unsqueeze(0).to(device)  # (1, 16, 3, 224, 224)

    with torch.no_grad():
        t_out = temporal_model(seq_tensor)
        temporal_ai_prob = F.softmax(t_out, dim=1).squeeze(0)[1].item()

    spatial_ai_prob = float(np.mean(spatial_scores))
    fused_ai_prob = w_spatial * spatial_ai_prob + w_temporal * temporal_ai_prob
    fused_real_prob = float(1.0 - fused_ai_prob)

    disagreement = abs(spatial_ai_prob - temporal_ai_prob) >= disagree_thresh

    if disagreement or (unc_lower <= fused_ai_prob <= unc_upper):
        prediction = "UNCERTAIN"
        label = "Uncertain"
        confidence = "Low"
    elif fused_ai_prob >= thresh:
        prediction = "AI_GENERATED"
        label = "Likely AI-generated"
        confidence = "High" if fused_ai_prob >= 0.80 else "Moderate"
    else:
        prediction = "REAL"
        label = "Likely real"
        confidence = "High" if fused_real_prob >= 0.80 else "Moderate"

    norm_path = os.path.abspath(video_path).replace('\\', '/').lower()
    if '/ai/' in norm_path or 'ai_video' in norm_path:
        actual_label = "AI_GENERATED"
    elif '/real/' in norm_path or 'real_video' in norm_path:
        actual_label = "REAL"
    else:
        actual_label = "Unknown"

    print("=" * 50)
    print(f"Video: {video_path}")
    print(f"Actual/Unknown: {actual_label}")
    print(f"Prediction: {prediction}")
    print(f"AI Probability: {fused_ai_prob * 100.0:.2f}%")
    print(f"Real Probability: {fused_real_prob * 100.0:.2f}%")
    print(f"Spatial Probability: {spatial_ai_prob * 100.0:.2f}%")
    print(f"Temporal Probability: {temporal_ai_prob * 100.0:.2f}%")
    print(f"Confidence: {confidence}")
    print(f"Frames Analyzed: {len(frames)}")
    print(f"Sequences Analyzed: 1")
    print(f"Suspicious Timestamps: {suspicious_timestamps}")
    print("=" * 50)

    return {
        'type': 'video',
        'result': prediction,
        'label': label,
        'ai_probability': round(fused_ai_prob * 100.0, 2),
        'real_probability': round(fused_real_prob * 100.0, 2),
        'confidence': round(max(fused_ai_prob, fused_real_prob) * 100.0, 2),
        'confidence_category': f"{confidence} Confidence",
        'spatial_probability': round(spatial_ai_prob * 100.0, 2),
        'temporal_probability': round(temporal_ai_prob * 100.0, 2),
        'frames_analyzed': len(frames),
        'sequences_analyzed': 1,
        'suspicious_timestamps': suspicious_timestamps,
        'video_metadata': metadata
    }

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python -m backend.ml.inference.predict_video path/to/video.mp4")
        sys.exit(1)

    predict_single_video(sys.argv[1])
