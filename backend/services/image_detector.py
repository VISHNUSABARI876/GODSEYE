import os
import sys
import json
import torch
import torch.nn.functional as F
from PIL import Image

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

try:
    from backend.ml.models.efficientnet_detector import get_efficientnet_model
    from backend.ml.datasets.dataset import get_transforms, compute_fft_tensor
except ImportError:
    from ml.models.efficientnet_detector import get_efficientnet_model
    from ml.datasets.dataset import get_transforms, compute_fft_tensor

class ImageDetectorService:
    def __init__(self, model_path=None):
        base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
        
        if model_path:
            if not os.path.isabs(model_path):
                model_path = os.path.abspath(os.path.join(base_dir, model_path))
        else:
            model_path = os.path.abspath(os.path.join(base_dir, 'ml', 'models', 'best_model.pth'))

        self.model_path = model_path
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Image detection model checkpoint not found at: '{self.model_path}'. Ensure best_model.pth is committed and deployed.")

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = get_efficientnet_model(self.model_path, pretrained=False).to(self.device)
        self.model.eval()
        self.transforms = get_transforms(is_training=False)

        # Load optimal classification threshold and uncertainty bounds from model_config.json
        config_path = os.path.join(base_dir, 'ml', 'models', 'model_config.json')
        self.threshold = 0.50
        self.unc_lower = 0.45
        self.unc_upper = 0.55

        if os.path.exists(config_path):
            try:
                with open(config_path, 'r') as f:
                    cfg = json.load(f)
                    self.threshold = float(cfg.get('classification_threshold', 0.50))
                    self.unc_lower = float(cfg.get('uncertainty_lower', self.threshold - 0.05))
                    self.unc_upper = float(cfg.get('uncertainty_upper', self.threshold + 0.05))
            except Exception as e:
                print(f"Could not load threshold from config: {e}")

    def predict_image(self, image_input):
        """
        Runs PyTorch EfficientNet-B0 + 2D FFT Hybrid Vision Classifier.
        Returns dict with calibrated probabilities and probabilistic wording.
        """
        if isinstance(image_input, str):
            pil_img = Image.open(image_input).convert('RGB')
        elif isinstance(image_input, Image.Image):
            pil_img = image_input.convert('RGB')
        else:
            # OpenCV BGR numpy array
            import cv2
            img_rgb = cv2.cvtColor(image_input, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(img_rgb)

        rgb_tensor = self.transforms(pil_img).unsqueeze(0).to(self.device)
        fft_tensor = compute_fft_tensor(pil_img).unsqueeze(0).to(self.device)

        with torch.no_grad():
            logits = self.model(rgb_tensor, fft_tensor)
            probs = F.softmax(logits, dim=1).squeeze(0).cpu().numpy()

        real_prob = float(probs[0])
        ai_prob = float(probs[1])

        # Classification decision logic with UNCERTAIN category
        if self.unc_lower <= ai_prob <= self.unc_upper:
            raw_result = 'UNCERTAIN'
            label = 'Uncertain'
            confidence_val = round(max(ai_prob, real_prob) * 100.0, 2)
            confidence_category = "Low Confidence"
        elif ai_prob >= self.threshold:
            raw_result = 'AI-Generated'
            label = 'Likely AI-generated'
            confidence_val = round(ai_prob * 100.0, 2)
            confidence_category = "High Confidence" if ai_prob >= 0.80 else "Moderate Confidence"
        else:
            raw_result = 'Real'
            label = 'Likely real'
            confidence_val = round(real_prob * 100.0, 2)
            confidence_category = "High Confidence" if real_prob >= 0.80 else "Moderate Confidence"

        return {
            'result': raw_result,
            'label': label,
            'ai_probability': round(ai_prob * 100.0, 2),
            'real_probability': round(real_prob * 100.0, 2),
            'confidence': confidence_val,
            'confidence_category': confidence_category,
            'threshold_used': self.threshold
        }
