import os
import sys
import json
import torch
import torch.nn.functional as F
from PIL import Image

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

try:
    from backend.ml.models.efficientnet_detector import get_efficientnet_model
    from backend.ml.datasets.dataset import get_transforms, compute_fft_tensor
except ImportError:
    from ml.models.efficientnet_detector import get_efficientnet_model
    from ml.datasets.dataset import get_transforms, compute_fft_tensor

def predict_single_image(image_path):
    if not os.path.exists(image_path):
        print(f"Error: Image file does not exist at {image_path}")
        sys.exit(1)

    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
    models_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'models'))
    best_model_path = os.path.join(models_dir, 'best_model.pth')
    config_path = os.path.join(models_dir, 'model_config.json')

    # Default threshold configuration
    threshold = 0.50
    unc_lower = 0.45
    unc_upper = 0.55

    if os.path.exists(config_path):
        try:
            with open(config_path, 'r') as f:
                cfg = json.load(f)
                threshold = float(cfg.get('classification_threshold', 0.50))
                unc_lower = float(cfg.get('uncertainty_lower', threshold - 0.05))
                unc_upper = float(cfg.get('uncertainty_upper', threshold + 0.05))
        except Exception:
            pass

    # Determine ground truth label if in dataset folder
    norm_path = os.path.abspath(image_path)
    if 'AI_GENERATED' in norm_path:
        actual_label = 'AI_GENERATED'
    elif 'REAL' in norm_path:
        actual_label = 'REAL'
    else:
        actual_label = 'Unknown'

    # Load Model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = get_efficientnet_model(best_model_path, pretrained=False).to(device)
    model.eval()

    transforms = get_transforms(is_training=False)
    pil_img = Image.open(image_path).convert('RGB')

    rgb_tensor = transforms(pil_img).unsqueeze(0).to(device)
    fft_tensor = compute_fft_tensor(pil_img).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(rgb_tensor, fft_tensor)
        probs = F.softmax(logits, dim=1).squeeze(0).cpu().numpy()

    real_prob = float(probs[0])
    ai_prob = float(probs[1])

    # Uncertainty handling
    if unc_lower <= ai_prob <= unc_upper:
        prediction = "UNCERTAIN"
        confidence = "Low"
    elif ai_prob >= threshold:
        prediction = "AI_GENERATED"
        confidence = "High" if ai_prob >= 0.80 else "Moderate"
    else:
        prediction = "REAL"
        confidence = "High" if real_prob >= 0.80 else "Moderate"

    print("=" * 50)
    print(f"Image: {image_path}")
    print(f"Actual/Unknown: {actual_label}")
    print(f"Prediction: {prediction}")
    print(f"AI Probability: {ai_prob * 100.0:.2f}%")
    print(f"Real Probability: {real_prob * 100.0:.2f}%")
    print(f"Confidence: {confidence}")
    print(f"Threshold: {threshold:.2f}")
    print("=" * 50)

    return {
        'image': image_path,
        'actual': actual_label,
        'prediction': prediction,
        'ai_probability': round(ai_prob * 100.0, 2),
        'real_probability': round(real_prob * 100.0, 2),
        'confidence': confidence,
        'threshold': threshold
    }

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python -m backend.ml.inference.predict path/to/image.jpg")
        sys.exit(1)
    
    predict_single_image(sys.argv[1])
