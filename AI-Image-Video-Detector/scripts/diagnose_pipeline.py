import os
import sys
import time
import json
import hashlib
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageOps
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.linear_model import LogisticRegression

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'backend')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from backend.services.image_detector import ImageDetectorService
from backend.ml.datasets.dataset import get_transforms, compute_fft_tensor, AIDetectionDataset
from backend.ml.models.efficientnet_detector import get_efficientnet_model

DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data'))
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'backend'))
MODEL_PATH = os.path.join(BASE_DIR, 'ml', 'models', 'best_model.pth')
CONFIG_PATH = os.path.join(BASE_DIR, 'ml', 'models', 'model_config.json')

def run_diagnostics():
    print("==================================================")
    print("1. AUDIT COMPLETE INFERENCE PIPELINE TRACE")
    print("==================================================")
    
    test_img_path = os.path.join(DATA_DIR, 'test', 'REAL', 'real_test_001.png')
    if not os.path.exists(test_img_path):
        # Fallback to any real image
        real_files = os.listdir(os.path.join(DATA_DIR, 'test', 'REAL'))
        test_img_path = os.path.join(DATA_DIR, 'test', 'REAL', real_files[0])

    pil_img = Image.open(test_img_path)
    orig_w, orig_h = pil_img.size
    color_mode = pil_img.mode

    service = ImageDetectorService()
    service_result = service.predict_image(test_img_path)

    # Detailed Step-by-step Tensor Inspection
    transforms = get_transforms(is_training=False)
    pil_rgb = pil_img.convert('RGB')
    rgb_t = transforms(pil_rgb).unsqueeze(0)
    fft_t = compute_fft_tensor(pil_rgb).unsqueeze(0)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = get_efficientnet_model(MODEL_PATH, pretrained=False).to(device)
    model.eval()

    with torch.no_grad():
        logits = model(rgb_t.to(device), fft_t.to(device))
        probs = F.softmax(logits, dim=1).squeeze(0).cpu().numpy()

    print(f"Original Image: {test_img_path}")
    print(f"Dimensions: {orig_w}x{orig_h} | Color Mode: {color_mode}")
    print(f"Resized Dimensions: 224x224")
    print(f"RGB Tensor Shape: {list(rgb_t.shape)} | Dtype: {rgb_t.dtype}")
    print(f"FFT Tensor Shape: {list(fft_t.shape)} | Dtype: {fft_t.dtype}")
    print(f"Model Path: {MODEL_PATH}")
    print(f"Raw Model Logits: {logits.squeeze(0).cpu().numpy().tolist()}")
    print(f"Softmax Probabilities: Real={probs[0]:.4f}, AI={probs[1]:.4f}")
    print(f"Service Response: {service_result}")

    print("\n==================================================")
    print("2. VERIFY MODEL CHECKPOINT & CLASS MAPPING")
    print("==================================================")
    mtime = os.path.getmtime(MODEL_PATH)
    mtime_str = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(mtime))
    print(f"Model File Path: {MODEL_PATH}")
    print(f"File Modification Time: {mtime_str}")

    with open(CONFIG_PATH, 'r') as f:
        cfg = json.load(f)
    print(f"Config Architecture: {cfg.get('model_architecture')}")
    print(f"Config Class Mapping: {cfg.get('class_mapping')}")
    print(f"Config Threshold: {cfg.get('classification_threshold')}")

    # Verify Class Mapping logic in forward pass: Index 0 = REAL, Index 1 = AI_GENERATED
    assert cfg['class_mapping']['0'] == 'REAL', "CRITICAL BUG: Index 0 is not REAL!"
    assert cfg['class_mapping']['1'] == 'AI_GENERATED', "CRITICAL BUG: Index 1 is not AI_GENERATED!"
    print("Class mapping verification: INDEX 0 = REAL, INDEX 1 = AI_GENERATED (VERIFIED CORRECT)")

    print("\n==================================================")
    print("3. TEST KNOWN TEST DATA (20 REAL, 20 AI)")
    print("==================================================")
    real_test_dir = os.path.join(DATA_DIR, 'test', 'REAL')
    ai_test_dir = os.path.join(DATA_DIR, 'test', 'AI_GENERATED')

    real_files = sorted([f for f in os.listdir(real_test_dir) if f.endswith('.png')])[:20]
    ai_files = sorted([f for f in os.listdir(ai_test_dir) if f.endswith('.png')])[:20]

    table_rows = []
    real_correct = 0
    ai_correct = 0

    print(f"{'Filename':<30} | {'Actual':<12} | {'Predicted':<12} | {'AI Prob':<10} | {'Correct'}")
    print("-" * 75)

    for f in real_files:
        path = os.path.join(real_test_dir, f)
        res = service.predict_image(path)
        pred = res['result']
        ai_p = res['ai_probability']
        is_corr = (pred == 'Real')
        if is_corr: real_correct += 1
        print(f"{f:<30} | {'REAL':<12} | {pred:<12} | {ai_p:<10.2f}% | {is_corr}")
        table_rows.append((f, 'REAL', pred, ai_p, is_corr))

    for f in ai_files:
        path = os.path.join(ai_test_dir, f)
        res = service.predict_image(path)
        pred = res['result']
        ai_p = res['ai_probability']
        is_corr = (pred == 'AI-Generated')
        if is_corr: ai_correct += 1
        print(f"{f:<30} | {'AI_GENERATED':<12} | {pred:<12} | {ai_p:<10.2f}% | {is_corr}")
        table_rows.append((f, 'AI_GENERATED', pred, ai_p, is_corr))

    real_acc = (real_correct / len(real_files)) * 100.0 if real_files else 0.0
    ai_acc = (ai_correct / len(ai_files)) * 100.0 if ai_files else 0.0
    total_acc = ((real_correct + ai_correct) / (len(real_files) + len(ai_files))) * 100.0

    print(f"\nInternal Test REAL Accuracy: {real_acc:.2f}% ({real_correct}/{len(real_files)})")
    print(f"Internal Test AI Accuracy:   {ai_acc:.2f}% ({ai_correct}/{len(ai_files)})")
    print(f"Internal Test Overall Acc:   {total_acc:.2f}%")

    print("\n==================================================")
    print("4. TEST EXTERNAL DATASET & GENERALIZATION")
    print("==================================================")
    ext_real_dir = os.path.join(DATA_DIR, 'external_test', 'REAL')
    ext_ai_dir = os.path.join(DATA_DIR, 'external_test', 'AI_GENERATED')

    ext_real_files = sorted([f for f in os.listdir(ext_real_dir) if f.endswith('.png')]) if os.path.exists(ext_real_dir) else []
    ext_ai_files = sorted([f for f in os.listdir(ext_ai_dir) if f.endswith('.png')]) if os.path.exists(ext_ai_dir) else []

    ext_real_corr = sum(1 for f in ext_real_files if service.predict_image(os.path.join(ext_real_dir, f))['result'] == 'Real')
    ext_ai_corr = sum(1 for f in ext_ai_files if service.predict_image(os.path.join(ext_ai_dir, f))['result'] == 'AI-Generated')

    ext_real_acc = (ext_real_corr / len(ext_real_files)) * 100.0 if ext_real_files else 0.0
    ext_ai_acc = (ext_ai_corr / len(ext_ai_files)) * 100.0 if ext_ai_files else 0.0
    ext_total_acc = ((ext_real_corr + ext_ai_corr) / (len(ext_real_files) + len(ext_ai_files))) * 100.0 if (ext_real_files + ext_ai_files) else 0.0

    print(f"External Test REAL Accuracy: {ext_real_acc:.2f}% ({ext_real_corr}/{len(ext_real_files)})")
    print(f"External Test AI Accuracy:   {ext_ai_acc:.2f}% ({ext_ai_corr}/{len(ext_ai_files)})")
    print(f"External Test Overall Acc:   {ext_total_acc:.2f}%")

    print("\n==================================================")
    print("5. CONTROLLED SHORTCUT & PREPROCESSING EXPERIMENTS")
    print("==================================================")
    # Experiment A: JPEG Compression Standardization
    # Compress all test images to JPEG quality Q=75 and re-evaluate
    exp_jpeg_dir = os.path.join(DATA_DIR, 'error_analysis', 'jpeg_shortcut_test')
    os.makedirs(exp_jpeg_dir, exist_ok=True)
    
    jpeg_correct = 0
    total_exp = 0

    for f in real_files:
        path = os.path.join(real_test_dir, f)
        img = Image.open(path).convert('RGB')
        jp_path = os.path.join(exp_jpeg_dir, f"jpg_{f}.jpg")
        img.save(jp_path, 'JPEG', quality=75)
        res = service.predict_image(jp_path)
        if res['result'] == 'Real': jpeg_correct += 1
        total_exp += 1

    for f in ai_files:
        path = os.path.join(ai_test_dir, f)
        img = Image.open(path).convert('RGB')
        jp_path = os.path.join(exp_jpeg_dir, f"jpg_{f}.jpg")
        img.save(jp_path, 'JPEG', quality=75)
        res = service.predict_image(jp_path)
        if res['result'] == 'AI-Generated': jpeg_correct += 1
        total_exp += 1

    jpeg_acc = (jpeg_correct / total_exp) * 100.0
    print(f"Original Test Set Accuracy: {total_acc:.2f}%")
    print(f"After JPEG Quality (Q=75) Standardization Accuracy: {jpeg_acc:.2f}%")
    print(f"Shortcut Impact Delta: {total_acc - jpeg_acc:.2f}%")

    print("\n==================================================")
    print("6. TRAIN / VAL / TEST LEAKAGE DETECTOR")
    print("==================================================")
    splits = ['train', 'validation', 'test', 'external_test']
    all_hashes = {}
    for s in splits:
        for c in ['REAL', 'AI_GENERATED']:
            d_path = os.path.join(DATA_DIR, s, c)
            if not os.path.exists(d_path): continue
            for fname in os.listdir(d_path):
                if not fname.endswith('.png'): continue
                f_path = os.path.join(d_path, fname)
                with open(f_path, 'rb') as fp:
                    h = hashlib.md5(fp.read()).hexdigest()
                if h not in all_hashes: all_hashes[h] = []
                all_hashes[h].append((s, c, fname))

    cross_split_leaks = [locs for locs in all_hashes.values() if len(set(loc[0] for loc in locs)) > 1]
    print(f"Total Images Scanned: {sum(len(v) for v in all_hashes.values())}")
    print(f"Total Unique MD5 Hashes: {len(all_hashes)}")
    print(f"Cross-Split Hash Collisions: {len(cross_split_leaks)}")

    print("\n==================================================")
    print("7. CLASS DISTRIBUTION AUDIT")
    print("==================================================")
    counts = {s: {c: len(os.listdir(os.path.join(DATA_DIR, s, c))) if os.path.exists(os.path.join(DATA_DIR, s, c)) else 0 for c in ['REAL', 'AI_GENERATED']} for s in splits}
    print(json.dumps(counts, indent=2))

    print("\n==================================================")
    print("11. PROBABILITY DISTRIBUTION & HIGH-CONFIDENCE ERRORS")
    print("==================================================")
    high_conf_dir = os.path.join(DATA_DIR, 'error_analysis', 'high_confidence_errors')
    os.makedirs(high_conf_dir, exist_ok=True)

    real_ai_probs = []
    ai_ai_probs = []
    high_conf_errors = []

    for f in real_files:
        path = os.path.join(real_test_dir, f)
        res = service.predict_image(path)
        real_ai_probs.append(res['ai_probability'])
        if res['ai_probability'] >= 80.0:
            high_conf_errors.append((f, 'REAL', res['ai_probability']))

    for f in ai_files:
        path = os.path.join(ai_test_dir, f)
        res = service.predict_image(path)
        ai_ai_probs.append(res['ai_probability'])
        if res['ai_probability'] <= 20.0:
            high_conf_errors.append((f, 'AI_GENERATED', res['ai_probability']))

    print(f"REAL images AI Probabilities: min={min(real_ai_probs):.2f}%, max={max(real_ai_probs):.2f}%, mean={np.mean(real_ai_probs):.2f}%")
    print(f"AI images AI Probabilities:   min={min(ai_ai_probs):.2f}%, max={max(ai_ai_probs):.2f}%, mean={np.mean(ai_ai_probs):.2f}%")
    print(f"High-Confidence Errors Count: {len(high_conf_errors)}")

    print("\n==================================================")
    print("13. SIMPLE BASELINE CLASSIFIER COMPARISON")
    print("==================================================")
    # Train a simple Logistic Regression baseline on mean/std image stats
    train_dataset = AIDetectionDataset(root_dir=DATA_DIR, split='train', is_training=False)
    test_dataset = AIDetectionDataset(root_dir=DATA_DIR, split='test', is_training=False)

    X_train, y_train = [], []
    for rgb_t, fft_t, lbl, _ in train_dataset:
        feat = torch.cat([rgb_t.mean(dim=(1, 2)), rgb_t.std(dim=(1, 2)), fft_t.mean(dim=(1, 2)), fft_t.std(dim=(1, 2))]).numpy()
        X_train.append(feat)
        y_train.append(lbl.item())

    X_test, y_test = [], []
    for rgb_t, fft_t, lbl, _ in test_dataset:
        feat = torch.cat([rgb_t.mean(dim=(1, 2)), rgb_t.std(dim=(1, 2)), fft_t.mean(dim=(1, 2)), fft_t.std(dim=(1, 2))]).numpy()
        X_test.append(feat)
        y_test.append(lbl.item())

    lr_model = LogisticRegression()
    lr_model.fit(X_train, y_train)
    lr_preds = lr_model.predict(X_test)
    lr_acc = accuracy_score(y_test, lr_preds) * 100.0

    print(f"Simple Baseline Logistic Regression Test Accuracy: {lr_acc:.2f}%")
    print(f"EfficientNet Detector Test Accuracy:                {total_acc:.2f}%")

if __name__ == '__main__':
    run_diagnostics()
