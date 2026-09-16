import os
import sys
import json
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix
)

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

try:
    from backend.ml.models.efficientnet_detector import get_efficientnet_model
    from backend.ml.models.temporal_gru_detector import get_video_temporal_model
    from backend.ml.datasets.video_dataset import VideoSequenceDataset
    from backend.ml.datasets.dataset import compute_fft_tensor
except ImportError:
    from ml.models.efficientnet_detector import get_efficientnet_model
    from ml.models.temporal_gru_detector import get_video_temporal_model
    from ml.datasets.video_dataset import VideoSequenceDataset
    from ml.datasets.dataset import compute_fft_tensor

def evaluate_video_dataset(spatial_model, temporal_model, loader, w_spatial=0.5, w_temporal=0.5, threshold=0.30, device='cpu'):
    spatial_model.eval()
    temporal_model.eval()

    all_labels = []
    spatial_probs = []
    temporal_probs = []
    fused_probs = []
    file_paths = []

    with torch.no_grad():
        for seq_t, labels, paths in loader:
            seq_t = seq_t.to(device)  # (B, T, 3, 224, 224)
            b, t, c, h, w = seq_t.shape

            # 1. Temporal Model Forward Pass
            temp_logits = temporal_model(seq_t)
            temp_p = torch.softmax(temp_logits, dim=1)[:, 1].cpu().numpy()

            # 2. Spatial Model Frame-by-Frame Average
            frames_flat = seq_t.view(b * t, c, h, w)
            # Create zero fft dummy tensors for spatial evaluation fallback
            fft_flat = torch.zeros((b * t, 1, 224, 224), device=device)
            spat_logits = spatial_model(frames_flat, fft_flat)
            spat_p_flat = torch.softmax(spat_logits, dim=1)[:, 1].view(b, t).cpu().numpy()
            spat_p = spat_p_flat.mean(axis=1)

            # 3. Fused Probability
            fused_p = w_spatial * spat_p + w_temporal * temp_p

            all_labels.extend(labels.numpy())
            spatial_probs.extend(spat_p)
            temporal_probs.extend(temp_p)
            fused_probs.extend(fused_p)
            file_paths.extend(paths)

    all_labels = np.array(all_labels)
    spatial_probs = np.array(spatial_probs)
    temporal_probs = np.array(temporal_probs)
    fused_probs = np.array(fused_probs)

    def calc_metrics(probs, thresh):
        preds = (probs >= thresh).astype(int)
        acc = accuracy_score(all_labels, preds) * 100.0
        prec = precision_score(all_labels, preds, zero_division=0) * 100.0
        rec = recall_score(all_labels, preds, zero_division=0) * 100.0
        f1 = f1_score(all_labels, preds, zero_division=0) * 100.0
        try:
            auc = roc_auc_score(all_labels, probs)
        except Exception:
            auc = 0.5
        return {'accuracy': round(acc, 2), 'precision': round(prec, 2), 'recall': round(rec, 2), 'f1': round(f1, 2), 'auc': round(auc, 4)}

    spatial_m = calc_metrics(spatial_probs, threshold)
    temporal_m = calc_metrics(temporal_probs, threshold)
    fused_m = calc_metrics(fused_probs, threshold)

    return spatial_m, temporal_m, fused_m, all_labels, spatial_probs, temporal_probs, fused_probs, file_paths

def run_video_evaluation():
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
    data_dir = os.path.join(base_dir, 'data')
    models_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'models'))
    training_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'training'))
    error_dir = os.path.join(data_dir, 'video_error_analysis')
    os.makedirs(training_dir, exist_ok=True)
    os.makedirs(error_dir, exist_ok=True)

    spatial_model_path = os.path.join(models_dir, 'best_model.pth')
    temporal_model_path = os.path.join(models_dir, 'video_temporal_model.pth')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print("=" * 60)
    print("STARTING VIDEO TEMPORAL & FUSION EVALUATION")
    print("=" * 60)

    spatial_model = get_efficientnet_model(spatial_model_path, pretrained=False).to(device)
    temporal_model = get_video_temporal_model(temporal_model_path, pretrained_spatial=False).to(device)

    val_dataset = VideoSequenceDataset(root_dir=data_dir, split='video_validation', sequence_length=16, is_training=False)
    test_dataset = VideoSequenceDataset(root_dir=data_dir, split='video_test', sequence_length=16, is_training=False)
    ext_dataset = VideoSequenceDataset(root_dir=data_dir, split='video_external_test', sequence_length=16, is_training=False)

    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False)
    ext_loader = DataLoader(ext_dataset, batch_size=4, shuffle=False)

    # 1. Fusion Weights Optimization on VALIDATION SET
    candidate_weights = [(0.25, 0.75), (0.40, 0.60), (0.50, 0.50), (0.60, 0.40), (0.75, 0.25)]
    best_w = (0.50, 0.50)
    best_val_f1 = -1.0

    print("\n--- FUSION WEIGHTS OPTIMIZATION (VALIDATION SET ONLY) ---")
    for ws, wt in candidate_weights:
        _, _, fm, _, _, _, _, _ = evaluate_video_dataset(spatial_model, temporal_model, val_loader, w_spatial=ws, w_temporal=wt, threshold=0.30, device=device)
        print(f"Weights (Spatial={ws:.2f}, Temporal={wt:.2f}) -> Val Acc: {fm['accuracy']}% | F1: {fm['f1']}% | AUC: {fm['auc']}")
        if fm['f1'] > best_val_f1:
            best_val_f1 = fm['f1']
            best_w = (ws, wt)

    best_w_spatial, best_w_temporal = best_w
    print(f"\n[SELECTED FUSION WEIGHTS]: Spatial={best_w_spatial:.2f}, Temporal={best_w_temporal:.2f} (Val F1: {best_val_f1:.2f}%)")

    # Save Fusion Config
    fusion_config = {
        'spatial_weight': best_w_spatial,
        'temporal_weight': best_w_temporal,
        'classification_threshold': 0.30,
        'uncertainty_lower': 0.25,
        'uncertainty_upper': 0.35,
        'disagreement_threshold': 0.40,
        'sequence_length': 16
    }
    config_path = os.path.join(models_dir, 'fusion_config.json')
    with open(config_path, 'w') as f:
        json.dump(fusion_config, f, indent=2)

    # 2. Evaluate on Unseen Video Test Set
    spat_m, temp_m, fused_m, test_labels, spat_probs, temp_probs, fused_probs, test_paths = evaluate_video_dataset(
        spatial_model, temporal_model, test_loader, w_spatial=best_w_spatial, w_temporal=best_w_temporal, threshold=0.30, device=device
    )

    # 3. Evaluate External Video Set
    _, _, ext_fused_m, _, _, _, _, _ = evaluate_video_dataset(
        spatial_model, temporal_model, ext_loader, w_spatial=best_w_spatial, w_temporal=best_w_temporal, threshold=0.30, device=device
    )

    print("\n" + "=" * 60)
    print("UNSEEN VIDEO TEST SET PERFORMANCE COMPARISON")
    print("-" * 60)
    print(f"CURRENT FRAME MODEL:  Acc={spat_m['accuracy']}% | Prec={spat_m['precision']}% | Rec={spat_m['recall']}% | F1={spat_m['f1']}% | AUC={spat_m['auc']}")
    print(f"NEW TEMPORAL MODEL:   Acc={temp_m['accuracy']}% | Prec={temp_m['precision']}% | Rec={temp_m['recall']}% | F1={temp_m['f1']}% | AUC={temp_m['auc']}")
    print(f"FUSED MODEL:          Acc={fused_m['accuracy']}% | Prec={fused_m['precision']}% | Rec={fused_m['recall']}% | F1={fused_m['f1']}% | AUC={fused_m['auc']}")
    print(f"EXTERNAL TEST ACC:    {ext_fused_m['accuracy']}%")

    eval_json = {
        'current_frame_model': spat_m,
        'new_temporal_model': temp_m,
        'fused_model': fused_m,
        'external_test_fused_accuracy': ext_fused_m['accuracy'],
        'selected_fusion_weights': {'spatial': best_w_spatial, 'temporal': best_w_temporal}
    }
    results_path = os.path.join(training_dir, 'video_evaluation_results.json')
    with open(results_path, 'w') as f:
        json.dump(eval_json, f, indent=2)

    # 4. Video Error & Disagreement Analysis
    report_path = os.path.join(error_dir, 'report.md')
    disagreements = []
    with open(report_path, 'w') as f:
        f.write("# Video Detection Error Analysis & Model Comparison Report\n\n")
        f.write("## Model Performance Comparison (Unseen Video Test Set)\n")
        f.write(f"- **Current Spatial Frame Model Accuracy**: `{spat_m['accuracy']}%` (F1: `{spat_m['f1']}%`)\n")
        f.write(f"- **New Temporal GRU Model Accuracy**: `{temp_m['accuracy']}%` (F1: `{temp_m['f1']}%`)\n")
        f.write(f"- **Fused Dual-Branch Model Accuracy**: `{fused_m['accuracy']}%` (F1: `{fused_m['f1']}%`)\n")
        f.write(f"- **External Unseen Video Accuracy**: `{ext_fused_m['accuracy']}%`\n\n")

        f.write("## Detailed Test Video Inspection\n")
        f.write("| Video Filename | Actual | Spatial AI Prob | Temporal AI Prob | Fused Prob | Prediction | Disagreement |\n")
        f.write("| :--- | :--- | :--- | :--- | :--- | :--- | :--- |\n")

        for i in range(len(test_labels)):
            v_name = os.path.basename(test_paths[i])
            actual = 'AI' if test_labels[i] == 1 else 'REAL'
            sp = spat_probs[i] * 100.0
            tp = temp_probs[i] * 100.0
            fp = fused_probs[i] * 100.0
            disagree = abs(sp - tp) >= 40.0
            pred = 'AI-Generated' if fp >= 30.0 else 'Real'
            if disagree:
                disagreements.append((v_name, actual, sp, tp))

            f.write(f"| `{v_name}` | {actual} | {sp:.2f}% | {tp:.2f}% | {fp:.2f}% | {pred} | {'YES' if disagree else 'NO'} |\n")

        f.write("\n## Model Disagreement Analysis\n")
        if disagreements:
            for v_name, actual, sp, tp in disagreements:
                f.write(f"- `{v_name}` (Actual: {actual}) -> Spatial={sp:.2f}% vs Temporal={tp:.2f}%\n")
        else:
            f.write("- None (Spatial and Temporal models showed strong consensus on test set)\n")

    print("\n" + "=" * 60)
    print(f"VIDEO EVALUATION COMPLETE. Saved report to {report_path}")
    print("=" * 60)

if __name__ == '__main__':
    run_video_evaluation()
