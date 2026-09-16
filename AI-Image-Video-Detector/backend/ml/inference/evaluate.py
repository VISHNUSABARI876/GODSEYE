import os
import sys
import shutil
import json
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, roc_curve
)

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

try:
    from backend.ml.models.efficientnet_detector import get_efficientnet_model
    from backend.ml.datasets.dataset import AIDetectionDataset
except ImportError:
    from ml.models.efficientnet_detector import get_efficientnet_model
    from ml.datasets.dataset import AIDetectionDataset

def evaluate_dataset(model, loader, threshold=0.50, uncertainty_margin=0.05, device='cpu'):
    model.eval()
    all_labels = []
    all_probs = []
    all_paths = []

    with torch.no_grad():
        for rgb_t, fft_t, labels, filepaths in loader:
            rgb_t, fft_t = rgb_t.to(device), fft_t.to(device)
            outputs = model(rgb_t, fft_t)
            probs = torch.softmax(outputs, dim=1)[:, 1]  # AI_GENERATED prob

            all_labels.extend(labels.numpy())
            all_probs.extend(probs.cpu().numpy())
            all_paths.extend(filepaths)

    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)
    
    # Binary predictions based on decision threshold
    preds = (all_probs >= threshold).astype(int)

    acc = accuracy_score(all_labels, preds)
    prec = precision_score(all_labels, preds, zero_division=0)
    rec = recall_score(all_labels, preds, zero_division=0)
    f1 = f1_score(all_labels, preds, zero_division=0)
    try:
        auc = roc_auc_score(all_labels, all_probs)
    except Exception:
        auc = 0.5

    cm = confusion_matrix(all_labels, preds, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)

    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0

    metrics = {
        'accuracy': round(float(acc * 100.0), 2),
        'precision': round(float(prec * 100.0), 2),
        'recall': round(float(rec * 100.0), 2),
        'f1_score': round(float(f1 * 100.0), 2),
        'roc_auc': round(float(auc), 4),
        'specificity': round(float(specificity * 100.0), 2),
        'sensitivity': round(float(sensitivity * 100.0), 2),
        'false_positive_rate': round(float(fpr * 100.0), 2),
        'false_negative_rate': round(float(fnr * 100.0), 2),
        'confusion_matrix': {'TN': int(tn), 'FP': int(fp), 'FN': int(fn), 'TP': int(tp)}
    }

    return metrics, all_labels, preds, all_probs, all_paths

def plot_confusion_matrix_and_roc(cm_dict, all_labels, all_probs, save_path):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # 1. Confusion Matrix Plot
    cm_matrix = np.array([
        [cm_dict['TN'], cm_dict['FP']],
        [cm_dict['FN'], cm_dict['TP']]
    ])
    im = ax1.imshow(cm_matrix, interpolation='nearest', cmap=plt.cm.Blues)
    ax1.set_title('Confusion Matrix (Unseen Test Set)')
    fig.colorbar(im, ax=ax1)
    tick_marks = np.arange(2)
    ax1.set_xticks(tick_marks)
    ax1.set_xticklabels(['REAL', 'AI_GENERATED'])
    ax1.set_yticks(tick_marks)
    ax1.set_yticklabels(['REAL', 'AI_GENERATED'])
    ax1.set_ylabel('Actual Label')
    ax1.set_xlabel('Predicted Label')

    for i in range(2):
        for j in range(2):
            ax1.text(j, i, str(cm_matrix[i, j]), horizontalalignment="center",
                     color="white" if cm_matrix[i, j] > cm_matrix.max() / 2. else "black",
                     fontsize=14, weight='bold')

    # 2. ROC Curve Plot
    try:
        fpr, tpr, _ = roc_curve(all_labels, all_probs)
        auc_val = roc_auc_score(all_labels, all_probs)
        ax2.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC Curve (AUC = {auc_val:.4f})')
        ax2.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        ax2.set_xlim([0.0, 1.0])
        ax2.set_ylim([0.0, 1.05])
        ax2.set_xlabel('False Positive Rate')
        ax2.set_ylabel('True Positive Rate')
        ax2.set_title('Receiver Operating Characteristic (ROC)')
        ax2.legend(loc="lower right")
    except Exception as e:
        ax2.text(0.5, 0.5, f"ROC Plot Error: {e}", ha='center', va='center')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved confusion matrix & ROC curve plot to: {save_path}")

def run_full_evaluation():
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
    data_dir = os.path.join(base_dir, 'data')
    models_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'models'))
    training_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'training'))
    os.makedirs(training_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)

    best_model_path = os.path.join(models_dir, 'best_model.pth')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print("=" * 60)
    print("STARTING MODEL EVALUATION & THRESHOLD CALIBRATION")
    print(f"Loading checkpoint: {best_model_path}")
    print("=" * 60)

    model = get_efficientnet_model(best_model_path, pretrained=False).to(device)

    # DataLoaders
    val_dataset = AIDetectionDataset(root_dir=data_dir, split='validation', is_training=False)
    test_dataset = AIDetectionDataset(root_dir=data_dir, split='test', is_training=False)
    ext_dataset = AIDetectionDataset(root_dir=data_dir, split='external_test', is_training=False)

    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)
    ext_loader = DataLoader(ext_dataset, batch_size=16, shuffle=False)

    # 1. Decision Threshold Optimization strictly on VALIDATION SET
    candidate_thresholds = [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70]
    best_thresh = 0.50
    best_val_f1 = -1.0

    print("\n--- DECISION THRESHOLD OPTIMIZATION (VALIDATION SET ONLY) ---")
    for t in candidate_thresholds:
        m, _, _, _, _ = evaluate_dataset(model, val_loader, threshold=t, device=device)
        print(f"Threshold T={t:.2f} -> Val Acc: {m['accuracy']}% | F1: {m['f1_score']}% | FPR: {m['false_positive_rate']}%")
        if m['f1_score'] > best_val_f1:
            best_val_f1 = m['f1_score']
            best_thresh = t

    # Set uncertainty window bounds around threshold (e.g. T +- 0.05)
    uncertainty_lower = max(0.0, round(best_thresh - 0.05, 2))
    uncertainty_upper = min(1.0, round(best_thresh + 0.05, 2))

    print(f"\n[SELECTED OPTIMAL THRESHOLD]: T = {best_thresh:.2f} (Val F1: {best_val_f1:.2f}%)")
    print(f"[UNCERTAINTY ZONE]: [{uncertainty_lower:.2f}, {uncertainty_upper:.2f}]")

    # Save Model Config
    model_config = {
        'model_architecture': 'EfficientNet-B0 + 2D FFT Frequency Hybrid Network',
        'input_size': [224, 224],
        'normalization': {
            'mean': [0.485, 0.456, 0.406],
            'std': [0.229, 0.224, 0.225]
        },
        'class_mapping': {'0': 'REAL', '1': 'AI_GENERATED'},
        'classification_threshold': best_thresh,
        'uncertainty_lower': uncertainty_lower,
        'uncertainty_upper': uncertainty_upper
    }

    config_path = os.path.join(models_dir, 'model_config.json')
    with open(config_path, 'w') as f:
        json.dump(model_config, f, indent=2)
    print(f"Saved config to {config_path}")

    # 2. Evaluate on Completely Unseen Internal Test Set
    test_metrics, test_labels, test_preds, test_probs, test_paths = evaluate_dataset(
        model, test_loader, threshold=best_thresh, device=device
    )

    # Save evaluation_results.json into backend/ml/training/
    eval_results_path = os.path.join(training_dir, 'evaluation_results.json')
    with open(eval_results_path, 'w') as f:
        json.dump(test_metrics, f, indent=2)
    print(f"Saved evaluation metrics to {eval_results_path}")

    # 3. Generate & Save Confusion Matrix PNG into backend/ml/training/
    cm_png_path = os.path.join(training_dir, 'confusion_matrix.png')
    plot_confusion_matrix_and_roc(test_metrics['confusion_matrix'], test_labels, test_probs, cm_png_path)

    # 4. Evaluate on External Test Set
    ext_metrics, _, _, _, _ = evaluate_dataset(
        model, ext_loader, threshold=best_thresh, device=device
    )

    print("\n" + "=" * 60)
    print("FINAL UNSEEN TEST SET PERFORMANCE")
    print("-" * 60)
    print(f"  * Accuracy:             {test_metrics['accuracy']}%")
    print(f"  * Precision:            {test_metrics['precision']}%")
    print(f"  * Recall:               {test_metrics['recall']}%")
    print(f"  * F1 Score:             {test_metrics['f1_score']}%")
    print(f"  * ROC-AUC:              {test_metrics['roc_auc']}")
    print(f"  * Specificity:          {test_metrics['specificity']}%")
    print(f"  * Sensitivity:          {test_metrics['sensitivity']}%")
    print(f"  * False Positive Rate:  {test_metrics['false_positive_rate']}%")
    print(f"  * False Negative Rate:  {test_metrics['false_negative_rate']}%")
    print(f"  * Confusion Matrix:     {test_metrics['confusion_matrix']}")
    print(f"  * External Test Acc:    {ext_metrics['accuracy']}%")

    # 5. Error Analysis Export
    error_dir = os.path.join(data_dir, 'error_analysis')
    fp_dir = os.path.join(error_dir, 'false_positive')
    fn_dir = os.path.join(error_dir, 'false_negative')
    os.makedirs(fp_dir, exist_ok=True)
    os.makedirs(fn_dir, exist_ok=True)

    for d in [fp_dir, fn_dir]:
        for f_name in os.listdir(d):
            p = os.path.join(d, f_name)
            if os.path.isfile(p):
                os.remove(p)

    fp_list = []
    fn_list = []
    detailed_test_records = []

    for i in range(len(test_labels)):
        true_lbl = int(test_labels[i])
        pred_lbl = int(test_preds[i])
        ai_prob = float(test_probs[i])
        real_prob = float(1.0 - ai_prob)
        path = test_paths[i]
        fname = os.path.basename(path)
        
        is_correct = (true_lbl == pred_lbl)
        
        record = {
            'filename': fname,
            'actual_label': 'AI_GENERATED' if true_lbl == 1 else 'REAL',
            'predicted_label': 'AI_GENERATED' if pred_lbl == 1 else 'REAL',
            'ai_probability': round(ai_prob, 4),
            'real_probability': round(real_prob, 4),
            'is_correct': is_correct
        }
        detailed_test_records.append(record)

        if true_lbl == 0 and pred_lbl == 1:
            shutil.copy(path, os.path.join(fp_dir, fname))
            fp_list.append((fname, ai_prob))
        elif true_lbl == 1 and pred_lbl == 0:
            shutil.copy(path, os.path.join(fn_dir, fname))
            fn_list.append((fname, ai_prob))

    # Save detailed test records
    test_details_path = os.path.join(error_dir, 'test_predictions_detail.json')
    with open(test_details_path, 'w') as f:
        json.dump(detailed_test_records, f, indent=2)

    # Write Markdown Error Report
    report_path = os.path.join(error_dir, 'report.md')
    with open(report_path, 'w') as f:
        f.write("# Error Analysis Report\n\n")
        f.write(f"- **Optimal Decision Threshold**: `{best_thresh}`\n")
        f.write(f"- **Uncertainty Region**: `[{uncertainty_lower}, {uncertainty_upper}]`\n")
        f.write(f"- **Unseen Test Accuracy**: `{test_metrics['accuracy']}%`\n")
        f.write(f"- **Unseen Test F1 Score**: `{test_metrics['f1_score']}%`\n")
        f.write(f"- **External Test Accuracy**: `{ext_metrics['accuracy']}%`\n\n")
        
        f.write("## False Positives (REAL misclassified as AI_GENERATED)\n")
        if fp_list:
            for fname, p in fp_list:
                f.write(f"- `{fname}` (AI Prob: {p*100:.2f}%)\n")
        else:
            f.write("- None (0 false positives on test set)\n")

        f.write("\n## False Negatives (AI_GENERATED misclassified as REAL)\n")
        if fn_list:
            for fname, p in fn_list:
                f.write(f"- `{fname}` (AI Prob: {p*100:.2f}%)\n")
        else:
            f.write("- None (0 false negatives on test set)\n")

    print("\n" + "=" * 60)
    print(f"EVALUATION & ERROR ANALYSIS COMPLETE. Saved to {report_path}")
    print("=" * 60)

    return {
        'threshold': best_thresh,
        'test_metrics': test_metrics,
        'ext_metrics': ext_metrics
    }

if __name__ == '__main__':
    run_full_evaluation()
