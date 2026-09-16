import os
import json
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

try:
    from backend.ml.models.efficientnet_detector import EfficientNetDetector
    from backend.ml.datasets.dataset import AIDetectionDataset
except ImportError:
    from ml.models.efficientnet_detector import EfficientNetDetector
    from ml.datasets.dataset import AIDetectionDataset

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def train_detector():
    set_seed(42)

    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
    data_dir = os.path.join(base_dir, 'data')
    models_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'models'))
    os.makedirs(models_dir, exist_ok=True)

    print("=" * 60)
    print("STARTING EFFICIENTNET-B0 DETECTOR TRAINING")
    print(f"Data root: {data_dir}")
    print("=" * 60)

    # Prepare PyTorch Datasets
    train_dataset = AIDetectionDataset(root_dir=data_dir, split='train', is_training=True)
    val_dataset = AIDetectionDataset(root_dir=data_dir, split='validation', is_training=False)

    print(f"Train samples: {len(train_dataset)} | Validation samples: {len(val_dataset)}")

    if len(train_dataset) == 0:
        raise ValueError("Train dataset is empty! Run scripts/prepare_dataset.py first.")

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=0)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training device: {device}")

    # Initialize EfficientNet Detector
    model = EfficientNetDetector(pretrained=True).to(device)

    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)

    num_epochs = 15
    best_val_loss = float('inf')
    early_stop_patience = 5
    patience_counter = 0

    history = {
        'epoch': [],
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': [],
        'val_precision': [],
        'val_recall': [],
        'val_f1': [],
        'val_auc': []
    }

    best_model_path = os.path.join(models_dir, 'best_model.pth')

    for epoch in range(1, num_epochs + 1):
        # Training Phase
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0

        for rgb_t, fft_t, labels, _ in train_loader:
            rgb_t, fft_t, labels = rgb_t.to(device), fft_t.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(rgb_t, fft_t)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * rgb_t.size(0)
            preds = torch.argmax(outputs, dim=1)
            correct_train += (preds == labels).sum().item()
            total_train += labels.size(0)

        epoch_train_loss = running_loss / total_train
        epoch_train_acc = correct_train / total_train

        # Validation Phase
        model.eval()
        val_running_loss = 0.0
        all_val_labels = []
        all_val_preds = []
        all_val_probs = []

        with torch.no_grad():
            for rgb_t, fft_t, labels, _ in val_loader:
                rgb_t, fft_t, labels = rgb_t.to(device), fft_t.to(device), labels.to(device)

                outputs = model(rgb_t, fft_t)
                loss = criterion(outputs, labels)
                val_running_loss += loss.item() * rgb_t.size(0)

                probs = torch.softmax(outputs, dim=1)[:, 1]  # Prob of AI_GENERATED
                preds = torch.argmax(outputs, dim=1)

                all_val_labels.extend(labels.cpu().numpy())
                all_val_preds.extend(preds.cpu().numpy())
                all_val_probs.extend(probs.cpu().numpy())

        epoch_val_loss = val_running_loss / len(val_dataset)
        val_acc = accuracy_score(all_val_labels, all_val_preds)
        val_prec = precision_score(all_val_labels, all_val_preds, zero_division=0)
        val_rec = recall_score(all_val_labels, all_val_preds, zero_division=0)
        val_f1 = f1_score(all_val_labels, all_val_preds, zero_division=0)
        try:
            val_auc = roc_auc_score(all_val_labels, all_val_probs)
        except Exception:
            val_auc = 0.5

        scheduler.step(epoch_val_loss)

        # Log history
        history['epoch'].append(epoch)
        history['train_loss'].append(round(epoch_train_loss, 4))
        history['train_acc'].append(round(epoch_train_acc, 4))
        history['val_loss'].append(round(epoch_val_loss, 4))
        history['val_acc'].append(round(val_acc, 4))
        history['val_precision'].append(round(val_prec, 4))
        history['val_recall'].append(round(val_rec, 4))
        history['val_f1'].append(round(val_f1, 4))
        history['val_auc'].append(round(val_auc, 4))

        print(f"Epoch {epoch:02d}/{num_epochs:02d} | "
              f"Train Loss: {epoch_train_loss:.4f} Acc: {epoch_train_acc*100:.2f}% | "
              f"Val Loss: {epoch_val_loss:.4f} Acc: {val_acc*100:.2f}% F1: {val_f1:.4f} AUC: {val_auc:.4f}")

        # Checkpoint Best Model
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            patience_counter = 0
            torch.save(model.state_dict(), best_model_path)
            print(f"  -> Saved best model checkpoint (Val Loss: {best_val_loss:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= early_stop_patience:
                print(f"\nEarly stopping triggered after {epoch} epochs.")
                break

    # Save training history JSON
    history_path = os.path.join(models_dir, 'training_history.json')
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)

    print("\n" + "=" * 60)
    print(f"TRAINING COMPLETE. Best Checkpoint: {best_model_path}")
    print("=" * 60)

if __name__ == '__main__':
    train_detector()
