import os
import sys
import json
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

try:
    from backend.ml.models.temporal_gru_detector import GRUTemporalModel, SpatialFeatureExtractor
    from backend.ml.datasets.video_dataset import VideoSequenceDataset
except ImportError:
    from ml.models.temporal_gru_detector import GRUTemporalModel, SpatialFeatureExtractor
    from ml.datasets.video_dataset import VideoSequenceDataset

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def extract_dataset_embeddings(spatial_extractor, dataset, device, batch_size=2):
    spatial_extractor.eval()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    all_embeddings = []
    all_labels = []

    with torch.no_grad():
        for seq_t, labels, _ in loader:
            b, t, c, h, w = seq_t.shape
            seq_flat = seq_t.view(b * t, c, h, w).to(device)
            feats = spatial_extractor(seq_flat)  # (B*T, 1280)
            feats_seq = feats.view(b, t, -1).cpu()  # (B, T, 1280)
            
            all_embeddings.append(feats_seq)
            all_labels.append(labels)

    X_tensor = torch.cat(all_embeddings, dim=0)
    y_tensor = torch.cat(all_labels, dim=0)
    return TensorDataset(X_tensor, y_tensor)

def train_video_detector():
    set_seed(42)

    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
    data_dir = os.path.join(base_dir, 'data')
    models_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'models'))
    os.makedirs(models_dir, exist_ok=True)

    print("=" * 60)
    print("STARTING HIGH-SPEED TEMPORAL GRU VIDEO DETECTOR TRAINING")
    print(f"Data root: {data_dir}")
    print("=" * 60)

    train_raw_dataset = VideoSequenceDataset(root_dir=data_dir, split='video_train', sequence_length=16, is_training=True)
    val_raw_dataset = VideoSequenceDataset(root_dir=data_dir, split='video_validation', sequence_length=16, is_training=False)

    print(f"Train video sequences: {len(train_raw_dataset)} | Val video sequences: {len(val_raw_dataset)}")

    if len(train_raw_dataset) == 0:
        raise ValueError("Video train dataset is empty! Run scripts/prepare_video_dataset.py first.")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training device: {device}")

    # Extract Spatial Features once
    print("Extracting EfficientNet-B0 spatial embeddings for train/val sequences...")
    spatial_extractor = SpatialFeatureExtractor(pretrained=True).to(device)
    
    train_dataset = extract_dataset_embeddings(spatial_extractor, train_raw_dataset, device)
    val_dataset = extract_dataset_embeddings(spatial_extractor, val_raw_dataset, device)

    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)

    model = GRUTemporalModel(pretrained_spatial=False).to(device)
    # Re-assign spatial extractor
    model.spatial_extractor = spatial_extractor

    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)

    num_epochs = 15
    best_val_loss = float('inf')
    early_stop_patience = 5
    patience_counter = 0

    history = {
        'epoch': [], 'train_loss': [], 'train_acc': [],
        'val_loss': [], 'val_acc': [], 'val_precision': [], 'val_recall': [], 'val_f1': [], 'val_auc': []
    }

    best_model_path = os.path.join(models_dir, 'video_temporal_model.pth')

    for epoch in range(1, num_epochs + 1):
        # Training Phase
        model.train()
        model.spatial_extractor.eval()
        running_loss = 0.0
        correct_train = 0
        total_train = 0

        for feats_seq, labels in train_loader:
            feats_seq, labels = feats_seq.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(None, precomputed_embeddings=feats_seq)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * feats_seq.size(0)
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
            for feats_seq, labels in val_loader:
                feats_seq, labels = feats_seq.to(device), labels.to(device)
                outputs = model(None, precomputed_embeddings=feats_seq)
                loss = criterion(outputs, labels)
                val_running_loss += loss.item() * feats_seq.size(0)

                probs = torch.softmax(outputs, dim=1)[:, 1]
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

        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            patience_counter = 0
            torch.save(model.state_dict(), best_model_path)
            print(f"  -> Saved best GRU temporal checkpoint (Val Loss: {best_val_loss:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= early_stop_patience:
                print(f"\nEarly stopping triggered after {epoch} epochs.")
                break

    history_path = os.path.join(models_dir, 'video_training_history.json')
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)

    print("\n" + "=" * 60)
    print(f"VIDEO TEMPORAL TRAINING COMPLETE. Saved to {best_model_path}")
    print("=" * 60)

if __name__ == '__main__':
    train_video_detector()
