import os
import cv2
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset

try:
    from backend.ml.datasets.dataset import get_transforms
except ImportError:
    from ml.datasets.dataset import get_transforms

class VideoSequenceDataset(Dataset):
    """
    PyTorch Dataset for loading 16-frame video sequences from video files.
    Ensures video-level split isolation across train, validation, and test sets.
    """
    def __init__(self, root_dir, split='video_train', sequence_length=16, is_training=False):
        self.split_dir = os.path.join(root_dir, split)
        self.sequence_length = sequence_length
        self.is_training = is_training
        self.transforms = get_transforms(is_training=is_training)

        self.samples = []  # (video_path, label)

        real_dir = os.path.join(self.split_dir, 'real')
        ai_dir = os.path.join(self.split_dir, 'ai')

        if os.path.exists(real_dir):
            for fname in sorted(os.listdir(real_dir)):
                if fname.lower().endswith(('.mp4', '.avi', '.mov', '.mkv', '.webm')):
                    self.samples.append((os.path.join(real_dir, fname), 0))

        if os.path.exists(ai_dir):
            for fname in sorted(os.listdir(ai_dir)):
                if fname.lower().endswith(('.mp4', '.avi', '.mov', '.mkv', '.webm')):
                    self.samples.append((os.path.join(ai_dir, fname), 1))

    def __len__(self):
        return len(self.samples)

    def extract_sequence(self, video_path):
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            # Fallback black sequence if unreadable
            return torch.zeros((self.sequence_length, 3, 224, 224), dtype=torch.float32)

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or self.sequence_length
        indices = np.linspace(0, max(0, total_frames - 1), num=self.sequence_length, dtype=int)

        frame_tensors = []
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret and frame is not None:
                img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_img = Image.fromarray(img_rgb)
            else:
                pil_img = Image.new('RGB', (224, 224), color=0)

            frame_tensor = self.transforms(pil_img)  # (3, 224, 224)
            frame_tensors.append(frame_tensor)

        cap.release()

        # Stack sequence into (16, 3, 224, 224)
        sequence_tensor = torch.stack(frame_tensors, dim=0)
        return sequence_tensor

    def __getitem__(self, idx):
        video_path, label = self.samples[idx]
        sequence_tensor = self.extract_sequence(video_path)
        return sequence_tensor, torch.tensor(label, dtype=torch.long), video_path
