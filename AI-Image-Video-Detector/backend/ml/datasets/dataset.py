import os
import cv2
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as T

# Standard ImageNet Mean and Standard Deviation for EfficientNet
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

def get_transforms(is_training=False):
    """
    Returns torchvision transforms for Spatial RGB Branch.
    Ensures IDENTICAL normalization across training, validation, testing, and web API inference.
    """
    if is_training:
        return T.Compose([
            T.Resize((224, 224)),
            T.RandomHorizontalFlip(p=0.5),
            T.RandomRotation(degrees=15),
            T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.15),
            T.GaussianBlur(kernel_size=(3, 3), sigma=(0.1, 1.0)),
            T.ToTensor(),
            T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
        ])
    else:
        return T.Compose([
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
        ])

def compute_fft_tensor(pil_img, target_size=(224, 224)):
    """
    Computes normalized 2D FFT magnitude spectrum tensor from PIL Image.
    """
    img_np = np.array(pil_img.convert('RGB'))
    img_resized = cv2.resize(img_np, target_size, interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(img_resized, cv2.COLOR_RGB2GRAY)
    
    fft_2d = np.fft.fft2(gray)
    fft_shift = np.fft.fftshift(fft_2d)
    magnitude = np.log(np.abs(fft_shift) + 1.0)
    
    max_val = np.max(magnitude)
    if max_val > 0:
        magnitude = magnitude / max_val
        
    mag_tensor = torch.from_numpy(magnitude).float().unsqueeze(0)  # (1, 224, 224)
    return mag_tensor


class AIDetectionDataset(Dataset):
    """
    PyTorch Dataset loading REAL (0) and AI_GENERATED (1) image partitions.
    """
    def __init__(self, root_dir, split='train', is_training=False):
        self.split_dir = os.path.join(root_dir, split)
        self.is_training = is_training
        self.transforms = get_transforms(is_training=is_training)
        
        self.samples = []  # List of tuples (filepath, label)
        
        real_dir = os.path.join(self.split_dir, 'REAL')
        ai_dir = os.path.join(self.split_dir, 'AI_GENERATED')
        
        if os.path.exists(real_dir):
            for fname in os.listdir(real_dir):
                if fname.lower().endswith(('.png', '.jpg', '.jpeg', '.webp', '.bmp')):
                    self.samples.append((os.path.join(real_dir, fname), 0))
                    
        if os.path.exists(ai_dir):
            for fname in os.listdir(ai_dir):
                if fname.lower().endswith(('.png', '.jpg', '.jpeg', '.webp', '.bmp')):
                    self.samples.append((os.path.join(ai_dir, fname), 1))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        filepath, label = self.samples[idx]
        try:
            pil_img = Image.open(filepath).convert('RGB')
        except Exception as e:
            # Fallback black image if corrupted
            pil_img = Image.new('RGB', (224, 224), color=0)

        rgb_tensor = self.transforms(pil_img)
        fft_tensor = compute_fft_tensor(pil_img)

        return rgb_tensor, fft_tensor, torch.tensor(label, dtype=torch.long), filepath
