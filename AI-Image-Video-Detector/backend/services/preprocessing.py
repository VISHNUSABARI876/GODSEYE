import cv2
import torch
import numpy as np
from PIL import Image

class ImagePreprocessor:
    def __init__(self, target_size=(224, 224)):
        self.target_size = target_size
        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    def preprocess_image(self, image_input):
        """
        Accepts PIL Image, file path, or OpenCV BGR numpy array.
        Returns:
            rgb_tensor: (1, 3, 224, 224) torch FloatTensor
            fft_tensor: (1, 1, 224, 224) torch FloatTensor
        """
        if isinstance(image_input, str):
            # File path
            img = cv2.imread(image_input)
            if img is None:
                raise ValueError(f"Could not read image file: {image_input}")
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        elif isinstance(image_input, Image.Image):
            img_rgb = np.array(image_input.convert('RGB'))
        elif isinstance(image_input, np.ndarray):
            if len(image_input.shape) == 3 and image_input.shape[2] == 3:
                # Assuming BGR from OpenCV unless RGB
                img_rgb = cv2.cvtColor(image_input, cv2.COLOR_BGR2RGB)
            else:
                img_rgb = image_input
        else:
            raise TypeError("Unsupported image_input type")

        # Resize to target size
        img_resized = cv2.resize(img_rgb, self.target_size, interpolation=cv2.INTER_AREA)

        # 1. Spatial Preprocessing (Normalize [0, 1] and standard ImageNet mean/std)
        img_normalized = (img_resized.astype(np.float32) / 255.0 - self.mean) / self.std
        # HWC to CHW
        img_chw = np.transpose(img_normalized, (2, 0, 1))
        rgb_tensor = torch.from_numpy(img_chw).float().unsqueeze(0)  # (1, 3, 224, 224)

        # 2. Frequency Domain Preprocessing (Grayscale 2D FFT)
        gray = cv2.cvtColor(img_resized, cv2.COLOR_RGB2GRAY)
        fft_2d = np.fft.fft2(gray)
        fft_shift = np.fft.fftshift(fft_2d)
        magnitude_spectrum = np.log(np.abs(fft_shift) + 1.0)
        
        # Normalize magnitude spectrum to [0, 1]
        fft_max = np.max(magnitude_spectrum)
        if fft_max > 0:
            magnitude_spectrum = magnitude_spectrum / fft_max

        fft_chw = np.expand_dims(magnitude_spectrum.astype(np.float32), axis=0)
        fft_tensor = torch.from_numpy(fft_chw).float().unsqueeze(0)  # (1, 1, 224, 224)

        return rgb_tensor, fft_tensor
