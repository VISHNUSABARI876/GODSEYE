try:
    from .dataset import AIDetectionDataset, get_transforms, compute_fft_tensor
except ImportError:
    from backend.ml.datasets.dataset import AIDetectionDataset, get_transforms, compute_fft_tensor

__all__ = ['AIDetectionDataset', 'get_transforms', 'compute_fft_tensor']
