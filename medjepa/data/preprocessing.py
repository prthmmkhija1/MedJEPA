"""
Preprocessing pipeline for medical images.
Handles different formats and normalizes everything to a consistent format.
"""

import numpy as np
import torch
from PIL import Image
from pathlib import Path
import pydicom
from typing import Tuple, Optional


class MedicalImagePreprocessor:
    """
    Takes any medical image and converts it to a clean, normalized format.

    What "normalization" means:
    - Resize all images to the same size (e.g., 224x224)
    - Scale pixel values to 0-1 range (instead of 0-255 or 0-65535)
    - Handle grayscale vs color images consistently
    """

    def __init__(self, target_size: Tuple[int, int] = (224, 224)):
        """
        Args:
            target_size: What size to make all images. (224, 224) is standard for ViT.
        """
        self.target_size = target_size

    def load_image(self, path: str) -> np.ndarray:
        """
        Load a medical image from any common format.

        Args:
            path: Path to the image file

        Returns:
            numpy array of the image pixels
        """
        path = Path(path)
        suffix = path.suffix.lower()

        if suffix == ".dcm":
            # DICOM format (hospital standard)
            return self._load_dicom(path)
        elif suffix in [".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"]:
            # Regular image formats
            return self._load_standard_image(path)
        elif suffix == ".nii" or path.name.endswith(".nii.gz"):
            # NIfTI format (brain scans)
            return self._load_nifti(path)
        else:
            raise ValueError(f"Unknown image format: {suffix}")

    def _load_dicom(self, path: Path) -> np.ndarray:
        """Load a DICOM file and extract the pixel data."""
        ds = pydicom.dcmread(str(path))
        pixel_array = ds.pixel_array.astype(np.float32)
        return pixel_array

    def _load_standard_image(self, path: Path) -> np.ndarray:
        """Load a standard image file (JPG, PNG, etc.)."""
        img = Image.open(path)
        return np.array(img, dtype=np.float32)

    def _load_nifti(self, path: Path) -> np.ndarray:
        """Load a NIfTI file (3D brain/body scans)."""
        import nibabel as nib
        nii = nib.load(str(path))
        return nii.get_fdata().astype(np.float32)

    def normalize_intensity(self, image: np.ndarray) -> np.ndarray:
        """
        Scale pixel values to 0-1 range.

        Why: Different scanners produce different value ranges.
        An X-ray from Hospital A might have values 0-4095,
        while Hospital B produces 0-65535.
        Normalizing to 0-1 makes them comparable.
        """
        # Handle edge case: image is all one value
        img_min = image.min()
        img_max = image.max()
        if img_max - img_min == 0:
            return np.zeros_like(image)
        return (image - img_min) / (img_max - img_min)

    def resize_image(self, image: np.ndarray) -> np.ndarray:
        """Resize image to target size. Expects input in [0, 1] range."""
        # Clip to [0,1] for safety, then convert to uint8 for PIL
        image_clipped = np.clip(image, 0.0, 1.0)
        pil_image = Image.fromarray((image_clipped * 255).astype(np.uint8))
        pil_image = pil_image.resize(self.target_size, Image.LANCZOS)
        return np.array(pil_image, dtype=np.float32) / 255.0

    def ensure_3_channels(self, image: np.ndarray) -> np.ndarray:
        """
        Make sure image has 3 color channels (RGB).

        Why: X-rays are grayscale (1 channel), skin photos are color (3 channels).
        The model expects a consistent number of channels.
        We convert grayscale to 3 channels by repeating the single channel 3 times.
        """
        if image.ndim == 2:
            # Grayscale → repeat to make 3 channels
            image = np.stack([image, image, image], axis=-1)
        elif image.ndim == 3 and image.shape[-1] == 1:
            image = np.repeat(image, 3, axis=-1)
        elif image.ndim == 3 and image.shape[-1] == 4:
            # RGBA → drop alpha channel
            image = image[:, :, :3]
        return image

    def preprocess(self, path: str) -> np.ndarray:
        """
        Full preprocessing pipeline: load → normalize → resize → ensure 3 channels.

        Args:
            path: Path to any medical image file

        Returns:
            numpy array of shape (224, 224, 3) with values in [0, 1]
        """
        image = self.load_image(path)
        image = self.normalize_intensity(image)
        image = self.resize_image(image)
        image = self.ensure_3_channels(image)
        return image

    def preprocess_to_tensor(self, path: str) -> torch.Tensor:
        """
        Full pipeline: load → normalize → resize → 3ch → PyTorch tensor.

        Returns:
            torch.Tensor of shape (3, H, W) with values in [0, 1],
            ready for model input.
        """
        image = self.preprocess(path)  # (H, W, 3) numpy
        return torch.from_numpy(image).permute(2, 0, 1)  # (3, H, W) tensor


class VolumetricPreprocessor:
    """
    Preprocessor for 3D medical data (CT scans, MRI volumes).

    A CT scan is like a stack of 2D X-ray slices.
    Imagine slicing a loaf of bread — each slice is a 2D image,
    and the full loaf is a 3D volume.
    """

    def __init__(
        self,
        target_size: Tuple[int, int, int] = (128, 128, 64),
        # width=128, height=128, depth=64 slices
    ):
        self.target_size = target_size

    def load_nifti_volume(self, path: str) -> np.ndarray:
        """Load a full 3D volume from a NIfTI file."""
        import nibabel as nib
        nii = nib.load(path)
        volume = nii.get_fdata().astype(np.float32)
        return volume

    def normalize_volume(self, volume: np.ndarray) -> np.ndarray:
        """Normalize the entire 3D volume to 0-1 range."""
        v_min, v_max = volume.min(), volume.max()
        if v_max - v_min == 0:
            return np.zeros_like(volume)
        return (volume - v_min) / (v_max - v_min)

    def resize_volume(self, volume: np.ndarray) -> np.ndarray:
        """Resize 3D volume to target size using simple interpolation."""
        from scipy.ndimage import zoom

        current_shape = volume.shape
        zoom_factors = [
            t / c for t, c in zip(self.target_size, current_shape[:3])
        ]
        resized = zoom(volume, zoom_factors, order=1)  # Linear interpolation
        return resized

    def preprocess(self, path: str) -> np.ndarray:
        """Full 3D preprocessing pipeline."""
        volume = self.load_nifti_volume(path)
        volume = self.normalize_volume(volume)
        volume = self.resize_volume(volume)
        return volume
