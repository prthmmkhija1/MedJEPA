"""
Preprocessing pipeline for medical images.
Handles different formats and normalizes everything to a consistent format.
"""

import numpy as np
import torch
from PIL import Image
from pathlib import Path
import pydicom
from typing import Tuple, Optional, Dict


# ----------------------------------------------------------------
# CT Window Presets  (center, width) in Hounsfield Units
# ----------------------------------------------------------------
CT_WINDOW_PRESETS: Dict[str, Tuple[float, float]] = {
    "soft_tissue": (40.0, 400.0),
    "lung":        (-600.0, 1500.0),
    "bone":        (400.0, 1800.0),
    "brain":       (40.0, 80.0),
    "liver":       (60.0, 150.0),
    "mediastinum": (50.0, 350.0),
}


def apply_ct_window(
    image: np.ndarray,
    center: float,
    width: float,
) -> np.ndarray:
    """
    Apply a CT windowing (window center / window width) to a Hounsfield-unit
    image and rescale the result to [0, 1].

    Pixels below (center - width/2) become 0; pixels above (center + width/2)
    become 1.  Everything in between is linearly mapped.
    """
    low = center - width / 2.0
    high = center + width / 2.0
    windowed = np.clip(image, low, high)
    return (windowed - low) / max(high - low, 1e-8)


class MedicalImagePreprocessor:
    """
    Takes any medical image and converts it to a clean, normalized format.

    What "normalization" means:
    - Resize all images to the same size (e.g., 224x224)
    - Scale pixel values to 0-1 range (instead of 0-255 or 0-65535)
    - Handle grayscale vs color images consistently
    """

    def __init__(
        self,
        target_size: Tuple[int, int] = (224, 224),
        normalization: str = "minmax",
        # "minmax"  — simple min-max scaling to [0, 1]  (default, works for all)
        # "zscore"  — zero-mean unit-variance (useful for MRI)
        # "ct_window" — CT Hounsfield windowing (see ct_window_preset)
        ct_window_preset: str = "soft_tissue",
        # Only used when normalization="ct_window".  One of the keys in
        # CT_WINDOW_PRESETS, or pass custom (center, width).
    ):
        """
        Args:
            target_size: What size to make all images. (224, 224) is standard for ViT.
            normalization: Intensity normalization strategy.
            ct_window_preset: Preset name for CT windowing.
        """
        self.target_size = target_size
        self.normalization = normalization
        self.ct_window_preset = ct_window_preset

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
        Scale pixel values according to the selected normalization strategy.

        - minmax: Maps [min, max] → [0, 1].
        - zscore: Zero-mean, unit-variance (then clipped to roughly [0, 1]).
        - ct_window: Applies CT windowing preset then maps to [0, 1].
        """
        if self.normalization == "zscore":
            mean = image.mean()
            std = image.std()
            if std < 1e-8:
                return np.zeros_like(image)
            normed = (image - mean) / std
            # Clip to [-3, 3] standard deviations and rescale to [0, 1]
            normed = np.clip(normed, -3.0, 3.0)
            return (normed + 3.0) / 6.0

        elif self.normalization == "ct_window":
            preset = CT_WINDOW_PRESETS.get(
                self.ct_window_preset,
                CT_WINDOW_PRESETS["soft_tissue"],
            )
            return apply_ct_window(image, center=preset[0], width=preset[1])

        else:  # minmax (default)
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
        return torch.tensor(image, dtype=torch.float32).permute(2, 0, 1)  # (3, H, W) tensor


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
        target_spacing: Optional[Tuple[float, float, float]] = None,
        # If set (e.g. (1.0, 1.0, 1.0) mm), volumes are resampled to isotropic
        # voxel spacing *before* resizing to target_size. Requires the NIfTI
        # header to contain valid affine/spacing info.
        normalization: str = "minmax",
        # "minmax" — default; "zscore" — zero-mean unit-variance;
        # "ct_window" — CT Hounsfield windowing (see ct_window_preset)
        ct_window_preset: str = "soft_tissue",
    ):
        self.target_size = target_size
        self.target_spacing = target_spacing
        self.normalization = normalization
        self.ct_window_preset = ct_window_preset

    def load_nifti_volume(self, path: str) -> np.ndarray:
        """Load a full 3D volume from a NIfTI file."""
        import nibabel as nib
        nii = nib.load(path)
        volume = nii.get_fdata().astype(np.float32)
        return volume

    def load_nifti_with_header(self, path: str):
        """Load a NIfTI file and return (volume, header, affine)."""
        import nibabel as nib
        nii = nib.load(path)
        return nii.get_fdata().astype(np.float32), nii.header, nii.affine

    def resample_to_spacing(
        self,
        volume: np.ndarray,
        current_spacing: Tuple[float, float, float],
        target_spacing: Tuple[float, float, float],
    ) -> np.ndarray:
        """
        Resample a volume so that each voxel has the desired physical spacing.

        This is important because different scanners produce volumes with
        different voxel sizes (e.g., 0.5mm × 0.5mm × 5mm vs 1mm × 1mm × 1mm).
        Resampling to a uniform spacing makes volumes comparable.
        """
        from scipy.ndimage import zoom

        zoom_factors = [
            cs / ts for cs, ts in zip(current_spacing, target_spacing)
        ]
        resampled = zoom(volume, zoom_factors, order=1)
        return resampled

    def normalize_volume(self, volume: np.ndarray) -> np.ndarray:
        """Normalize the entire 3D volume using the selected strategy."""
        if self.normalization == "zscore":
            mean = volume.mean()
            std = volume.std()
            if std < 1e-8:
                return np.zeros_like(volume)
            normed = (volume - mean) / std
            normed = np.clip(normed, -3.0, 3.0)
            return (normed + 3.0) / 6.0

        elif self.normalization == "ct_window":
            preset = CT_WINDOW_PRESETS.get(
                self.ct_window_preset,
                CT_WINDOW_PRESETS["soft_tissue"],
            )
            return apply_ct_window(volume, center=preset[0], width=preset[1])

        else:  # minmax
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
        """
        Full 3D preprocessing pipeline.

        If ``target_spacing`` is configured, spacing-aware resampling is
        applied before resizing (requires the NIfTI header).
        """
        import nibabel as nib

        nii = nib.load(path)
        volume = nii.get_fdata().astype(np.float32)

        # Spacing-aware resampling (optional)
        if self.target_spacing is not None:
            try:
                hdr = nii.header
                current_spacing = tuple(hdr.get_zooms()[:3])
                if all(s > 0 for s in current_spacing):
                    volume = self.resample_to_spacing(
                        volume, current_spacing, self.target_spacing,
                    )
            except Exception:
                pass  # fall through to plain resize

        volume = self.normalize_volume(volume)
        volume = self.resize_volume(volume)
        return volume
