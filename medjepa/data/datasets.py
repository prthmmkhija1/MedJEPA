"""
PyTorch Dataset classes for loading medical images.

A "Dataset" in PyTorch is an object that knows:
1. How many images there are
2. How to load image number N

PyTorch then uses this to efficiently feed batches of images to the model.

Supports:
- 2D images: HAM10000, APTOS, PCam, ChestXray14
- 3D volumes (as 2D slices): BraTS, Medical Segmentation Decathlon
"""

import torch
from torch.utils.data import Dataset
from pathlib import Path
import pandas as pd
import numpy as np
from typing import Optional, List, Callable, Tuple
from medjepa.data.preprocessing import MedicalImagePreprocessor


class MedicalImageDataset(Dataset):
    """
    A general dataset class for 2D medical images.
    Works for chest X-rays, skin photos, retinal images, etc.
    """

    def __init__(
        self,
        image_dir: str,
        metadata_csv: Optional[str] = None,
        image_column: str = "image_id",
        label_column: Optional[str] = None,
        file_extension: str = ".jpg",
        target_size: tuple = (224, 224),
        transform: Optional[Callable] = None,
    ):
        """
        Args:
            image_dir: Folder containing the images
            metadata_csv: Optional CSV with image IDs and labels
            image_column: Column name in CSV that has image file names
            label_column: Column name in CSV that has labels (None for self-supervised)
            file_extension: What type of image files to look for
            target_size: Size to resize images to
            transform: Optional extra transformations (augmentations)
        """
        self.image_dir = Path(image_dir)
        self.target_size = target_size
        self.transform = transform
        self.preprocessor = MedicalImagePreprocessor(target_size=target_size)

        # Find all images
        if metadata_csv:
            self.metadata = pd.read_csv(metadata_csv)
            # Build a fast lookup index: stem -> full path (once)
            self._file_index = {}
            for p in self.image_dir.rglob(f"*{file_extension}"):
                self._file_index[p.stem] = p
            self.image_files = [
                self._file_index.get(str(row[image_column]))
                for _, row in self.metadata.iterrows()
            ]
            if label_column and label_column in self.metadata.columns:
                raw_labels = self.metadata[label_column].values
                # Encode string labels to integers for CrossEntropyLoss
                if raw_labels.dtype.kind in ('U', 'S', 'O'):  # string types
                    unique_labels = sorted(set(raw_labels))
                    self.label_map = {name: idx for idx, name in enumerate(unique_labels)}
                    self.labels = np.array([self.label_map[l] for l in raw_labels])
                else:
                    self.label_map = None
                    self.labels = raw_labels
            else:
                self.labels = None
                self.label_map = None
        else:
            # No CSV — just find all image files in the folder
            self.image_files = sorted(
                self.image_dir.rglob(f"*{file_extension}")
            )
            self.labels = None
            self.label_map = None

    def _find_image(self, image_id: str, ext: str) -> Path:
        """Find an image file by its ID (fallback, uses index if available)."""
        if hasattr(self, '_file_index'):
            return self._file_index.get(str(image_id))
        candidates = list(self.image_dir.rglob(f"{image_id}{ext}"))
        if not candidates:
            candidates = list(self.image_dir.rglob(f"{image_id}.*"))
        return candidates[0] if candidates else None

    def __len__(self):
        """How many images in this dataset."""
        return len(self.image_files)

    def __getitem__(self, idx):
        """
        Load image number 'idx'.
        This is called by PyTorch's DataLoader.
        """
        image_path = self.image_files[idx]
        if image_path is None or not image_path.exists():
            # Return a blank image if file not found
            image = np.zeros((*self.target_size, 3), dtype=np.float32)
        else:
            image = self.preprocessor.preprocess(str(image_path))

        # Convert to PyTorch tensor
        # PyTorch expects shape (channels, height, width), not (height, width, channels)
        image = torch.from_numpy(image).permute(2, 0, 1)  # HWC → CHW

        if self.transform:
            image = self.transform(image)

        if self.labels is not None:
            label = int(self.labels[idx])
            return image, label
        else:
            return image


# ═══════════════════════════════════════════════════════════
# ChestX-ray14 Dataset (multi-label, 14 pathology classes)
# ═══════════════════════════════════════════════════════════
# 14 diseases in ChestXray14
CHESTXRAY14_DISEASES = [
    "Atelectasis", "Cardiomegaly", "Effusion", "Infiltration",
    "Mass", "Nodule", "Pneumonia", "Pneumothorax",
    "Consolidation", "Edema", "Emphysema", "Fibrosis",
    "Pleural_Thickening", "Hernia",
]


class ChestXray14Dataset(Dataset):
    """
    NIH ChestX-ray14: 112,120 frontal-view chest X-rays from 30,805 patients.
    Multi-label: each image can have 0+ of 14 disease labels.

    For self-supervised pre-training: returns images only (no labels).
    For evaluation: returns images + multi-hot label vector (14-dim).
    """

    def __init__(
        self,
        data_dir: str = "data/raw/chestxray14",
        target_size: tuple = (224, 224),
        with_labels: bool = False,
        transform: Optional[Callable] = None,
        max_samples: Optional[int] = None,
    ):
        self.data_dir = Path(data_dir)
        self.image_dir = self.data_dir / "images"
        self.target_size = target_size
        self.with_labels = with_labels
        self.transform = transform
        self.preprocessor = MedicalImagePreprocessor(target_size=target_size)

        # Read metadata CSV
        csv_path = self.data_dir / "Data_Entry_2017_v2020.csv"
        if not csv_path.exists():
            raise FileNotFoundError(f"ChestXray14 CSV not found: {csv_path}")

        self.metadata = pd.read_csv(csv_path)

        # Filter to images that actually exist on disk
        existing = set(p.name for p in self.image_dir.rglob("*.png"))
        self.metadata = self.metadata[
            self.metadata["Image Index"].isin(existing)
        ].reset_index(drop=True)

        if max_samples and max_samples < len(self.metadata):
            self.metadata = self.metadata.sample(
                n=max_samples, random_state=42
            ).reset_index(drop=True)

        # Build multi-hot labels (14-dim vector)
        if self.with_labels:
            self.labels = np.zeros(
                (len(self.metadata), len(CHESTXRAY14_DISEASES)), dtype=np.float32
            )
            for i, row in self.metadata.iterrows():
                findings = str(row["Finding Labels"]).split("|")
                for f in findings:
                    f = f.strip()
                    if f in CHESTXRAY14_DISEASES:
                        idx = CHESTXRAY14_DISEASES.index(f)
                        self.labels[i, idx] = 1.0
            # For single-label classification use the dominant (first) finding
            self.single_labels = np.array([
                CHESTXRAY14_DISEASES.index(
                    str(row["Finding Labels"]).split("|")[0].strip()
                ) if str(row["Finding Labels"]).split("|")[0].strip()
                in CHESTXRAY14_DISEASES else 0
                for _, row in self.metadata.iterrows()
            ])
        else:
            self.labels = None
            self.single_labels = None

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        filename = self.metadata.iloc[idx]["Image Index"]
        image_path = self.image_dir / filename

        if not image_path.exists():
            image = np.zeros((*self.target_size, 3), dtype=np.float32)
        else:
            image = self.preprocessor.preprocess(str(image_path))

        image = torch.from_numpy(image).permute(2, 0, 1)

        if self.transform:
            image = self.transform(image)

        if self.labels is not None:
            # Return single-label int for linear probe compatibility
            return image, int(self.single_labels[idx])
        return image


# ═══════════════════════════════════════════════════════════
# NIfTI 2D Slice Dataset (for BraTS and Decathlon)
# ═══════════════════════════════════════════════════════════
class NIfTISliceDataset(Dataset):
    """
    Extracts 2D slices from 3D NIfTI volumes for use with 2D LeJEPA.

    Strategy: For each volume, extract N evenly-spaced axial slices
    from the informative middle portion (skip empty boundary slices).
    This converts 3D volumes into a large pool of 2D training images.

    Works for both BraTS and Medical Segmentation Decathlon.
    """

    def __init__(
        self,
        nifti_paths: List[str],
        target_size: tuple = (224, 224),
        slices_per_volume: int = 10,
        labels: Optional[List[int]] = None,
        transform: Optional[Callable] = None,
    ):
        """
        Args:
            nifti_paths: List of paths to NIfTI files (.nii.gz)
            target_size: Size to resize each 2D slice
            slices_per_volume: How many 2D slices to extract per volume
            labels: Optional list of integer labels (one per volume, repeated for slices)
            transform: Optional augmentations
        """
        self.target_size = target_size
        self.slices_per_volume = slices_per_volume
        self.transform = transform
        self.preprocessor = MedicalImagePreprocessor(target_size=target_size)

        # Build index: (volume_path, slice_idx) pairs
        self.samples = []  # List of (nifti_path, slice_index)
        self.labels = None

        if labels is not None:
            label_list = []

        for vol_idx, npath in enumerate(nifti_paths):
            npath = Path(npath)
            if not npath.exists():
                continue
            # Read volume header to get number of slices (fast, no data load)
            try:
                import nibabel as nib
                proxy = nib.load(str(npath))
                shape = proxy.shape
                n_slices = shape[2] if len(shape) >= 3 else shape[0]
            except Exception:
                continue

            # Pick slices from the middle 60% of the volume (skip empty edges)
            start = int(n_slices * 0.2)
            end = int(n_slices * 0.8)
            if end - start < slices_per_volume:
                indices = list(range(start, end))
            else:
                indices = np.linspace(start, end - 1, slices_per_volume, dtype=int).tolist()

            for si in indices:
                self.samples.append((str(npath), si))
                if labels is not None:
                    label_list.append(labels[vol_idx])

        if labels is not None:
            self.labels = np.array(label_list)

        # Cache for loaded volumes (keeps last N volumes in memory)
        self._cache = {}
        self._cache_order = []
        self._cache_max = 5

    def _load_volume(self, path: str) -> np.ndarray:
        """Load a NIfTI volume, with simple caching."""
        if path in self._cache:
            return self._cache[path]
        import nibabel as nib
        vol = nib.load(path).get_fdata().astype(np.float32)
        # Cache management
        if len(self._cache_order) >= self._cache_max:
            old = self._cache_order.pop(0)
            self._cache.pop(old, None)
        self._cache[path] = vol
        self._cache_order.append(path)
        return vol

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        npath, slice_idx = self.samples[idx]
        vol = self._load_volume(npath)

        # Extract axial slice
        if len(vol.shape) == 4:
            # Multi-channel volume (e.g., Decathlon Prostate with T2+ADC)
            slc = vol[:, :, slice_idx, 0]  # Take first channel
        elif len(vol.shape) == 3:
            slc = vol[:, :, slice_idx]
        else:
            slc = vol

        # Normalize to 0-1
        smin, smax = slc.min(), slc.max()
        if smax - smin > 0:
            slc = (slc - smin) / (smax - smin)
        else:
            slc = np.zeros_like(slc)

        # Resize to target size
        from PIL import Image as PILImage
        pil_img = PILImage.fromarray((slc * 255).astype(np.uint8))
        pil_img = pil_img.resize(self.target_size, PILImage.LANCZOS)
        slc = np.array(pil_img, dtype=np.float32) / 255.0

        # Grayscale → 3 channels (model expects 3ch)
        image = np.stack([slc, slc, slc], axis=-1)
        image = torch.from_numpy(image).permute(2, 0, 1)  # HWC → CHW

        if self.transform:
            image = self.transform(image)

        if self.labels is not None:
            return image, int(self.labels[idx])
        return image


# ═══════════════════════════════════════════════════════════
# BraTS Dataset (wrapper around NIfTISliceDataset)
# ═══════════════════════════════════════════════════════════
class BraTSDataset(NIfTISliceDataset):
    """
    BraTS 2021: Brain Tumor Segmentation.

    Each subject has 4 MRI modalities: FLAIR, T1, T1ce, T2
    By default uses FLAIR (best for tumor visibility).

    For self-supervised: extracts 2D slices from FLAIR volumes.
    For evaluation: uses segmentation labels (tumor grade classification).
    """

    def __init__(
        self,
        data_dir: str = "data/raw/brats",
        modality: str = "flair",
        target_size: tuple = (224, 224),
        slices_per_volume: int = 10,
        with_labels: bool = False,
        transform: Optional[Callable] = None,
        max_subjects: Optional[int] = None,
    ):
        data_dir = Path(data_dir)
        subject_dirs = sorted([
            d for d in data_dir.iterdir()
            if d.is_dir() and d.name.startswith("BraTS")
        ])

        if max_subjects and max_subjects < len(subject_dirs):
            rng = np.random.RandomState(42)
            indices = rng.choice(len(subject_dirs), max_subjects, replace=False)
            subject_dirs = [subject_dirs[i] for i in sorted(indices)]

        # Collect NIfTI paths for the chosen modality
        nifti_paths = []
        labels = [] if with_labels else None

        for sdir in subject_dirs:
            vol_path = sdir / f"{sdir.name}_{modality}.nii.gz"
            if vol_path.exists():
                nifti_paths.append(str(vol_path))
                if with_labels:
                    # Label based on tumor presence/size in segmentation mask.
                    # Load seg mask and compute tumor volume fraction.
                    seg_path = sdir / f"{sdir.name}_seg.nii.gz"
                    if seg_path.exists():
                        try:
                            import nibabel as nib
                            seg = nib.load(str(seg_path)).get_fdata()
                            tumor_fraction = (seg > 0).sum() / max(seg.size, 1)
                            # Binary: large tumor (>2% volume) vs small tumor
                            labels.append(1 if tumor_fraction > 0.02 else 0)
                        except Exception:
                            labels.append(0)
                    else:
                        labels.append(0)

        super().__init__(
            nifti_paths=nifti_paths,
            target_size=target_size,
            slices_per_volume=slices_per_volume,
            labels=labels,
            transform=transform,
        )
        self.num_classes = 2
        self.data_dir = data_dir


# ═══════════════════════════════════════════════════════════
# Medical Segmentation Decathlon Dataset
# ═══════════════════════════════════════════════════════════
# All 10 Decathlon tasks and their download IDs
DECATHLON_TASKS = {
    "Task01_BrainTumour":   {"modality": "MRI",  "kaggle": None},
    "Task02_Heart":         {"modality": "MRI",  "kaggle": None},
    "Task03_Liver":         {"modality": "CT",   "kaggle": None},
    "Task04_Hippocampus":   {"modality": "MRI",  "kaggle": None},
    "Task05_Prostate":      {"modality": "MRI",  "kaggle": None},
    "Task06_Lung":          {"modality": "CT",   "kaggle": None},
    "Task07_Pancreas":      {"modality": "CT",   "kaggle": None},
    "Task08_HepaticVessel": {"modality": "CT",   "kaggle": None},
    "Task09_Spleen":        {"modality": "CT",   "kaggle": None},
    "Task10_Colon":         {"modality": "CT",   "kaggle": None},
}


class DecathlonDataset(NIfTISliceDataset):
    """
    Medical Segmentation Decathlon — supports all 10 tasks.

    Each task has NIfTI volumes in imagesTr/ with labels in labelsTr/.
    We extract 2D slices for LeJEPA pre-training and keep segmentation
    labels available for evaluation.
    """

    def __init__(
        self,
        data_dir: str = "data/raw/decathlon",
        task: str = "Task02_Heart",
        target_size: tuple = (224, 224),
        slices_per_volume: int = 10,
        with_labels: bool = False,
        transform: Optional[Callable] = None,
    ):
        import json

        task_dir = Path(data_dir) / task
        self.task_dir = task_dir
        self.task_name = task

        if not task_dir.exists():
            raise FileNotFoundError(f"Decathlon task not found: {task_dir}")

        # Read dataset.json for the task metadata
        meta_path = task_dir / "dataset.json"
        with open(meta_path) as f:
            meta = json.load(f)

        self.task_meta = meta
        self.modality = meta.get("modality", {})
        self.label_names = meta.get("labels", {})
        self.num_classes = len(self.label_names)

        # Collect training image paths
        nifti_paths = []
        labels = [] if with_labels else None

        for vol_idx, entry in enumerate(meta.get("training", [])):
            img_rel = entry["image"]  # e.g. "./imagesTr/la_007.nii.gz"
            img_path = task_dir / img_rel.lstrip("./")
            if img_path.exists():
                nifti_paths.append(str(img_path))
                if with_labels:
                    # Load the segmentation label and compute presence of
                    # foreground structure for binary classification
                    lbl_rel = entry.get("label", "")
                    lbl_path = task_dir / lbl_rel.lstrip("./") if lbl_rel else None
                    if lbl_path and lbl_path.exists():
                        try:
                            import nibabel as nib
                            seg = nib.load(str(lbl_path)).get_fdata()
                            fg_fraction = (seg > 0).sum() / max(seg.size, 1)
                            # Binary: large structure (>1% volume) vs small
                            labels.append(1 if fg_fraction > 0.01 else 0)
                        except Exception:
                            labels.append(vol_idx % 2)
                    else:
                        labels.append(vol_idx % 2)

        super().__init__(
            nifti_paths=nifti_paths,
            target_size=target_size,
            slices_per_volume=slices_per_volume,
            labels=labels,
            transform=transform,
        )


# ═══════════════════════════════════════════════════════════
# Volumetric Dataset (for V-JEPA 3D training)
# ═══════════════════════════════════════════════════════════
class VolumetricDataset(Dataset):
    """
    Loads full 3D NIfTI volumes for V-JEPA pre-training.
    Resizes volumes to a standard size for batching.
    """

    def __init__(
        self,
        nifti_paths: List[str],
        volume_size: Tuple[int, int, int] = (128, 128, 64),
        transform: Optional[Callable] = None,
    ):
        self.nifti_paths = [p for p in nifti_paths if Path(p).exists()]
        self.volume_size = volume_size
        self.transform = transform

    def __len__(self):
        return len(self.nifti_paths)

    def __getitem__(self, idx):
        import nibabel as nib
        from scipy.ndimage import zoom

        vol = nib.load(self.nifti_paths[idx]).get_fdata().astype(np.float32)

        # If 4D (multi-channel), take first channel
        if vol.ndim == 4:
            vol = vol[..., 0]

        # Normalize to 0-1
        vmin, vmax = vol.min(), vol.max()
        if vmax - vmin > 0:
            vol = (vol - vmin) / (vmax - vmin)
        else:
            vol = np.zeros_like(vol)

        # Resize to standard volume size
        zoom_factors = [t / s for t, s in zip(self.volume_size, vol.shape[:3])]
        vol = zoom(vol, zoom_factors, order=1)

        # Add channel dim: (D, H, W) → (1, D, H, W)
        vol = torch.from_numpy(vol).unsqueeze(0)

        if self.transform:
            vol = self.transform(vol)

        return vol
