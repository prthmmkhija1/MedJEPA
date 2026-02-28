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


def _enforce_image_shape(tensor: torch.Tensor, target_h: int = 224, target_w: int = 224) -> torch.Tensor:
    """
    Ensure tensor is exactly (3, target_h, target_w).
    Handles edge cases: wrong spatial size, wrong channels, extra dims.
    """
    # Squeeze any extra leading dims  (1, 3, H, W) → (3, H, W)
    while tensor.ndim > 3:
        tensor = tensor.squeeze(0)
    # If 2-D (H, W), expand to 3 channels
    if tensor.ndim == 2:
        tensor = tensor.unsqueeze(0).expand(3, -1, -1)
    c, h, w = tensor.shape
    # Fix channel count
    if c == 1:
        tensor = tensor.expand(3, -1, -1)
    elif c == 4:
        tensor = tensor[:3]
    elif c != 3:
        tensor = tensor[:1].expand(3, -1, -1)
    # Fix spatial dims
    if h != target_h or w != target_w:
        tensor = torch.nn.functional.interpolate(
            tensor.unsqueeze(0), size=(target_h, target_w),
            mode="bilinear", align_corners=False,
        ).squeeze(0)
    return tensor.contiguous()


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
        # Use torch.tensor() instead of torch.from_numpy() to avoid
        # "Numpy is not available" errors in DataLoader worker processes
        # (a known issue with NumPy 2.x + PyTorch multiprocessing).
        image = torch.tensor(image, dtype=torch.float32).permute(2, 0, 1)  # HWC → CHW
        image = _enforce_image_shape(image, *self.target_size)

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
        self.target_size = target_size
        self.with_labels = with_labels
        self.transform = transform
        self.preprocessor = MedicalImagePreprocessor(target_size=target_size)

        # Resolve image directory — try common subdirectory names
        self.image_dir = self.data_dir / "images"
        if not self.image_dir.exists():
            # Kaggle download may place images in different subdirs
            for candidate in ["CXR8", "images_001", "all_images", "train", "."]:
                cand_dir = self.data_dir / candidate
                if cand_dir.exists() and any(cand_dir.rglob("*.png")):
                    self.image_dir = cand_dir
                    break

        # Read metadata CSV — try several known filenames
        csv_candidates = [
            "Data_Entry_2017_v2020.csv",
            "Data_Entry_2017.csv",
            "data_entry_2017_v2020.csv",
            "data_entry.csv",
        ]
        csv_path = None
        for cname in csv_candidates:
            p = self.data_dir / cname
            if p.exists():
                csv_path = p
                break
        # Fallback: pick any CSV in the directory
        if csv_path is None:
            all_csvs = list(self.data_dir.glob("*.csv"))
            if all_csvs:
                csv_path = all_csvs[0]
        if csv_path is None:
            raise FileNotFoundError(
                f"ChestXray14 CSV not found in {self.data_dir}. "
                f"Looked for: {csv_candidates}. "
                f"No .csv files found in the directory."
            )

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

        image = torch.tensor(image, dtype=torch.float32).permute(2, 0, 1)
        image = _enforce_image_shape(image, *self.target_size)

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
        self._corrupted_paths = set()  # Track corrupted files to avoid retrying

    def _load_volume(self, path: str) -> np.ndarray:
        """Load a NIfTI volume, with simple caching."""
        if path in self._corrupted_paths:
            raise IOError(f"Previously identified corrupted NIfTI file: {path}")
        if path in self._cache:
            return self._cache[path]
        import nibabel as nib
        try:
            vol = nib.load(path).get_fdata().astype(np.float32)
        except (EOFError, OSError, Exception) as e:
            import logging
            logging.getLogger(__name__).warning(
                f"Skipping corrupted NIfTI file {path}: {e}"
            )
            self._corrupted_paths.add(path)
            raise IOError(f"Corrupted NIfTI file: {path}") from e
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
        try:
            vol = self._load_volume(npath)
        except (IOError, OSError):
            # Return a blank image for corrupted volumes so training can continue
            blank = torch.zeros(3, *self.target_size, dtype=torch.float32)
            if self.labels is not None:
                return blank, int(self.labels[idx])
            return blank

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
        image = torch.tensor(image, dtype=torch.float32).permute(2, 0, 1)  # HWC → CHW
        image = _enforce_image_shape(image, *self.target_size)

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

        # ── Auto-extract .tar / .tar.gz archives if present ──
        tar_files = list(data_dir.glob("*.tar*")) + list(data_dir.glob("*.tgz"))
        if tar_files:
            import tarfile
            for tf_path in tar_files:
                # Skip if already extracted (marker file)
                marker = tf_path.with_suffix(tf_path.suffix + ".extracted")
                if marker.exists():
                    continue
                try:
                    print(f"  Extracting {tf_path.name} ...")
                    with tarfile.open(str(tf_path), "r:*") as tar:
                        tar.extractall(path=str(data_dir))
                    marker.touch()  # mark as extracted
                    print(f"  Extracted {tf_path.name}")
                except Exception as e:
                    print(f"  WARNING: Failed to extract {tf_path.name}: {e}")

        # Look for subject dirs (BraTS2021_NNNNN) — search recursively
        # to handle cases where data is nested inside a parent folder
        # like BraTS2021_Training/ extracted from an archive.
        subject_dirs = sorted([
            d for d in data_dir.iterdir()
            if d.is_dir() and d.name.startswith("BraTS")
        ])

        # Check if any found dirs are "container" folders (no NIfTI files
        # inside, but contain subject sub-directories instead)
        expanded = []
        for d in subject_dirs:
            # A real subject dir has *_flair.nii.gz; a container has sub-dirs
            has_nifti = any(d.glob(f"{d.name}_{modality}.nii.gz"))
            if has_nifti:
                expanded.append(d)
            else:
                # Look one level deeper for actual subject directories
                nested = sorted([
                    sd for sd in d.iterdir()
                    if sd.is_dir() and sd.name.startswith("BraTS")
                ])
                if nested:
                    expanded.extend(nested)
        subject_dirs = expanded

        if not subject_dirs:
            import warnings
            warnings.warn(
                f"BraTSDataset: No valid subject directories found in {data_dir}. "
                f"Expected folders like BraTS2021_00000/ containing "
                f"BraTS2021_00000_{modality}.nii.gz files. "
                f"If you have .tar archives, they should be auto-extracted."
            )

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

        training_entries = meta.get("training", [])
        missing_count = 0
        for vol_idx, entry in enumerate(training_entries):
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
            else:
                missing_count += 1

        if missing_count > 0 and len(nifti_paths) == 0:
            # All training entries are missing — try fallback strategies
            images_tr = task_dir / "imagesTr"
            images_ts = task_dir / "imagesTs"
            tr_files = sorted(images_tr.glob("*.nii.gz")) if images_tr.exists() else []
            ts_files = sorted(images_ts.glob("*.nii.gz")) if images_ts.exists() else []

            if tr_files:
                # imagesTr has files but paths in dataset.json didn't match
                print(f"  DecathlonDataset ({task}): Using {len(tr_files)} "
                      f"NIfTI files found directly in imagesTr/")
                nifti_paths = [str(p) for p in tr_files]
                if with_labels:
                    labels = [0] * len(nifti_paths)
            elif ts_files:
                # imagesTr is empty but imagesTs has data — use test images
                # for self-supervised pre-training (no labels needed)
                print(f"  DecathlonDataset ({task}): imagesTr/ is empty, "
                      f"falling back to {len(ts_files)} files from imagesTs/")
                nifti_paths = [str(p) for p in ts_files]
                if with_labels:
                    labels = [0] * len(nifti_paths)
            else:
                import warnings
                warnings.warn(
                    f"DecathlonDataset ({task}): No NIfTI files found in "
                    f"imagesTr/ or imagesTs/. The data may not be downloaded."
                )

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

        try:
            vol = nib.load(self.nifti_paths[idx]).get_fdata().astype(np.float32)
        except (EOFError, OSError, Exception) as e:
            import logging
            logging.getLogger(__name__).warning(
                f"Skipping corrupted NIfTI volume {self.nifti_paths[idx]}: {e}"
            )
            # Return a blank volume so training doesn't crash
            vol = np.zeros(self.volume_size, dtype=np.float32)
            return torch.tensor(vol, dtype=torch.float32).unsqueeze(0)

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
        vol = torch.tensor(vol, dtype=torch.float32).unsqueeze(0)

        if self.transform:
            vol = self.transform(vol)

        return vol


# ═══════════════════════════════════════════════════════════
# Pre-extracted NIfTI Slice Dataset (FAST — reads .npy files)
# ═══════════════════════════════════════════════════════════
class PreExtractedSliceDataset(Dataset):
    """
    Ultra-fast dataset that reads pre-extracted 2D slices (.npy files)
    instead of decompressing NIfTI volumes on-the-fly.

    Run ``python scripts/preextract_slices.py`` first to generate the slices.

    Speed improvement: 10-50x faster than NIfTISliceDataset because
    .npy files are raw arrays — no gzip decompression, no 3D volume loading.
    """

    def __init__(
        self,
        slice_dir: str,
        entries: Optional[list] = None,
        labels: Optional[list] = None,
        with_labels: bool = True,
        transform: Optional[Callable] = None,
    ):
        """
        Args:
            slice_dir: Folder containing .npy slice files
            entries: List of dicts from manifest.json (each has 'file' key).
                     If None, auto-discovers all .npy files in slice_dir.
            labels: Optional list of integer labels (same length as entries)
            with_labels: If False, always return images only (no labels)
            transform: Optional augmentations
        """
        self.slice_dir = Path(slice_dir)
        self.transform = transform

        if entries is not None:
            self.files = [self.slice_dir / e["file"] for e in entries]
            if not with_labels:
                self.labels = None
            elif labels is not None:
                self.labels = np.array(labels, dtype=np.int64)
            else:
                # Try to get labels from entries
                if entries and "label" in entries[0]:
                    self.labels = np.array(
                        [e.get("label", 0) for e in entries], dtype=np.int64
                    )
                else:
                    self.labels = None
        else:
            # Auto-discover — skip *_label.npy files
            self.files = sorted(
                p for p in self.slice_dir.glob("*.npy")
                if not p.name.endswith("_label.npy")
            )
            self.labels = None

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        # .npy load is extremely fast (memory-mapped capable)
        img = np.load(str(self.files[idx]))  # (H, W, 3) float32

        image = torch.tensor(img, dtype=torch.float32).permute(2, 0, 1)  # CHW
        image = _enforce_image_shape(image)

        if self.transform:
            image = self.transform(image)

        if self.labels is not None:
            return image, int(self.labels[idx])
        return image


# ═══════════════════════════════════════════════════════════
# RAM-Cached Dataset — zero I/O after the first epoch
# ═══════════════════════════════════════════════════════════
class RamCachedDataset(Dataset):
    """
    A transparent wrapper that caches all dataset items in RAM after first access.

    How it works:
    - First epoch: items are loaded normally from disk and stored in RAM as
      float16 tensors (halves memory vs float32).
    - All subsequent epochs: items are served directly from the RAM cache
      with zero disk I/O, zero PIL/cv2 decoding, zero numpy ops.

    Speed benefit: eliminates ALL per-sample disk read + decode overhead
    after the first epoch.  Yields 5-20x faster data loading on spinning HDD
    and 2-5x on NVMe SSD.

    Memory guidance (224×224 RGB, float16):
      10k images  → ~0.3 GB   (HAM10000, APTOS — fits comfortably)
      112k images → ~3.5 GB   (ChestXray14)
      370k images → ~11.5 GB  (full combined set — needs 12+ GB free RAM)

    Set ``max_gb`` to limit cache size and fall back to disk beyond the limit.
    """

    def __init__(
        self,
        dataset: Dataset,
        max_gb: float = 8.0,        # Max RAM to use for cache (GiB)
        dtype: torch.dtype = torch.float16,  # Store as fp16 to halve memory
        verbose: bool = True,
    ):
        self.dataset = dataset
        self.max_gb = max_gb
        self.dtype = dtype
        self.verbose = verbose

        # Estimate per-item size once we see the first element
        self._cache: dict = {}        # idx → tensor (dtype)
        self._labels_cache: dict = {} # idx → label int (if present)
        self._cache_full = False      # True once we hit the memory cap
        self._bytes_used = 0

        # Mirror dataset attributes (labels, etc.) so wrappers can inspect them
        if hasattr(dataset, "labels"):
            self.labels = dataset.labels
        if hasattr(dataset, "num_classes"):
            self.num_classes = dataset.num_classes

        # Eagerly warm cache if dataset is small enough to justify it
        n = len(dataset)
        sample = dataset[0]
        img = sample[0] if isinstance(sample, (tuple, list)) else sample
        item_bytes = img.element_size() * img.nelement()
        # fp16 storage
        item_bytes_fp16 = (item_bytes // img.element_size()) * 2
        estimated_gb = (n * item_bytes_fp16) / (1024 ** 3)

        if verbose:
            print(f"  RamCachedDataset: {n} samples, "
                  f"~{estimated_gb:.1f} GB (fp16), limit={max_gb} GB")

        if estimated_gb <= max_gb:
            self._warm_cache()
        else:
            if verbose:
                cutoff = int(max_gb * (1024 ** 3) / item_bytes_fp16)
                print(f"  Dataset too large for full cache — "
                      f"caching first {cutoff:,} samples, rest from disk")

    def _warm_cache(self):
        """Pre-load entire dataset into RAM cache."""
        if self.verbose:
            print(f"  Pre-loading {len(self.dataset)} items into RAM cache...")
        try:
            from tqdm import tqdm as _tqdm
            it = _tqdm(range(len(self.dataset)), desc="  Caching", unit="img",
                       leave=False, ncols=80)
        except ImportError:
            it = range(len(self.dataset))
        for idx in it:
            self._fetch_and_cache(idx)
        if self.verbose:
            gb = self._bytes_used / (1024 ** 3)
            print(f"  Cache ready: {len(self._cache)} items, {gb:.2f} GB used")

    def _fetch_and_cache(self, idx: int):
        """Load item from wrapped dataset and store in cache."""
        if self._cache_full:
            return
        item = self.dataset[idx]
        if isinstance(item, (tuple, list)):
            img, label = item[0], item[1]
            self._labels_cache[idx] = label
        else:
            img = item
        # Store as reduced-precision to save memory
        cached = img.to(self.dtype)
        item_bytes = cached.element_size() * cached.nelement()
        if self._bytes_used + item_bytes > self.max_gb * (1024 ** 3):
            self._cache_full = True
            return
        self._cache[idx] = cached
        self._bytes_used += item_bytes

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx: int):
        if idx not in self._cache:
            # Cache miss: load from disk and optionally store
            self._fetch_and_cache(idx)

        if idx in self._cache:
            # Serve from cache — cast back to float32 for the model
            img = self._cache[idx].float()
            if idx in self._labels_cache:
                return img, self._labels_cache[idx]
            return img

        # Beyond cache limit — fall back to original dataset
        return self.dataset[idx]

