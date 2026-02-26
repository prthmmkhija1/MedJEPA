"""
PyTorch Dataset classes for loading medical images.

A "Dataset" in PyTorch is an object that knows:
1. How many images there are
2. How to load image number N

PyTorch then uses this to efficiently feed batches of images to the model.
"""

import torch
from torch.utils.data import Dataset
from pathlib import Path
import pandas as pd
import numpy as np
from typing import Optional, List, Callable
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
