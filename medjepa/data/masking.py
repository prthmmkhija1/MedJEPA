"""
Masking strategies for JEPA self-supervised learning.

The model sees some patches (context) and must predict the hidden patches (target).
This file implements different ways to choose which patches to hide.
"""

import torch
import numpy as np
from typing import Tuple, List, Optional


class PatchMasker2D:
    """
    Masking for 2D medical images.

    An image of size 224x224, divided into 16x16 patches, gives a 14x14 grid
    of patches (224/16 = 14). That's 196 total patches.

    We might hide 75% of them (147 patches) and show 25% (49 patches).
    """

    def __init__(
        self,
        image_size: int = 224,
        patch_size: int = 16,
        mask_ratio: float = 0.75,
        # How much of the image to hide. 0.75 = hide 75%.
        num_target_blocks: int = 4,
        # How many separate rectangular blocks to hide
        mask_cache_size: int = 256,
        # Pre-generate this many masks to avoid CPU overhead each step
    ):
        self.image_size = image_size
        self.patch_size = patch_size
        self.grid_size = image_size // patch_size  # e.g., 224/16 = 14
        self.num_patches = self.grid_size ** 2      # e.g., 14*14 = 196
        self.mask_ratio = mask_ratio
        self.num_target_blocks = num_target_blocks

        # Pre-generate a cache of masks to eliminate CPU→GPU overhead
        self._mask_cache = []
        self._cache_idx = 0
        self._cache_device = None
        for _ in range(mask_cache_size):
            c, t = self._generate_block_mask_raw()
            self._mask_cache.append((c, t))

    def generate_mask(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate context (visible) and target (hidden) patch indices.

        Returns:
            context_indices: Which patches the model can see
            target_indices: Which patches the model must predict
        """
        num_masked = int(self.num_patches * self.mask_ratio)
        num_visible = self.num_patches - num_masked

        # Randomly shuffle all patch indices
        all_indices = np.random.permutation(self.num_patches)

        # Split into visible (context) and hidden (target)
        context_indices = torch.tensor(sorted(all_indices[:num_visible]))
        target_indices = torch.tensor(sorted(all_indices[num_visible:]))

        return context_indices, target_indices

    def generate_block_mask(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate block-style masking (like I-JEPA).
        Uses a pre-generated cache for speed — no numpy/CPU work per step.
        """
        c, t = self._mask_cache[self._cache_idx % len(self._mask_cache)]
        self._cache_idx += 1
        # Refresh cache periodically for diversity
        if self._cache_idx >= len(self._mask_cache):
            self._cache_idx = 0
            # Reshuffle order
            import random
            random.shuffle(self._mask_cache)
        return c, t

    def _generate_block_mask_raw(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate one block-style mask (CPU, used for cache building).
        Instead of random patches, hides rectangular BLOCKS.

        This is more natural — in a chest X-ray, hiding a rectangular region
        forces the model to understand spatial relationships (e.g., if you
        hide the left lung, the model must understand anatomy to predict it).
        """
        mask = np.zeros((self.grid_size, self.grid_size), dtype=bool)
        total_to_mask = int(self.num_patches * self.mask_ratio)
        masked_so_far = 0

        for _ in range(self.num_target_blocks):
            remaining = total_to_mask - masked_so_far
            if remaining <= 0:
                break

            # Random block size
            block_h = np.random.randint(
                self.grid_size // 4, self.grid_size // 2 + 1
            )
            block_w = np.random.randint(
                self.grid_size // 4, self.grid_size // 2 + 1
            )

            # Random position
            top = np.random.randint(0, self.grid_size - block_h + 1)
            left = np.random.randint(0, self.grid_size - block_w + 1)

            # Apply mask
            mask[top:top + block_h, left:left + block_w] = True
            masked_so_far = mask.sum()

        # Convert 2D mask to 1D patch indices
        mask_flat = mask.flatten()
        target_indices = torch.tensor(np.where(mask_flat)[0].tolist(), dtype=torch.long)
        context_indices = torch.tensor(np.where(~mask_flat)[0].tolist(), dtype=torch.long)

        return context_indices, target_indices

    def visualize_mask(self, context_indices, target_indices):
        """
        Create a visual representation of the mask for debugging.
        Returns a grid where 0=visible, 1=hidden.
        """
        grid = np.zeros(self.num_patches)
        grid[target_indices.numpy()] = 1
        return grid.reshape(self.grid_size, self.grid_size)


class AnatomyAwareMasker:
    """
    Anatomy-aware masking for medical images.

    Instead of masking purely random regions, this masker biases the target
    blocks toward *anatomically interesting* areas (high-intensity variance,
    edges, potential lesion regions).  The context (visible) patches are
    chosen from less informative areas so the model must *reason* about
    anatomy to predict the masked targets.

    Strategy:
      1. Compute a per-patch "saliency" score from the image itself
         (local intensity variance, edge magnitude, or foreground fraction).
      2. Sample target blocks with probability proportional to saliency.
      3. Remaining patches become the context.
    """

    def __init__(
        self,
        image_size: int = 224,
        patch_size: int = 16,
        mask_ratio: float = 0.75,
        saliency_method: str = "variance",
        # "variance" — local intensity variance per patch
        # "edge"     — Sobel edge magnitude per patch
        # "foreground" — fraction of non-zero pixels per patch
        temperature: float = 2.0,
        # Controls how strongly saliency biases the sampling.
        # Higher = more bias toward salient patches.
    ):
        self.image_size = image_size
        self.patch_size = patch_size
        self.grid_size = image_size // patch_size
        self.num_patches = self.grid_size ** 2
        self.mask_ratio = mask_ratio
        self.saliency_method = saliency_method
        self.temperature = temperature

    def _compute_patch_saliency(
        self,
        image: np.ndarray,
    ) -> np.ndarray:
        """
        Compute a saliency score for each patch in the image.

        Args:
            image: numpy array of shape (H, W) or (H, W, C) with values in [0, 1].

        Returns:
            saliency: 1D array of shape (num_patches,) with per-patch scores.
        """
        # Convert to grayscale if needed
        if image.ndim == 3:
            gray = image.mean(axis=-1)
        else:
            gray = image

        # Resize to exact image_size for clean grid alignment
        if gray.shape[0] != self.image_size or gray.shape[1] != self.image_size:
            from PIL import Image as PILImage
            pil = PILImage.fromarray((np.clip(gray, 0, 1) * 255).astype(np.uint8))
            pil = pil.resize((self.image_size, self.image_size), PILImage.LANCZOS)
            gray = np.array(pil, dtype=np.float32) / 255.0

        saliency = np.zeros(self.num_patches, dtype=np.float32)
        ps = self.patch_size

        for i in range(self.grid_size):
            for j in range(self.grid_size):
                patch = gray[i * ps:(i + 1) * ps, j * ps:(j + 1) * ps]
                idx = i * self.grid_size + j

                if self.saliency_method == "variance":
                    saliency[idx] = patch.var()
                elif self.saliency_method == "edge":
                    # Simple Sobel approximation using numpy differences
                    gx = np.diff(patch, axis=1)
                    gy = np.diff(patch, axis=0)
                    saliency[idx] = float(np.sqrt(gx ** 2).mean() + np.sqrt(gy ** 2).mean())
                elif self.saliency_method == "foreground":
                    # Fraction of pixels above a small threshold (not background)
                    saliency[idx] = (patch > 0.05).mean()
                else:
                    saliency[idx] = 1.0  # uniform fallback

        return saliency

    def generate_mask(
        self,
        image: np.ndarray,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate anatomy-aware context and target masks.

        Args:
            image: numpy array of shape (H, W) or (H, W, C).

        Returns:
            context_indices, target_indices
        """
        saliency = self._compute_patch_saliency(image)

        # Apply temperature and convert to probabilities
        # Higher temperature → more bias toward salient patches
        logits = saliency * self.temperature
        # Softmax-style normalization
        logits = logits - logits.max()  # numerical stability
        probs = np.exp(logits)
        probs = probs / probs.sum()

        num_target = int(self.num_patches * self.mask_ratio)

        # Sample target patches (without replacement) biased by saliency
        target_indices = np.random.choice(
            self.num_patches,
            size=num_target,
            replace=False,
            p=probs,
        )
        target_indices = np.sort(target_indices)

        all_set = set(range(self.num_patches))
        context_indices = np.array(sorted(all_set - set(target_indices)))

        return torch.tensor(context_indices), torch.tensor(target_indices)


class PatchMasker3D:
    """
    Masking for 3D medical volumes (CT, MRI).

    A 3D volume has width, height, AND depth (number of slices).
    We divide it into 3D cubes (like small dice) and hide some of them.
    """

    def __init__(
        self,
        volume_size: Tuple[int, int, int] = (128, 128, 64),
        patch_size: Tuple[int, int, int] = (16, 16, 8),
        mask_ratio: float = 0.75,
        num_target_blocks: int = 4,
        # How many 3D rectangular blocks to mask (for block masking).
    ):
        self.volume_size = volume_size
        self.patch_size = patch_size
        self.grid_size = tuple(
            v // p for v, p in zip(volume_size, patch_size)
        )
        self.num_patches = (
            self.grid_size[0] * self.grid_size[1] * self.grid_size[2]
        )
        self.mask_ratio = mask_ratio
        self.num_target_blocks = num_target_blocks

    def generate_mask(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generate random 3D mask."""
        num_masked = int(self.num_patches * self.mask_ratio)
        all_indices = np.random.permutation(self.num_patches)

        context_indices = torch.tensor(sorted(all_indices[num_masked:]))
        target_indices = torch.tensor(sorted(all_indices[:num_masked]))

        return context_indices, target_indices

    def generate_block_mask(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate 3D block-style masking (analogous to 2D block masking).

        Hides contiguous 3D rectangular sub-volumes. This forces the model
        to understand the 3D spatial structure of anatomy (e.g., if a
        cuboid covering part of a tumour is masked, the model must infer
        its content from surrounding tissue).
        """
        gH, gW, gD = self.grid_size
        mask = np.zeros((gH, gW, gD), dtype=bool)
        total_to_mask = int(self.num_patches * self.mask_ratio)

        for _ in range(self.num_target_blocks):
            if mask.sum() >= total_to_mask:
                break

            # Random block dimensions (between 1/4 and 1/2 of each grid dim)
            bH = np.random.randint(max(gH // 4, 1), max(gH // 2 + 1, 2))
            bW = np.random.randint(max(gW // 4, 1), max(gW // 2 + 1, 2))
            bD = np.random.randint(max(gD // 4, 1), max(gD // 2 + 1, 2))

            # Random position (ensuring it fits)
            top = np.random.randint(0, max(gH - bH + 1, 1))
            left = np.random.randint(0, max(gW - bW + 1, 1))
            front = np.random.randint(0, max(gD - bD + 1, 1))

            mask[top:top + bH, left:left + bW, front:front + bD] = True

        # Convert 3D mask to 1D patch indices
        mask_flat = mask.flatten()
        target_indices = torch.tensor(np.where(mask_flat)[0].tolist(), dtype=torch.long)
        context_indices = torch.tensor(np.where(~mask_flat)[0].tolist(), dtype=torch.long)

        return context_indices, target_indices


class TemporalMasker:
    """
    Masking for medical video/sequences.
    Hides entire time frames or temporal chunks.

    Used for: cardiac MRI sequences, surgical videos, ultrasound clips.
    """

    def __init__(
        self,
        num_frames: int = 16,
        patch_size_spatial: int = 16,
        image_size: int = 224,
        mask_ratio_temporal: float = 0.5,  # Hide 50% of frames
        mask_ratio_spatial: float = 0.75,  # Also hide 75% of patches in remaining frames
    ):
        self.num_frames = num_frames
        self.grid_size = image_size // patch_size_spatial
        self.num_spatial_patches = self.grid_size ** 2
        self.mask_ratio_temporal = mask_ratio_temporal
        self.mask_ratio_spatial = mask_ratio_spatial

    def generate_mask(self) -> dict:
        """Generate spatiotemporal mask."""
        # Temporal: which frames to hide
        num_hidden_frames = int(self.num_frames * self.mask_ratio_temporal)
        frame_order = np.random.permutation(self.num_frames)
        hidden_frames = sorted(frame_order[:num_hidden_frames])
        visible_frames = sorted(frame_order[num_hidden_frames:])

        # Spatial: in visible frames, which patches to show
        num_visible_patches = int(
            self.num_spatial_patches * (1 - self.mask_ratio_spatial)
        )
        spatial_indices = np.random.permutation(self.num_spatial_patches)
        visible_patches = sorted(spatial_indices[:num_visible_patches])

        return {
            "visible_frames": torch.tensor(visible_frames),
            "hidden_frames": torch.tensor(hidden_frames),
            "visible_patches": torch.tensor(visible_patches),
        }
