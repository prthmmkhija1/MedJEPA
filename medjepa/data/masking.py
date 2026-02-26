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
    ):
        self.image_size = image_size
        self.patch_size = patch_size
        self.grid_size = image_size // patch_size  # e.g., 224/16 = 14
        self.num_patches = self.grid_size ** 2      # e.g., 14*14 = 196
        self.mask_ratio = mask_ratio
        self.num_target_blocks = num_target_blocks

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
        target_indices = torch.tensor(np.where(mask_flat)[0])
        context_indices = torch.tensor(np.where(~mask_flat)[0])

        return context_indices, target_indices

    def visualize_mask(self, context_indices, target_indices):
        """
        Create a visual representation of the mask for debugging.
        Returns a grid where 0=visible, 1=hidden.
        """
        grid = np.zeros(self.num_patches)
        grid[target_indices.numpy()] = 1
        return grid.reshape(self.grid_size, self.grid_size)


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

    def generate_mask(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generate random 3D mask."""
        num_masked = int(self.num_patches * self.mask_ratio)
        all_indices = np.random.permutation(self.num_patches)

        context_indices = torch.tensor(sorted(all_indices[num_masked:]))
        target_indices = torch.tensor(sorted(all_indices[:num_masked]))

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
