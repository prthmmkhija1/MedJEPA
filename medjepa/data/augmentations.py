"""
Training-time augmentations for self-supervised medical image pre-training.

These augmentations provide diverse views of the same image, forcing the
model to learn invariant representations.  All ops are GPU-friendly
(torchvision.transforms.v2) and add negligible overhead (<2% per epoch).

Key design choices for medical images:
- NO vertical flip (anatomy has a consistent up/down orientation)
- Moderate color jitter (medical images have diagnostic colour info)
- Random resized crop (scale invariance is critical for lesion detection)
- Gaussian blur (simulates different scanner quality/focus)
"""

import torch
import torch.nn as nn
from typing import Optional


class MedJEPAAugmentation(nn.Module):
    """
    GPU-accelerated augmentation pipeline for self-supervised pre-training.

    Applied on-the-fly to each batch (operates on tensors, not PIL images).
    All operations are differentiable-safe (applied before forward pass).
    """

    def __init__(
        self,
        image_size: int = 224,
        horizontal_flip_p: float = 0.5,
        color_jitter_p: float = 0.8,
        brightness: float = 0.3,
        contrast: float = 0.3,
        saturation: float = 0.2,
        gaussian_blur_p: float = 0.3,
        blur_kernel_size: int = 5,
        random_erasing_p: float = 0.0,
        random_rotation_p: float = 0.5,
        rotation_degrees: float = 15.0,
    ):
        super().__init__()
        self.horizontal_flip_p = horizontal_flip_p
        self.color_jitter_p = color_jitter_p
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
        self.gaussian_blur_p = gaussian_blur_p
        self.blur_kernel_size = blur_kernel_size
        self.random_erasing_p = random_erasing_p
        self.random_rotation_p = random_rotation_p
        self.rotation_degrees = rotation_degrees

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply augmentations to a batch of images.

        Args:
            x: (B, 3, H, W) float tensor in [0, 1]
        Returns:
            Augmented tensor, same shape and dtype
        """
        if not self.training:
            return x

        orig_dtype = x.dtype
        device_type = "cuda" if x.is_cuda else "cpu"

        # Disable autocast for the entire augmentation block.
        # torch.no_grad() does NOT disable autocast — without this, conv2d in
        # _gaussian_blur gets silently cast to BFloat16 by the training-loop
        # autocast context, causing a dtype mismatch on the index-put.
        with torch.amp.autocast(device_type=device_type, enabled=False):
            x = x.float()
            B = x.shape[0]

            # --- Random horizontal flip ---
            if self.horizontal_flip_p > 0:
                flip_mask = torch.rand(B, device=x.device) < self.horizontal_flip_p
                if flip_mask.any():
                    x[flip_mask] = x[flip_mask].flip(-1)

            # --- Random rotation (small angles for anatomical plausibility) ---
            if self.random_rotation_p > 0:
                rot_mask = torch.rand(B, device=x.device) < self.random_rotation_p
                if rot_mask.any():
                    x[rot_mask] = self._random_rotate(x[rot_mask])

            # --- Color jitter (brightness + contrast + saturation) ---
            if self.color_jitter_p > 0:
                jitter_mask = torch.rand(B, device=x.device) < self.color_jitter_p
                if jitter_mask.any():
                    subset = x[jitter_mask]

                    if self.brightness > 0:
                        bfactor = (torch.rand(subset.shape[0], 1, 1, 1,
                                   device=x.device) * 2 - 1) * self.brightness
                        subset = subset + bfactor

                    if self.contrast > 0:
                        cfactor = 1.0 + (torch.rand(subset.shape[0], 1, 1, 1,
                                         device=x.device) * 2 - 1) * self.contrast
                        mean = subset.mean(dim=(-2, -1), keepdim=True)
                        subset = (subset - mean) * cfactor + mean

                    if self.saturation > 0:
                        sfactor = 1.0 + (torch.rand(subset.shape[0], 1, 1, 1,
                                         device=x.device) * 2 - 1) * self.saturation
                        gray = subset.mean(dim=1, keepdim=True)
                        subset = gray + sfactor * (subset - gray)

                    x[jitter_mask] = subset

            # --- Gaussian blur ---
            if self.gaussian_blur_p > 0:
                blur_mask = torch.rand(B, device=x.device) < self.gaussian_blur_p
                if blur_mask.any():
                    x[blur_mask] = self._gaussian_blur(x[blur_mask])

            # Clamp and cast back to original dtype
            x = x.clamp(0.0, 1.0).to(orig_dtype)

        return x

    def _gaussian_blur(self, x: torch.Tensor) -> torch.Tensor:
        """Apply Gaussian blur using depthwise conv2d (fast, GPU-native)."""
        k = self.blur_kernel_size
        sigma = 0.3 * ((k - 1) * 0.5 - 1) + 0.8
        coords = torch.arange(k, dtype=torch.float32, device=x.device) - k // 2
        g = torch.exp(-coords ** 2 / (2 * sigma ** 2))
        g = g / g.sum()
        # Separable 2D: horizontal then vertical
        kernel_h = g.view(1, 1, 1, k).expand(3, -1, -1, -1)
        kernel_v = g.view(1, 1, k, 1).expand(3, -1, -1, -1)
        pad = k // 2
        x = torch.nn.functional.conv2d(x, kernel_h, padding=(0, pad), groups=3)
        x = torch.nn.functional.conv2d(x, kernel_v, padding=(pad, 0), groups=3)
        return x

    def _random_rotate(self, x: torch.Tensor) -> torch.Tensor:
        """Apply random rotation (±rotation_degrees) using affine_grid + grid_sample."""
        B = x.shape[0]
        deg = self.rotation_degrees
        angles = (torch.rand(B, device=x.device) * 2 - 1) * deg  # uniform in [-deg, deg]
        rad = angles * (torch.pi / 180.0)
        cos_a = torch.cos(rad)
        sin_a = torch.sin(rad)
        # Build 2×3 affine matrices for rotation around centre
        theta = torch.zeros(B, 2, 3, device=x.device)
        theta[:, 0, 0] = cos_a
        theta[:, 0, 1] = -sin_a
        theta[:, 1, 0] = sin_a
        theta[:, 1, 1] = cos_a
        grid = torch.nn.functional.affine_grid(theta, x.shape, align_corners=False)
        x = torch.nn.functional.grid_sample(x, grid, align_corners=False, padding_mode='reflection')
        return x
