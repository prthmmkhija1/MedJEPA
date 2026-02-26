"""
Segmentation evaluation for MedJEPA.

Uses the pre-trained encoder as a feature extractor,
then adds a simple segmentation decoder to produce pixel-level predictions.

Metric: Dice Score
- 1.0 = perfect overlap between predicted and ground truth masks
- 0.0 = no overlap at all
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from medjepa.utils.device import get_device


class SimpleSegmentationHead(nn.Module):
    """
    A simple decoder that converts patch embeddings into a segmentation mask.

    Takes encoder output (patch embeddings) and upsamples them to pixel-level
    predictions (same size as the original image).
    """

    def __init__(
        self,
        embed_dim: int = 768,
        num_classes: int = 2,    # Usually 2: background + foreground (lesion)
        image_size: int = 224,
        patch_size: int = 16,
    ):
        super().__init__()
        self.grid_size = image_size // patch_size
        self.image_size = image_size

        # Decode: embed_dim → image pixels
        self.decoder = nn.Sequential(
            nn.Linear(embed_dim, 256),
            nn.ReLU(),
            nn.Linear(256, patch_size * patch_size * num_classes),
        )
        self.num_classes = num_classes
        self.patch_size = patch_size

    def forward(self, patch_embeddings: torch.Tensor) -> torch.Tensor:
        """
        Args:
            patch_embeddings: (batch, num_patches, embed_dim)
        Returns:
            Segmentation mask: (batch, num_classes, H, W)
        """
        batch_size = patch_embeddings.shape[0]

        # Decode each patch
        decoded = self.decoder(patch_embeddings)
        # Shape: (batch, num_patches, patch_size * patch_size * num_classes)

        # Reshape to spatial grid
        decoded = decoded.reshape(
            batch_size, self.grid_size, self.grid_size,
            self.patch_size, self.patch_size, self.num_classes
        )

        # Rearrange to full image
        decoded = decoded.permute(0, 5, 1, 3, 2, 4)
        decoded = decoded.reshape(
            batch_size, self.num_classes, self.image_size, self.image_size
        )

        return decoded


def dice_score(pred: torch.Tensor, target: torch.Tensor, smooth: float = 1e-5) -> float:
    """
    Compute Dice Score — the standard segmentation metric.

    Dice = 2 * |A ∩ B| / (|A| + |B|)

    Think of it like: how much do the predicted and actual outlines overlap?
    """
    pred = (pred > 0.5).float()
    intersection = (pred * target).sum()
    union = pred.sum() + target.sum()
    dice = (2.0 * intersection + smooth) / (union + smooth)
    return dice.item()
