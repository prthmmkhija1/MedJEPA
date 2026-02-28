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

        # Decode: embed_dim -> image pixels
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
    Compute Dice Score -- the standard segmentation metric.

    Dice = 2 * |A intersect B| / (|A| + |B|)

    Think of it like: how much do the predicted and actual outlines overlap?
    """
    pred = (pred > 0.5).float()
    intersection = (pred * target).sum()
    union = pred.sum() + target.sum()
    dice = (2.0 * intersection + smooth) / (union + smooth)
    return dice.item()


class SegmentationEvaluator:
    """
    Full segmentation evaluation pipeline.

    Given a frozen encoder, trains a segmentation head on 2D slices
    with ground-truth masks, then evaluates Dice score on a held-out set.

    Works with LeJEPA encoder on 2D slices extracted from 3D NIfTI volumes.
    """

    def __init__(
        self,
        pretrained_model,
        embed_dim: int = 768,
        num_classes: int = 2,
        image_size: int = 224,
        patch_size: int = 16,
        lr: float = 1e-3,
        epochs: int = 20,
    ):
        self.device = get_device()
        self.pretrained_model = pretrained_model.to(self.device)
        self.pretrained_model.eval()
        self.embed_dim = embed_dim
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_classes = num_classes
        self.lr = lr
        self.epochs = epochs

        self.seg_head = SimpleSegmentationHead(
            embed_dim=embed_dim,
            num_classes=num_classes,
            image_size=image_size,
            patch_size=patch_size,
        ).to(self.device)

    def _get_patch_embeddings(self, images: torch.Tensor) -> torch.Tensor:
        """Extract patch-level embeddings from the frozen encoder."""
        with torch.no_grad():
            # encoder.forward returns (B, num_patches, embed_dim)
            if hasattr(self.pretrained_model, 'encoder'):
                embeddings = self.pretrained_model.encoder(images)
            else:
                embeddings = self.pretrained_model(images)
        return embeddings

    def train_seg_head(self, train_loader):
        """
        Train the segmentation head on labeled data.

        Expects DataLoader yielding (image, mask) pairs where:
            image: (B, 3, H, W) float tensor
            mask:  (B, H, W) long tensor with class indices 0..num_classes-1
        """
        optimizer = torch.optim.Adam(self.seg_head.parameters(), lr=self.lr)
        criterion = nn.CrossEntropyLoss()

        self.seg_head.train()
        for epoch in range(self.epochs):
            epoch_loss = 0.0
            n_batches = 0

            for batch in train_loader:
                if len(batch) < 2:
                    continue
                images, masks = batch[0].to(self.device), batch[1].to(self.device)

                # Get frozen encoder features (patch-level)
                patch_embs = self._get_patch_embeddings(images)

                # Predict segmentation
                pred = self.seg_head(patch_embs)  # (B, num_classes, H, W)

                # Resize masks if needed
                if masks.shape[-2:] != pred.shape[-2:]:
                    masks = F.interpolate(
                        masks.unsqueeze(1).float(),
                        size=pred.shape[-2:],
                        mode='nearest',
                    ).squeeze(1).long()

                loss = criterion(pred, masks)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                n_batches += 1

            if n_batches > 0:
                avg_loss = epoch_loss / n_batches
                if (epoch + 1) % 5 == 0 or epoch == 0:
                    print(f"  Seg epoch {epoch+1}/{self.epochs} loss: {avg_loss:.4f}")

    def evaluate(self, test_loader) -> dict:
        """
        Evaluate segmentation with Dice score on test data.

        Returns dict with per-class and mean Dice scores.
        """
        self.seg_head.eval()
        all_dice = []
        per_class_dice = {c: [] for c in range(self.num_classes)}

        with torch.no_grad():
            for batch in test_loader:
                if len(batch) < 2:
                    continue
                images, masks = batch[0].to(self.device), batch[1].to(self.device)

                patch_embs = self._get_patch_embeddings(images)
                pred = self.seg_head(patch_embs)  # (B, num_classes, H, W)
                pred_probs = torch.softmax(pred, dim=1)

                # Resize masks if needed
                if masks.shape[-2:] != pred.shape[-2:]:
                    masks = F.interpolate(
                        masks.unsqueeze(1).float(),
                        size=pred.shape[-2:],
                        mode='nearest',
                    ).squeeze(1).long()

                # Compute per-class Dice
                for c in range(self.num_classes):
                    pred_c = pred_probs[:, c, :, :]
                    target_c = (masks == c).float()
                    d = dice_score(pred_c, target_c)
                    per_class_dice[c].append(d)
                    all_dice.append(d)

        results = {
            "mean_dice": sum(all_dice) / max(len(all_dice), 1),
            "per_class_dice": {
                c: sum(v) / max(len(v), 1) for c, v in per_class_dice.items()
            },
            "num_samples": len(all_dice) // max(self.num_classes, 1),
        }
        return results

    def evaluate_without_labels(self, image_loader) -> dict:
        """
        Quick evaluation when no segmentation labels are available.
        Reports prediction statistics (entropy, coverage) as proxy metrics.
        """
        self.seg_head.eval()
        total_coverage = []

        with torch.no_grad():
            for batch in image_loader:
                images = batch[0] if isinstance(batch, (list, tuple)) else batch
                images = images.to(self.device)

                patch_embs = self._get_patch_embeddings(images)
                pred = self.seg_head(patch_embs)
                pred_probs = torch.softmax(pred, dim=1)

                # Foreground coverage: fraction of pixels predicted as non-background
                fg = pred_probs[:, 1:, :, :].sum(dim=1)  # sum non-bg classes
                coverage = (fg > 0.5).float().mean().item()
                total_coverage.append(coverage)

        return {
            "mean_foreground_coverage": sum(total_coverage) / max(len(total_coverage), 1),
            "note": "No ground-truth labels; coverage is a proxy metric",
        }
