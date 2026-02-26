"""
V-JEPA: Extension for medical video and 3D volumetric data.

This handles:
- CT scans (stack of 2D slices = 3D volume)
- MRI sequences (3D volume changing over time = 4D)
- Surgical videos (2D frames over time = 3D)
- Cardiac MRI (heart beating = temporal sequence)

Key difference from 2D LeJEPA:
- Patches are 3D cubes (or 2D + time)
- Masking happens in space AND time
- The model must understand BOTH spatial anatomy AND temporal changes
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


class PatchEmbedding3D(nn.Module):
    """
    Convert a 3D volume into a sequence of patch embeddings.

    Example:
    A volume of size (1, 128, 128, 64) with patch size (16, 16, 8)
    becomes a grid of 8 x 8 x 8 = 512 patches.
    Each patch is projected to embed_dim dimensions.
    """

    def __init__(
        self,
        volume_size: Tuple[int, int, int] = (128, 128, 64),
        patch_size: Tuple[int, int, int] = (16, 16, 8),
        in_channels: int = 1,  # Medical images are usually grayscale
        embed_dim: int = 768,
    ):
        super().__init__()
        self.patch_size = patch_size
        self.grid_size = tuple(v // p for v, p in zip(volume_size, patch_size))
        self.num_patches = self.grid_size[0] * self.grid_size[1] * self.grid_size[2]

        # 3D convolution for patch extraction
        self.projection = nn.Conv3d(
            in_channels, embed_dim,
            kernel_size=patch_size, stride=patch_size
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Volume, shape (batch, channels, D, H, W)
        Returns:
            Patch embeddings, shape (batch, num_patches, embed_dim)
        """
        x = self.projection(x)                    # (B, E, gD, gH, gW)
        x = x.flatten(2)                           # (B, E, num_patches)
        x = x.transpose(1, 2)                      # (B, num_patches, E)
        return x


class VJEPA(nn.Module):
    """
    V-JEPA model for 3D medical data.

    Same principle as LeJEPA but in 3D:
    - Hide some 3D cubes of the volume
    - Predict their embeddings from the visible cubes
    - Use SIGReg to prevent collapse
    """

    def __init__(
        self,
        volume_size: Tuple[int, int, int] = (128, 128, 64),
        patch_size: Tuple[int, int, int] = (16, 16, 8),
        in_channels: int = 1,
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        predictor_dim: int = 384,
        predictor_depth: int = 6,
        mask_ratio: float = 0.75,
        lambda_reg: float = 1.0,
    ):
        super().__init__()

        grid_size = tuple(v // p for v, p in zip(volume_size, patch_size))
        num_patches = grid_size[0] * grid_size[1] * grid_size[2]

        # 3D patch embedding
        self.patch_embed = PatchEmbedding3D(
            volume_size, patch_size, in_channels, embed_dim
        )

        # 3D positional encoding
        self.pos_embed = nn.Parameter(
            torch.randn(1, num_patches, embed_dim) * 0.02
        )

        # Transformer blocks (same as 2D, attention doesn't care about dimensionality)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        self.encoder_norm = nn.LayerNorm(embed_dim)

        # Predictor (same architecture as 2D)
        predictor_layer = nn.TransformerEncoderLayer(
            d_model=predictor_dim,
            nhead=num_heads // 2,
            dim_feedforward=predictor_dim * 4,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.predictor_input_proj = nn.Linear(embed_dim, predictor_dim)
        self.predictor = nn.TransformerEncoder(
            predictor_layer, num_layers=predictor_depth
        )
        self.predictor_norm = nn.LayerNorm(predictor_dim)
        self.predictor_output_proj = nn.Linear(predictor_dim, embed_dim)

        self.mask_token = nn.Parameter(
            torch.randn(1, 1, predictor_dim) * 0.02
        )
        self.predictor_pos_embed = nn.Parameter(
            torch.randn(1, num_patches, predictor_dim) * 0.02
        )

        self.mask_ratio = mask_ratio
        self.num_patches = num_patches

        # Loss
        from medjepa.training.losses import SIGRegLoss
        self.loss_fn = SIGRegLoss(lambda_reg=lambda_reg)

    def _generate_mask(self):
        """Generate random 3D mask."""
        import numpy as np
        num_masked = int(self.num_patches * self.mask_ratio)
        indices = np.random.permutation(self.num_patches)
        context = torch.tensor(sorted(indices[num_masked:]))
        target = torch.tensor(sorted(indices[:num_masked]))
        return context, target

    def forward(self, volumes: torch.Tensor) -> dict:
        """
        Args:
            volumes: shape (batch, channels, D, H, W)
        """
        # Patch embed
        x = self.patch_embed(volumes) + self.pos_embed

        # Generate mask
        ctx_idx, tgt_idx = self._generate_mask()
        ctx_idx = ctx_idx.to(volumes.device)
        tgt_idx = tgt_idx.to(volumes.device)

        # Encode all patches
        all_embeddings = self.encoder_norm(self.encoder(x))

        # Separate context and target
        context_emb = all_embeddings[:, ctx_idx, :]
        target_emb = all_embeddings[:, tgt_idx, :]

        # Predict targets
        batch_size = volumes.shape[0]
        pred_ctx = self.predictor_input_proj(context_emb)
        pred_ctx = pred_ctx + self.predictor_pos_embed[:, ctx_idx, :]

        mask_tokens = self.mask_token.expand(batch_size, len(tgt_idx), -1)
        mask_tokens = mask_tokens + self.predictor_pos_embed[:, tgt_idx, :]

        pred_input = torch.cat([pred_ctx, mask_tokens], dim=1)
        pred_output = self.predictor_norm(self.predictor(pred_input))
        predicted = self.predictor_output_proj(pred_output[:, -len(tgt_idx):, :])

        # Loss
        losses = self.loss_fn(predicted, target_emb.detach(), all_embeddings)
        return losses

    def encode(self, volumes: torch.Tensor) -> torch.Tensor:
        """Get volume representations for downstream tasks."""
        with torch.no_grad():
            x = self.patch_embed(volumes) + self.pos_embed
            x = self.encoder_norm(self.encoder(x))
            return x.mean(dim=1)
