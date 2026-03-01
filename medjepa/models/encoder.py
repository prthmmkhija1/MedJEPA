"""
Encoder: Takes image patches and converts them into meaningful embeddings.

Uses Vision Transformer (ViT) architecture.

Think of it like this:
1. Cut the image into small patches (like 16x16 pixel squares)
2. Flatten each patch into a 1D vector
3. Add position information (so the model knows where each patch was)
4. Run through Transformer layers (these learn relationships between patches)
5. Output: one embedding vector per patch
"""

import torch
import torch.nn as nn
import math
from typing import Optional
from torch.utils.checkpoint import checkpoint as grad_checkpoint

# Enable PyTorch 2.x scaled_dot_product_attention (Flash Attention on A100)
if hasattr(torch.backends, "cuda"):
    try:
        torch.backends.cuda.enable_flash_sdp(True)
        torch.backends.cuda.enable_mem_efficient_sdp(True)
    except Exception:
        pass  # CPU-only build or older PyTorch


class PatchEmbedding(nn.Module):
    """
    Step 1: Cut image into patches and project each patch to embedding dimension.

    Example: A 224x224 image with 16x16 patches becomes 14x14 = 196 patches.
    Each 16x16x3 patch (768 numbers) gets projected to, say, 768 numbers.
    """

    def __init__(
        self,
        image_size: int = 224,
        patch_size: int = 16,
        in_channels: int = 3,      # 3 for RGB
        embed_dim: int = 768,      # Size of each embedding vector
    ):
        super().__init__()
        self.patch_size = patch_size
        self.num_patches = (image_size // patch_size) ** 2
        self.embed_dim = embed_dim

        # This conv layer does the patch cutting AND projection in one step
        # It's a clever trick: a convolution with kernel_size=patch_size
        # and stride=patch_size is equivalent to cutting into non-overlapping
        # patches and projecting each one
        self.projection = nn.Conv2d(
            in_channels, embed_dim,
            kernel_size=patch_size, stride=patch_size
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Images, shape (batch_size, 3, 224, 224)
        Returns:
            Patch embeddings, shape (batch_size, num_patches, embed_dim)
        """
        # channels_last memory format: ~10-30% faster Conv2d on A100
        x = x.contiguous(memory_format=torch.channels_last)
        # x shape: (B, 3, 224, 224) → (B, embed_dim, 14, 14)
        x = self.projection(x)
        # Flatten spatial dimensions: (B, embed_dim, 14, 14) → (B, embed_dim, 196)
        x = x.flatten(2)
        # Transpose: (B, embed_dim, 196) → (B, 196, embed_dim)
        x = x.transpose(1, 2)
        return x


class TransformerBlock(nn.Module):
    """
    One layer of the Transformer.

    Each block does:
    1. Self-attention: each patch looks at all other patches to understand context
    2. Feed-forward: process the information through a small neural network
    3. Residual connections and layer normalization for stable training
    """

    def __init__(
        self,
        embed_dim: int = 768,
        num_heads: int = 12,     # Number of attention "perspectives"
        mlp_ratio: float = 4.0,  # Feed-forward layer is 4x wider
        dropout: float = 0.0,
    ):
        super().__init__()

        # Layer normalization (keeps values in a nice range)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)

        # Multi-head self-attention
        self.attention = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=dropout, batch_first=True
        )

        # Feed-forward network (MLP)
        mlp_dim = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_dim),
            nn.GELU(),  # Activation function
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: shape (batch_size, num_patches, embed_dim)
        Returns:
            Same shape, but with refined representations
        """
        # Self-attention with residual connection
        normed = self.norm1(x)
        attended, _ = self.attention(normed, normed, normed)
        x = x + attended

        # Feed-forward with residual connection
        x = x + self.mlp(self.norm2(x))
        return x


class ViTEncoder(nn.Module):
    """
    The full Vision Transformer encoder.

    Puts together: Patch Embedding + Positional Encoding + Transformer Blocks
    """

    def __init__(
        self,
        image_size: int = 224,
        patch_size: int = 16,
        in_channels: int = 3,
        embed_dim: int = 768,
        depth: int = 12,         # Number of Transformer blocks
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        use_checkpoint: bool = False,  # Gradient checkpointing (saves memory)
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.use_checkpoint = use_checkpoint
        num_patches = (image_size // patch_size) ** 2

        # Step 1: Patch embedding
        self.patch_embed = PatchEmbedding(
            image_size, patch_size, in_channels, embed_dim
        )

        # Step 2: Positional encoding
        # Each patch gets a learnable position vector so the model
        # knows WHERE in the image each patch is
        self.pos_embed = nn.Parameter(
            torch.randn(1, num_patches, embed_dim) * 0.02
        )

        # Step 3: Stack of Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, mlp_ratio, dropout)
            for _ in range(depth)
        ])

        # Final normalization
        self.norm = nn.LayerNorm(embed_dim)

    def forward(
        self,
        x: torch.Tensor,
        patch_indices: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x: Images, shape (batch_size, 3, 224, 224)
            patch_indices: Optional — which patches to process
                           (used for masking: only encode visible patches)

        Returns:
            Embeddings, shape (batch_size, num_patches, embed_dim)
        """
        # Convert image to patch embeddings
        x = self.patch_embed(x)

        # Add positional encoding
        if patch_indices is not None:
            # Only add positions for selected patches
            pos = self.pos_embed[:, patch_indices, :]
            x = x[:, patch_indices, :] if x.shape[1] != len(patch_indices) else x
            x = x + pos
        else:
            x = x + self.pos_embed

        # Run through Transformer blocks
        for block in self.blocks:
            if self.use_checkpoint and self.training:
                x = grad_checkpoint(block, x, use_reentrant=False)
            else:
                x = block(x)

        # Final normalization
        x = self.norm(x)
        return x

    def patch_embed_only(self, x: torch.Tensor) -> torch.Tensor:
        """
        Run ONLY the patch embedding conv2d (no positional enc, no transformer).
        Returns all patch embeddings: shape (batch, num_patches, embed_dim).
        Call this once and share the result between context and target paths
        to avoid running the conv2d twice per step.
        """
        x = x.contiguous(memory_format=torch.channels_last)
        x = self.patch_embed.projection(x)   # (B, embed_dim, H, W)
        x = x.flatten(2).transpose(1, 2)     # (B, num_patches, embed_dim)
        return x

    def forward_from_embeds(
        self,
        all_patch_embeds: torch.Tensor,
        patch_indices: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Run the positional encoding + transformer blocks on pre-computed
        patch embeddings from ``patch_embed_only()``.

        Args:
            all_patch_embeds: (batch, num_patches, embed_dim) — output of
                              ``patch_embed_only()``.
            patch_indices:    Which patches to select. None = all patches.

        Returns:
            Encodings: (batch, len(patch_indices) or num_patches, embed_dim)
        """
        if patch_indices is not None:
            x = all_patch_embeds[:, patch_indices, :] + self.pos_embed[:, patch_indices, :]
        else:
            x = all_patch_embeds + self.pos_embed

        for block in self.blocks:
            if self.use_checkpoint and self.training:
                x = grad_checkpoint(block, x, use_reentrant=False)
            else:
                x = block(x)

        return self.norm(x)
