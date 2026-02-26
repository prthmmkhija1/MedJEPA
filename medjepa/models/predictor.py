"""
Predictor: Takes context embeddings and predicts target embeddings.

This is a smaller, simpler network than the encoder.
It takes the embeddings from visible patches and predicts
what the embeddings of the hidden patches should be.
"""

import torch
import torch.nn as nn
from typing import Optional


class JEPAPredictor(nn.Module):
    """
    Predicts the embeddings of hidden (target) patches
    given the embeddings of visible (context) patches.

    Architecture: A small Transformer that:
    1. Takes context embeddings as input
    2. Adds learnable "mask tokens" for target positions
    3. Uses attention to predict target embeddings from context
    """

    def __init__(
        self,
        embed_dim: int = 768,    # Must match encoder output dim
        predictor_dim: int = 384, # Predictor is usually smaller
        depth: int = 6,           # Fewer layers than encoder
        num_heads: int = 6,
        num_patches: int = 196,  # Total patches in the image
    ):
        super().__init__()

        # Project from encoder dimension to (smaller) predictor dimension
        self.input_proj = nn.Linear(embed_dim, predictor_dim)

        # Learnable mask tokens — these stand in for the hidden patches
        # The model will transform these into predictions
        self.mask_token = nn.Parameter(torch.randn(1, 1, predictor_dim) * 0.02)

        # Position embeddings for the predictor
        self.pos_embed = nn.Parameter(
            torch.randn(1, num_patches, predictor_dim) * 0.02
        )

        # Transformer blocks for prediction
        self.blocks = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=predictor_dim,
                nhead=num_heads,
                dim_feedforward=predictor_dim * 4,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            for _ in range(depth)
        ])

        self.norm = nn.LayerNorm(predictor_dim)

        # Project back to encoder dimension for loss computation
        self.output_proj = nn.Linear(predictor_dim, embed_dim)

    def forward(
        self,
        context_embeddings: torch.Tensor,
        context_indices: torch.Tensor,
        target_indices: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            context_embeddings: Encoder output for visible patches
                                Shape: (batch_size, num_context, embed_dim)
            context_indices: Which patches are visible (1D tensor of indices)
            target_indices: Which patches are hidden (1D tensor of indices)

        Returns:
            Predicted embeddings for target patches
            Shape: (batch_size, num_target, embed_dim)
        """
        batch_size = context_embeddings.shape[0]
        num_target = len(target_indices)

        # Project context to predictor dimension
        context = self.input_proj(context_embeddings)

        # Add positional encoding to context
        context = context + self.pos_embed[:, context_indices, :]

        # Create mask tokens for target positions
        targets = self.mask_token.expand(batch_size, num_target, -1)
        targets = targets + self.pos_embed[:, target_indices, :]

        # Concatenate context and target tokens
        # The predictor sees both, but only the target tokens need to be predicted
        full_sequence = torch.cat([context, targets], dim=1)

        # Run through Transformer blocks
        for block in self.blocks:
            full_sequence = block(full_sequence)

        # Extract only the target predictions (last num_target tokens)
        target_predictions = full_sequence[:, -num_target:, :]

        # Normalize and project back to encoder dimension
        target_predictions = self.norm(target_predictions)
        target_predictions = self.output_proj(target_predictions)

        return target_predictions
