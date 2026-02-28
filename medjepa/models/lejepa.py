"""
LeJEPA: The Complete Model for 2D Medical Images.

This puts together the encoder, predictor, masking, and loss into one model.
"""

import torch
import torch.nn as nn
from medjepa.models.encoder import ViTEncoder
from medjepa.models.predictor import JEPAPredictor
from medjepa.data.masking import PatchMasker2D
from medjepa.training.losses import SIGRegLoss


class LeJEPA(nn.Module):
    """
    LeJEPA for medical image self-supervised learning.

    Training flow:
    1. Take a batch of medical images
    2. Generate masks (which patches to hide)
    3. Encode CONTEXT patches → get context embeddings
    4. Encode TARGET patches → get target embeddings (ground truth)
    5. PREDICT target embeddings from context embeddings
    6. Compute loss: prediction should match target + SIGReg
    7. Update the model to minimize the loss

    Split encoding (default, ~2-3x faster):
        Instead of encoding ALL 196 patches through the 12-layer transformer,
        we encode only the ~49 context patches with gradients and the ~147
        target patches *without* gradients (they are detached anyway).
        Self-attention is O(n²), so processing fewer tokens is a huge win:
          Before: forward O(196²×12) + backward O(196²×12)
          After:  forward O(49²×12) + forward O(147²×12) no_grad
                  + backward O(49²×12) only
        Net effect: ~60-65% less total compute.
    """

    def __init__(
        self,
        image_size: int = 224,
        patch_size: int = 16,
        in_channels: int = 3,
        embed_dim: int = 768,
        encoder_depth: int = 12,
        encoder_heads: int = 12,
        predictor_dim: int = 384,
        predictor_depth: int = 6,
        predictor_heads: int = 6,
        mask_ratio: float = 0.75,
        lambda_reg: float = 1.0,
        split_encoding: bool = True,
        gradient_checkpointing: bool = False,
    ):
        super().__init__()
        self.split_encoding = split_encoding

        num_patches = (image_size // patch_size) ** 2

        # The ONE encoder (no momentum encoder, no teacher — just one!)
        self.encoder = ViTEncoder(
            image_size=image_size,
            patch_size=patch_size,
            in_channels=in_channels,
            embed_dim=embed_dim,
            depth=encoder_depth,
            num_heads=encoder_heads,
            use_checkpoint=gradient_checkpointing,
        )

        # The predictor (smaller than encoder)
        self.predictor = JEPAPredictor(
            embed_dim=embed_dim,
            predictor_dim=predictor_dim,
            depth=predictor_depth,
            num_heads=predictor_heads,
            num_patches=num_patches,
            use_checkpoint=gradient_checkpointing,
        )

        # Masking strategy
        self.masker = PatchMasker2D(
            image_size=image_size,
            patch_size=patch_size,
            mask_ratio=mask_ratio,
        )

        # Loss function
        self.loss_fn = SIGRegLoss(lambda_reg=lambda_reg)

        # Save config
        self.config = {
            "image_size": image_size,
            "patch_size": patch_size,
            "embed_dim": embed_dim,
            "encoder_depth": encoder_depth,
            "predictor_depth": predictor_depth,
            "mask_ratio": mask_ratio,
            "lambda_reg": lambda_reg,
            "split_encoding": split_encoding,
            "gradient_checkpointing": gradient_checkpointing,
        }

    def forward(self, images: torch.Tensor) -> dict:
        """
        One forward pass of training.

        Args:
            images: Batch of images, shape (batch_size, 3, 224, 224)

        Returns:
            Dictionary containing losses
        """
        batch_size = images.shape[0]

        # Step 1: Generate mask (move indices to GPU immediately so
        # torch.compile doesn't see CPU tensors in the graph)
        context_indices, target_indices = self.masker.generate_block_mask()
        context_indices = context_indices.to(images.device, non_blocking=True)
        target_indices = target_indices.to(images.device, non_blocking=True)

        if self.split_encoding:
            # ── Split encoding (fast path) ──────────────────────────────
            # Run the patch embedding conv2d ONCE for all 196 patches,
            # then fork: context path gets grad, target path gets no_grad.
            # This saves one full conv2d pass compared to calling encoder()
            # twice with different patch_indices.
            all_patch_embeds = self.encoder.patch_embed_only(images)

            # Context: positional enc + transformer on ~49 patches WITH gradients
            context_embeddings = self.encoder.forward_from_embeds(
                all_patch_embeds, patch_indices=context_indices
            )

            # Target: positional enc + transformer on ~147 patches WITHOUT gradients
            # (targets are always detached in the loss anyway)
            with torch.no_grad():
                target_embeddings = self.encoder.forward_from_embeds(
                    all_patch_embeds.detach(), patch_indices=target_indices
                )

            # Use context embeddings for SIGReg regularization
            reg_embeddings = context_embeddings
        else:
            # ── Full encoding (legacy path) ─────────────────────────────
            all_embeddings = self.encoder(images)
            context_embeddings = all_embeddings[:, context_indices, :]
            target_embeddings = all_embeddings[:, target_indices, :]
            reg_embeddings = all_embeddings

        # Step 4: Predict target embeddings from context
        predicted = self.predictor(
            context_embeddings, context_indices, target_indices
        )

        # Step 5: Compute loss
        # IMPORTANT: target_embeddings are detached (no gradient flows through them)
        # This is NOT a heuristic — it's part of the mathematical formulation
        losses = self.loss_fn(
            predicted_target=predicted,
            actual_target=target_embeddings.detach(),
            all_embeddings=reg_embeddings,
        )

        return losses

    def encode(self, images: torch.Tensor) -> torch.Tensor:
        """
        Encode images to get their representations.
        Used AFTER training for downstream tasks (classification, etc.)

        Args:
            images: shape (batch_size, 3, 224, 224)
        Returns:
            Embeddings: shape (batch_size, embed_dim)
        """
        with torch.no_grad():
            embeddings = self.encoder(images)
            # Average all patch embeddings → one embedding per image
            return embeddings.mean(dim=1)
