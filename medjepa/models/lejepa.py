"""
LeJEPA: The Complete Model for 2D Medical Images.

This puts together the encoder, predictor, masking, and loss into one model.

Key accuracy techniques (added for accuracy improvement):
- EMA (Exponential Moving Average) target encoder: produces smoother, more
  stable target embeddings — the single biggest accuracy booster in JEPA-style
  models (used by I-JEPA, DINO-v2, BYOL). Overhead: ~5%.
- Multi-scale feature pooling: averages features from the last N encoder
  layers for richer downstream representations.
- Training augmentations: applied on-GPU before the forward pass.
"""

import copy
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
    2. Apply GPU augmentations (if enabled)
    3. Generate masks (which patches to hide)
    4. Encode CONTEXT patches with the online encoder (gradients)
    5. Encode TARGET patches with the EMA encoder (no gradients, smoother)
    6. PREDICT target embeddings from context embeddings
    7. Compute loss: prediction should match target + SIGReg
    8. Update online encoder via backprop; update EMA encoder via momentum

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
        use_ema: bool = True,
        ema_momentum: float = 0.999,
        multiscale_layers: int = 4,
        augmentation: nn.Module = None,
    ):
        super().__init__()
        self.split_encoding = split_encoding
        self.use_ema = use_ema
        self.ema_momentum = ema_momentum
        self.multiscale_layers = multiscale_layers
        self.augmentation = augmentation

        num_patches = (image_size // patch_size) ** 2

        # Online encoder (receives gradients)
        self.encoder = ViTEncoder(
            image_size=image_size,
            patch_size=patch_size,
            in_channels=in_channels,
            embed_dim=embed_dim,
            depth=encoder_depth,
            num_heads=encoder_heads,
            use_checkpoint=gradient_checkpointing,
        )

        # EMA (momentum) target encoder — produces smoother, more stable targets.
        # Initialized as an exact copy of the online encoder; updated each step
        # via exponential moving average: θ_ema = m * θ_ema + (1-m) * θ_online
        if use_ema:
            self.ema_encoder = copy.deepcopy(self.encoder)
            for p in self.ema_encoder.parameters():
                p.requires_grad = False
        else:
            self.ema_encoder = None

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
            "use_ema": use_ema,
            "ema_momentum": ema_momentum,
        }

    @torch.no_grad()
    def update_ema(self):
        """
        Update the EMA target encoder: θ_ema = m * θ_ema + (1-m) * θ_online.

        Called once per optimizer step (by the trainer). Cost: negligible
        (~0.1ms — just a lerp over parameters, no forward/backward).
        """
        if not self.use_ema or self.ema_encoder is None:
            return
        m = self.ema_momentum
        for ema_p, online_p in zip(self.ema_encoder.parameters(),
                                    self.encoder.parameters()):
            ema_p.data.lerp_(online_p.data, 1.0 - m)

    def forward(self, images: torch.Tensor) -> dict:
        """
        One forward pass of training.

        Args:
            images: Batch of images, shape (batch_size, 3, 224, 224)

        Returns:
            Dictionary containing losses
        """
        batch_size = images.shape[0]

        # Step 0: Apply GPU augmentations (if provided)
        if self.augmentation is not None and self.training:
            images = self.augmentation(images)

        # Step 1: Generate mask (move indices to GPU immediately so
        # torch.compile doesn't see CPU tensors in the graph)
        context_indices, target_indices = self.masker.generate_block_mask()
        context_indices = context_indices.to(images.device, non_blocking=True)
        target_indices = target_indices.to(images.device, non_blocking=True)

        # Choose which encoder produces target embeddings
        target_encoder = self.ema_encoder if self.use_ema and self.ema_encoder is not None else self.encoder

        if self.split_encoding:
            # ── Split encoding (fast path) ──────────────────────────────
            # Online encoder: patch embedding + context forward (with grad)
            all_patch_embeds = self.encoder.patch_embed_only(images)
            context_embeddings = self.encoder.forward_from_embeds(
                all_patch_embeds, patch_indices=context_indices
            )

            # Target encoder: produces target embeddings (always no_grad)
            with torch.no_grad():
                if self.use_ema and self.ema_encoder is not None:
                    # EMA encoder runs its own patch embedding for cleaner targets
                    target_patch_embeds = target_encoder.patch_embed_only(images)
                    target_embeddings = target_encoder.forward_from_embeds(
                        target_patch_embeds, patch_indices=target_indices
                    )
                else:
                    target_embeddings = self.encoder.forward_from_embeds(
                        all_patch_embeds.detach(), patch_indices=target_indices
                    )

            # Use ONLY context_embeddings for SIGReg regularization.
            # Including EMA target embeddings (75% of tokens, detached, stable
            # variance) inflates the variance estimate and masks collapse of the
            # context encoder — the variance hinge never fires even when context
            # tokens are collapsing because the EMA targets dominate the std.
            reg_embeddings = context_embeddings
        else:
            # ── Full encoding (legacy path) ─────────────────────────────
            all_embeddings = self.encoder(images)
            context_embeddings = all_embeddings[:, context_indices, :]

            with torch.no_grad():
                if self.use_ema and self.ema_encoder is not None:
                    target_all = target_encoder(images)
                    target_embeddings = target_all[:, target_indices, :]
                else:
                    target_embeddings = all_embeddings[:, target_indices, :]

            # Regularize only context embeddings (same reasoning as split path)
            reg_embeddings = context_embeddings

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

        Uses multi-scale feature pooling: averages features from the last N
        transformer layers (not just the final one). This captures both
        low-level (edges, textures) and high-level (semantic) features,
        producing richer representations for linear probing & few-shot.

        Args:
            images: shape (batch_size, 3, 224, 224)
        Returns:
            Embeddings: shape (batch_size, embed_dim)
        """
        # Use EMA encoder for inference if available (it's more stable)
        encoder = self.ema_encoder if self.use_ema and self.ema_encoder is not None else self.encoder

        with torch.no_grad():
            if self.multiscale_layers > 1:
                # Multi-scale: collect intermediate features
                x = encoder.patch_embed(images)
                x = x + encoder.pos_embed

                layer_outputs = []
                for block in encoder.blocks:
                    x = block(x)
                    layer_outputs.append(x)

                # Average the last N layers
                n = min(self.multiscale_layers, len(layer_outputs))
                stacked = torch.stack(layer_outputs[-n:], dim=0)  # (N, B, T, D)
                pooled = stacked.mean(dim=0)  # (B, T, D)
                pooled = encoder.norm(pooled)
                return pooled.mean(dim=1)  # (B, D)
            else:
                embeddings = encoder(images)
                return embeddings.mean(dim=1)

    def encode_with_grad(self, images: torch.Tensor) -> torch.Tensor:
        """
        Encode images WITH gradient flow — used for full fine-tuning.

        Unlike encode() which wraps in torch.no_grad(), this allows gradients
        to flow through the encoder so it can be updated during fine-tuning.
        Always uses the online encoder (not EMA) since we want to update it.
        """
        encoder = self.encoder  # online encoder for fine-tuning

        if self.multiscale_layers > 1:
            x = encoder.patch_embed(images)
            x = x + encoder.pos_embed

            layer_outputs = []
            for block in encoder.blocks:
                x = block(x)
                layer_outputs.append(x)

            n = min(self.multiscale_layers, len(layer_outputs))
            stacked = torch.stack(layer_outputs[-n:], dim=0)
            pooled = stacked.mean(dim=0)
            pooled = encoder.norm(pooled)
            return pooled.mean(dim=1)
        else:
            embeddings = encoder(images)
            return embeddings.mean(dim=1)
