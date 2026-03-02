"""
Loss functions for MedJEPA training.

The "loss" is a number that tells the model how wrong it is.
Lower loss = better predictions. The model tries to minimize this number.

SIGReg = Sketched Isotropic Gaussian Regularization
This is the KEY innovation of LeJEPA. It prevents "collapse" — when the model
cheats by making all embeddings identical (which would be useless).

The "Sketched" part uses random projections to compute the covariance
regularisation efficiently in a lower-dimensional space, which is critical
when embed_dim is large.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional


class SIGRegLoss(nn.Module):
    """
    Sketched Isotropic Gaussian Regularization loss.

    Has THREE parts:
    1. PREDICTION LOSS: The predicted target embeddings should match
       the actual target embeddings (MSE — Mean Squared Error).

    2. VARIANCE LOSS: Each embedding dimension should have non-trivial
       variance (prevents collapse to a constant).

    3. COVARIANCE (SKETCHED) LOSS: The off-diagonal elements of the
       covariance matrix (computed in a *sketched* lower-dim space via
       random projection) should be close to zero.  Combined with the
       variance term this pushes the distribution toward an isotropic
       Gaussian.

    The single trade-off hyperparameter (lambda_reg) balances these goals.
    """

    def __init__(
        self,
        lambda_reg: float = 1.0,
        # The ONE hyperparameter: how much to emphasize regularization
        # vs prediction accuracy. Start with 1.0.
        sketch_dim: int = 256,
        # Dimensionality of the random sketch. Lower → faster but noisier.
        # Default 256 is a good balance.  Set to 0 to disable sketching
        # and use the full covariance (legacy behaviour).
        variance_target: float = 1.0,
        # Target standard deviation per dimension. Embeddings are pushed
        # so that each coordinate's std is close to this value.
        lambda_var: float = 2.5,
        # Weight for the variance regularization term.
        # Must be > ~1.5 to guarantee that full representation collapse is
        # energetically unfavorable (pred_loss decrease < var_loss increase).
        # 2.5 gives a comfortable safety margin.  In the healthy state
        # (cross-image std ≈ 1), var contribution ≈ 2.5 × 0.14 = 0.35;
        # during collapse (std ≈ 0) it surges to ≈ 2.5, strongly opposing
        # the encoder's tendency to produce image-independent embeddings.
        lambda_cov: float = 0.04,
        # Weight for the off-diagonal covariance term.
    ):
        super().__init__()
        self.lambda_reg = lambda_reg
        self.sketch_dim = sketch_dim
        self.variance_target = variance_target
        self.lambda_var = lambda_var
        self.lambda_cov = lambda_cov

        # Random projection matrix — registered as a buffer so it moves
        # with .to(device) and doesn't trigger CUDA-graph recompilation.
        # Built lazily on first forward pass once we know embed_dim.
        self._sketch_embed_dim: Optional[int] = None
        self.register_buffer("_sketch_matrix", torch.empty(0))

    def _get_sketch_matrix(self, embed_dim: int, device: torch.device) -> torch.Tensor:
        """
        Return a fixed random Gaussian projection matrix R of shape
        (embed_dim, sketch_dim) scaled by 1/sqrt(sketch_dim).

        This is re-created if embed_dim changes, it hasn't been built yet,
        or the matrix is on a different device (e.g. after model.to(device)).
        The matrix is registered as a buffer (no gradients, moves with device).
        """
        if (self._sketch_matrix.numel() == 0
                or self._sketch_embed_dim != embed_dim
                or self._sketch_matrix.device != device):
            k = min(self.sketch_dim, embed_dim)
            R = torch.randn(embed_dim, k, device=device) / math.sqrt(k)
            # Re-register as a proper buffer so it follows .to(device) calls.
            # NOTE: register_buffer inside forward can confuse torch.compile,
            # but the branch only fires on first call (or device change), not
            # every step — so the compile graph is stable after warm-up.
            self.register_buffer("_sketch_matrix", R)
            self._sketch_embed_dim = embed_dim
        return self._sketch_matrix

    def prediction_loss(
        self,
        predicted: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        """
        How different are the predictions from the actual embeddings?
        Uses MSE on raw (un-normalised) embeddings — this is the formulation
        used in the original I-JEPA paper and provides meaningful gradients
        throughout training.  Smooth-L1 on L2-normalised vectors compresses
        the loss to ~0.0003 within a few epochs, starving the model of
        learning signal.

        Args:
            predicted: What the predictor thinks the target embeddings are
            target: What the target embeddings actually are (from the encoder)
        """
        # Cast to float32 — bfloat16 mixed-precision can squash small
        # differences to zero, starving the encoder of gradient signal.
        loss = F.mse_loss(predicted.float(), target.float())
        return loss

    def variance_loss(
        self,
        embeddings: torch.Tensor,
    ) -> torch.Tensor:
        """
        Encourage each (position, dimension) to have non-trivial standard
        deviation **across images in the batch** (dim=0).

        CRITICAL: the std must be computed across the BATCH dimension,
        NOT across the flattened (batch × tokens) axis.  Flattening
        conflates positional diversity with image diversity, masking
        "position-only collapse" — where the encoder ignores image content
        and outputs a fixed embedding per spatial position.  In that case
        the cross-token std is high (different positions → different
        embeddings) yet the representation carries zero image information.

        By computing std across images for each (position, dimension), this
        term fires exactly when all images produce the same embedding at a
        given position — the collapse we must prevent.

        Uses softplus(target - std, beta=5) so the gradient is never zero.
        """
        # embeddings: (B, T, D)
        emb = embeddings.float()
        emb = torch.nan_to_num(emb, nan=0.0, posinf=1.0, neginf=-1.0)

        # Per-(position, dimension) std across the BATCH of images
        std = emb.std(dim=0, unbiased=False)               # (T, D)
        std = torch.nan_to_num(std, nan=0.0, posinf=0.0)

        var_loss = torch.mean(F.softplus(self.variance_target - std, beta=5.0))
        return var_loss

    def covariance_loss_sketched(
        self,
        embeddings: torch.Tensor,
    ) -> torch.Tensor:
        """
        Off-diagonal covariance loss computed on **mean-pooled** image
        representations in a sketched lower-dimensional space.

        Embeddings are first averaged across tokens to yield one vector
        per image (B, D).  Covariance is then computed across the batch.
        This decorrelates the per-image representation dimensions
        (VICReg / SIGReg formulation).

        Computing covariance on the flattened (B×T, D) matrix conflates
        positional variation with cross-image variation, letting the model
        satisfy the constraint through position-only diversity while
        image content collapses.

        Steps:
        1. Mean-pool tokens: (B, T, D) → (B, D)
        2. Center across batch
        3. Sketch-project: (B, D) → (B, k)
        4. Cov = X^T X / (B-1)
        5. Penalise off-diagonal entries
        """
        batch_size, num_tokens, embed_dim = embeddings.shape
        # Pool across tokens → one vector per image
        pooled = embeddings.mean(dim=1).float()            # (B, D)
        pooled = torch.nan_to_num(pooled, nan=0.0, posinf=1.0, neginf=-1.0)

        # Center across batch
        pooled = pooled - pooled.mean(dim=0, keepdim=True)
        n = pooled.shape[0]  # B

        if self.sketch_dim > 0 and self.sketch_dim < embed_dim:
            # --- Sketched covariance ---
            R = self._get_sketch_matrix(embed_dim, embeddings.device)
            projected = pooled @ R                         # (B, sketch_dim)
            k = projected.shape[1]
            cov = (projected.T @ projected) / max(n - 1, 1)  # (k, k)
        else:
            # --- Full covariance (legacy / small embed_dim) ---
            k = embed_dim
            cov = (pooled.T @ pooled) / max(n - 1, 1)     # (D, D)

        # Off-diagonal penalty: zero out diagonal, penalise the rest
        diag = torch.diagonal(cov)
        off_diag = cov - torch.diag(diag)
        cov_loss = (off_diag ** 2).sum() / k
        cov_loss = torch.nan_to_num(cov_loss, nan=0.0, posinf=0.0)
        return cov_loss

    def regularization_loss(
        self,
        embeddings: torch.Tensor,
    ) -> torch.Tensor:
        """
        Combined regularization: variance + sketched covariance.

        Together they push the embedding distribution toward an isotropic
        Gaussian (the "IG" in SIGReg).

        Returns a single scalar combining both terms.
        """
        var_loss = self.variance_loss(embeddings)
        cov_loss = self.covariance_loss_sketched(embeddings)
        return self.lambda_var * var_loss + self.lambda_cov * cov_loss

    def forward(
        self,
        predicted_target: torch.Tensor,
        actual_target: torch.Tensor,
        all_embeddings: torch.Tensor,
    ) -> dict:
        """
        Compute the total LeJEPA loss.

        Args:
            predicted_target: Predictor's guess for target embeddings
            actual_target: Encoder's actual target embeddings
            all_embeddings: All embeddings (for regularization)

        Returns:
            Dictionary with total loss and components
        """
        pred_loss = self.prediction_loss(predicted_target, actual_target)
        var_loss = self.variance_loss(all_embeddings)
        cov_loss = self.covariance_loss_sketched(all_embeddings)

        # nan_to_num: replaces any residual NaN with 0 so total_loss stays finite
        var_loss = torch.nan_to_num(var_loss, nan=0.0)
        cov_loss = torch.nan_to_num(cov_loss, nan=0.0)
        reg_loss = self.lambda_var * var_loss + self.lambda_cov * cov_loss

        total_loss = pred_loss + self.lambda_reg * reg_loss
        # nan_to_num: replace residual NaN with pred_loss (no .item() sync)
        total_loss = torch.nan_to_num(total_loss, nan=0.0, posinf=10.0)

        return {
            "total_loss": total_loss,
            "prediction_loss": pred_loss,
            "regularization_loss": reg_loss,
            "variance_loss": var_loss,
            "covariance_loss": cov_loss,
        }
