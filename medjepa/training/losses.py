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
        lambda_var: float = 1.0,
        # Weight for the variance regularization term.
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
        self._sketch_embed_dim: int | None = None
        self.register_buffer("_sketch_matrix", None)

    def _get_sketch_matrix(self, embed_dim: int, device: torch.device) -> torch.Tensor:
        """
        Return a fixed random Gaussian projection matrix R of shape
        (embed_dim, sketch_dim) scaled by 1/sqrt(sketch_dim).

        This is re-created if embed_dim changes or it hasn't been built yet.
        The matrix is registered as a buffer (no gradients, moves with device).
        """
        if self._sketch_matrix is None or self._sketch_embed_dim != embed_dim:
            k = min(self.sketch_dim, embed_dim)
            R = torch.randn(embed_dim, k, device=device) / math.sqrt(k)
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
        Uses MSE (Mean Squared Error) — the average squared difference.

        Args:
            predicted: What the predictor thinks the target embeddings are
            target: What the target embeddings actually are (from the encoder)
        """
        # Normalize both to unit length (makes comparison direction-based, not magnitude-based)
        predicted = F.normalize(predicted, dim=-1)
        target = F.normalize(target, dim=-1)

        # MSE loss
        loss = F.mse_loss(predicted, target)
        return loss

    def variance_loss(
        self,
        embeddings: torch.Tensor,
    ) -> torch.Tensor:
        """
        Encourage each dimension to have standard deviation close to
        ``variance_target``.  This is the explicit anti-collapse term:
        if any dimension's std drops to zero, this loss spikes.

        Uses a hinge-style formulation: max(0, target - std(x_d)) so that
        we only penalise dimensions whose std is *below* the target.
        """
        batch_size, num_tokens, embed_dim = embeddings.shape
        flat = embeddings.reshape(-1, embed_dim).float()

        # Per-dimension std
        std = flat.std(dim=0)  # (embed_dim,)

        # Hinge: penalise only when std < target
        var_loss = torch.mean(F.relu(self.variance_target - std))
        return var_loss

    def covariance_loss_sketched(
        self,
        embeddings: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute the off-diagonal covariance loss in a *sketched* space.

        1. Center the embeddings.
        2. Project to a lower-dimensional space via random matrix R.
        3. Compute covariance of the projected embeddings.
        4. Penalise off-diagonal elements (push toward zero → decorrelation).

        Using sketching reduces complexity from O(d^2) to O(d*k + k^2)
        where k = sketch_dim << d.
        """
        batch_size, num_tokens, embed_dim = embeddings.shape
        flat = embeddings.reshape(-1, embed_dim).float()

        # Center
        flat = flat - flat.mean(dim=0, keepdim=True)
        n = flat.shape[0]

        if self.sketch_dim > 0 and self.sketch_dim < embed_dim:
            # --- Sketched covariance ---
            R = self._get_sketch_matrix(embed_dim, embeddings.device)
            projected = flat @ R  # (n, sketch_dim)
            k = projected.shape[1]
            cov = (projected.T @ projected) / max(n - 1, 1)  # (k, k)
        else:
            # --- Full covariance (legacy / small embed_dim) ---
            k = embed_dim
            cov = (flat.T @ flat) / max(n - 1, 1)  # (d, d)

        # Off-diagonal penalty: zero out diagonal, penalise the rest
        diag = torch.diagonal(cov)
        off_diag = cov - torch.diag(diag)
        cov_loss = (off_diag ** 2).sum() / k
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
        reg_loss = self.lambda_var * var_loss + self.lambda_cov * cov_loss

        total_loss = pred_loss + self.lambda_reg * reg_loss

        # Clamp to prevent inf/nan from propagating and crashing training
        if not torch.isfinite(total_loss):
            total_loss = pred_loss.clamp(max=10.0) + self.lambda_reg * reg_loss.clamp(max=10.0)

        return {
            "total_loss": total_loss,
            "prediction_loss": pred_loss,
            "regularization_loss": reg_loss,
            "variance_loss": var_loss,
            "covariance_loss": cov_loss,
        }
