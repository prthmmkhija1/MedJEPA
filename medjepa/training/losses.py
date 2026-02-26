"""
Loss functions for MedJEPA training.

The "loss" is a number that tells the model how wrong it is.
Lower loss = better predictions. The model tries to minimize this number.

SIGReg = Sketched Isotropic Gaussian Regularization
This is the KEY innovation of LeJEPA. It prevents "collapse" — when the model
cheats by making all embeddings identical (which would be useless).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SIGRegLoss(nn.Module):
    """
    Sketched Isotropic Gaussian Regularization loss.

    Has two parts:
    1. PREDICTION LOSS: The predicted target embeddings should match
       the actual target embeddings (MSE — Mean Squared Error).

    2. REGULARIZATION: The embeddings should be spread out (not all the same).
       Specifically, the distribution of embeddings should look like a
       "nice" Gaussian (bell curve) — spread evenly in all directions.

    The single trade-off hyperparameter (lambda_reg) balances these two goals.
    """

    def __init__(
        self,
        lambda_reg: float = 1.0,
        # The ONE hyperparameter: how much to emphasize regularization
        # vs prediction accuracy. Start with 1.0.
    ):
        super().__init__()
        self.lambda_reg = lambda_reg

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

    def regularization_loss(
        self,
        embeddings: torch.Tensor,
    ) -> torch.Tensor:
        """
        Prevent collapse: make sure embeddings are diverse and spread out.

        We want the covariance matrix of embeddings to look like an identity matrix.
        (Identity matrix = each dimension is independent and equally important.)

        If all embeddings were the same, the covariance would be all zeros.
        By pushing it toward identity, we force diversity.
        """
        batch_size, num_tokens, embed_dim = embeddings.shape

        # Reshape: combine batch and token dimensions
        flat = embeddings.reshape(-1, embed_dim)

        # Force float32 for numerical stability (float16 overflows at ~65504
        # and the covariance matrix multiply easily exceeds that with large embed_dim)
        flat = flat.float()

        # Center the embeddings (subtract mean)
        flat = flat - flat.mean(dim=0, keepdim=True)

        # Compute covariance matrix
        # (what correlations exist between embedding dimensions?)
        n = flat.shape[0]
        cov = (flat.T @ flat) / max(n - 1, 1)

        # We want cov to be close to the identity matrix
        # Loss = how far is cov from identity?
        identity = torch.eye(embed_dim, device=embeddings.device)
        reg_loss = F.mse_loss(cov, identity)

        return reg_loss

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
        reg_loss = self.regularization_loss(all_embeddings)

        total_loss = pred_loss + self.lambda_reg * reg_loss

        # Clamp to prevent inf/nan from propagating and crashing training
        if not torch.isfinite(total_loss):
            total_loss = pred_loss.clamp(max=10.0) + self.lambda_reg * reg_loss.clamp(max=10.0)

        return {
            "total_loss": total_loss,
            "prediction_loss": pred_loss,
            "regularization_loss": reg_loss,
        }
