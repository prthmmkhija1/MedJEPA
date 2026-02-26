"""
The training loop: where the model actually LEARNS.

This is like a gym workout routine:
1. Show the model a batch of images (exercise)
2. The model makes predictions (attempt)
3. Compute how wrong it was (feedback)
4. Adjust the model weights to be less wrong next time (improve)
5. Repeat thousands of times
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path
import time
import json
from typing import Optional
from medjepa.utils.device import get_device


class MedJEPATrainer:
    """
    Handles the full training process.
    """

    def __init__(
        self,
        model: nn.Module,
        train_dataset,
        config: dict,
        val_dataset=None,
    ):
        """
        Args:
            model: The LeJEPA or VJEPA model
            train_dataset: Training data (PyTorch Dataset)
            config: Configuration dictionary
            val_dataset: Optional validation data
        """
        self.model = model
        self.config = config
        self.device = get_device()
        self.model = self.model.to(self.device)

        # DataLoader: feeds batches of images to the model
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=config.get("batch_size", 32),
            shuffle=True,       # Random order each epoch
            num_workers=config.get("num_workers", 0),
            # Use 0 workers on HP-Lite (Windows), 4 on Linux/Mac
            pin_memory=torch.cuda.is_available(),
            drop_last=True,     # Drop incomplete last batch
        )

        if val_dataset:
            self.val_loader = DataLoader(
                val_dataset,
                batch_size=config.get("batch_size", 32),
                shuffle=False,
                num_workers=config.get("num_workers", 0),
            )
        else:
            self.val_loader = None

        # Optimizer: the algorithm that updates model weights
        # AdamW is the standard choice for Transformers
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.get("learning_rate", 1e-3),
            weight_decay=config.get("weight_decay", 0.05),
            betas=(0.9, 0.999),
        )

        # Learning rate scheduler:
        # Cosine annealing = start with high lr, gradually decrease
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=config.get("num_epochs", 100),
            eta_min=1e-6,
        )

        # Mixed precision training (saves GPU memory)
        self.use_amp = config.get("mixed_precision", False) and torch.cuda.is_available()
        self.scaler = torch.amp.GradScaler("cuda") if self.use_amp else None

        # Tracking
        self.checkpoint_dir = Path(config.get("checkpoint_dir", "checkpoints"))
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.history = {"train_loss": [], "val_loss": [], "epochs": []}

    def train_one_epoch(self, epoch: int) -> float:
        """
        Train for one complete pass through the dataset.

        Returns:
            Average loss for this epoch
        """
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        start_time = time.time()

        for batch_idx, batch in enumerate(self.train_loader):
            # Handle both (images,) and (images, labels) formats
            if isinstance(batch, (list, tuple)):
                images = batch[0]
            else:
                images = batch

            images = images.to(self.device)

            # Zero gradients BEFORE forward pass
            self.optimizer.zero_grad()

            # Forward pass (with optional mixed precision)
            if self.use_amp:
                with torch.amp.autocast("cuda"):
                    losses = self.model(images)
                    loss = losses["total_loss"]

                # Backward pass with scaling
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                losses = self.model(images)
                loss = losses["total_loss"]

                # Backward pass
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()

            total_loss += loss.item()
            num_batches += 1

            # Print progress
            log_every = self.config.get("log_every", 50)
            if (batch_idx + 1) % log_every == 0:
                elapsed = time.time() - start_time
                avg_loss = total_loss / num_batches
                print(
                    f"  Epoch {epoch+1} | "
                    f"Batch {batch_idx+1}/{len(self.train_loader)} | "
                    f"Loss: {avg_loss:.4f} | "
                    f"Pred Loss: {losses['prediction_loss'].item():.4f} | "
                    f"Reg Loss: {losses['regularization_loss'].item():.4f} | "
                    f"Time: {elapsed:.1f}s"
                )

        # Step the learning rate scheduler
        self.scheduler.step()

        avg_loss = total_loss / max(num_batches, 1)
        return avg_loss

    def train(self) -> dict:
        """
        Full training loop: run for all epochs.
        """
        num_epochs = self.config.get("num_epochs", 100)
        save_every = self.config.get("save_every", 5)
        best_loss = float("inf")

        print("=" * 60)
        print(f"Starting MedJEPA Training")
        print(f"Device: {self.device}")
        print(f"Epochs: {num_epochs}")
        print(f"Batch size: {self.config.get('batch_size', 32)}")
        print(f"Training samples: {len(self.train_loader.dataset)}")
        print("=" * 60)

        for epoch in range(num_epochs):
            epoch_start = time.time()

            # Train
            train_loss = self.train_one_epoch(epoch)
            self.history["train_loss"].append(train_loss)
            self.history["epochs"].append(epoch)

            epoch_time = time.time() - epoch_start
            lr = self.optimizer.param_groups[0]["lr"]

            print(
                f"\nEpoch {epoch+1}/{num_epochs} completed | "
                f"Train Loss: {train_loss:.4f} | "
                f"LR: {lr:.6f} | "
                f"Time: {epoch_time:.1f}s\n"
            )

            # Save checkpoint
            if (epoch + 1) % save_every == 0:
                self.save_checkpoint(epoch, train_loss)

            # Save best model
            if train_loss < best_loss:
                best_loss = train_loss
                self.save_checkpoint(epoch, train_loss, is_best=True)

        print("=" * 60)
        print(f"Training complete! Best loss: {best_loss:.4f}")
        print("=" * 60)

        return self.history

    def save_checkpoint(
        self,
        epoch: int,
        loss: float,
        is_best: bool = False,
    ):
        """Save the model so we can resume later or use it for evaluation."""
        filename = "best_model.pt" if is_best else f"checkpoint_epoch_{epoch+1}.pt"
        path = self.checkpoint_dir / filename

        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "loss": loss,
            "config": self.config,
        }

        torch.save(checkpoint, path)
        print(f"  Saved checkpoint: {path}")

    def load_checkpoint(self, path: str):
        """Load a saved model to resume training or for evaluation."""
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if "scheduler_state_dict" in checkpoint:
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        print(f"Loaded checkpoint from epoch {checkpoint['epoch']+1}")
        return checkpoint["epoch"]
