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
from torch.utils.data import DataLoader, WeightedRandomSampler
from pathlib import Path
import time
import json
import numpy as np
from typing import Optional
from medjepa.utils.device import get_device
try:
    from tqdm import tqdm
except ImportError:
    tqdm = None
try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    SummaryWriter = None


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
        sampler=None,
    ):
        """
        Args:
            model: The LeJEPA or VJEPA model
            train_dataset: Training data (PyTorch Dataset)
            config: Configuration dictionary
            val_dataset: Optional validation data
            sampler: Optional pre-built WeightedRandomSampler (overrides _build_sampler)
        """
        self.model = model
        self.config = config
        self.device = get_device()
        self.model = self.model.to(self.device)

        # Use provided sampler, or build one from labels if available
        if sampler is None:
            sampler = self._build_sampler(train_dataset, config)

        # DataLoader: feeds batches of images to the model
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=config.get("batch_size", 32),
            shuffle=(sampler is None),  # shuffle only when no sampler
            sampler=sampler,
            num_workers=config.get("num_workers", 0),
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
        # Cosine annealing *with linear warmup*.
        # Warmup ramps the LR up from ~0 to peak over the first N epochs,
        # then cosine-decays for the rest.  This stabilises early training.
        self.warmup_epochs = config.get("warmup_epochs", 10)
        num_epochs = config.get("num_epochs", 100)
        cosine_epochs = max(num_epochs - self.warmup_epochs, 1)

        self.cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=cosine_epochs,
            eta_min=1e-6,
        )
        self.warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
            self.optimizer,
            start_factor=1e-4,
            end_factor=1.0,
            total_iters=max(self.warmup_epochs, 1),
        )
        self.scheduler = torch.optim.lr_scheduler.SequentialLR(
            self.optimizer,
            schedulers=[self.warmup_scheduler, self.cosine_scheduler],
            milestones=[self.warmup_epochs],
        )

        # Mixed precision training (saves GPU memory)
        self.use_amp = config.get("mixed_precision", False) and torch.cuda.is_available()
        self.scaler = torch.amp.GradScaler("cuda") if self.use_amp else None

        # Tracking
        self.checkpoint_dir = Path(config.get("checkpoint_dir", "checkpoints"))
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.history = {"train_loss": [], "val_loss": [], "epochs": []}

        # TensorBoard logging (optional)
        self.tb_writer = None
        if config.get("use_tensorboard", True) and SummaryWriter is not None:
            log_dir = Path(config.get("tensorboard_dir", "runs")) / "medjepa"
            self.tb_writer = SummaryWriter(log_dir=str(log_dir))
            print(f"  TensorBoard logging -> {log_dir}")

        self._global_step = 0

    # ------------------------------------------------------------------
    # WeightedRandomSampler: handles class-imbalanced medical datasets
    # ------------------------------------------------------------------
    @staticmethod
    def _build_sampler(dataset, config):
        """Build a WeightedRandomSampler if the dataset has labels and
        ``use_weighted_sampler`` is not explicitly disabled."""
        if not config.get("use_weighted_sampler", True):
            return None

        # Try to obtain integer labels from the dataset (or its inner dataset)
        labels = None
        for obj in (dataset, getattr(dataset, "dataset", None)):
            if obj is None:
                continue
            if hasattr(obj, "labels") and obj.labels is not None:
                raw = obj.labels
                # Only 1-D integer labels work with the sampler
                arr = np.asarray(raw)
                if arr.ndim == 1 and np.issubdtype(arr.dtype, np.integer):
                    labels = arr
                    break
                # float labels with integer values (common in CSV-loaded data)
                if arr.ndim == 1 and np.issubdtype(arr.dtype, np.floating):
                    if np.all(arr == arr.astype(int)):
                        labels = arr.astype(int)
                        break

        if labels is None:
            return None  # unlabelled / multi-label → just shuffle

        # Compute inverse-frequency sample weights
        classes, counts = np.unique(labels, return_counts=True)
        class_weight = {c: 1.0 / cnt for c, cnt in zip(classes, counts)}
        sample_weights = np.array([class_weight[int(l)] for l in labels],
                                  dtype=np.float64)

        sampler = WeightedRandomSampler(
            weights=torch.from_numpy(sample_weights),
            num_samples=len(sample_weights),
            replacement=True,
        )
        print(f"  WeightedRandomSampler enabled ({len(classes)} classes, "
              f"min count={int(counts.min())}, max count={int(counts.max())})")
        return sampler

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

        loader = self.train_loader
        if tqdm is not None:
            loader = tqdm(
                self.train_loader,
                desc=f"Epoch {epoch+1}/{self.config.get('num_epochs',100)}",
                unit="batch",
                leave=True,
                dynamic_ncols=True,
            )

        for batch_idx, batch in enumerate(loader):
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

            # Skip inf/nan batches in accumulation (numerical stability)
            loss_val = loss.item()
            if not (loss_val != loss_val or loss_val == float('inf')):
                total_loss += loss_val
                num_batches += 1

            self._global_step += 1

            # TensorBoard per-step logging
            if self.tb_writer is not None and num_batches > 0:
                self.tb_writer.add_scalar(
                    "train/loss_step", loss_val, self._global_step)
                self.tb_writer.add_scalar(
                    "train/pred_loss_step",
                    losses["prediction_loss"].item(), self._global_step)
                self.tb_writer.add_scalar(
                    "train/reg_loss_step",
                    losses["regularization_loss"].item(), self._global_step)

            # Update tqdm postfix or print progress
            avg_loss = total_loss / max(num_batches, 1)
            if tqdm is not None and hasattr(loader, 'set_postfix'):
                loader.set_postfix(
                    loss=f"{avg_loss:.4f}",
                    pred=f"{losses['prediction_loss'].item():.4f}",
                    reg=f"{losses['regularization_loss'].item():.4f}",
                )
            else:
                log_every = self.config.get("log_every", 50)
                if (batch_idx + 1) % log_every == 0:
                    elapsed = time.time() - start_time
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

            # TensorBoard epoch-level logging
            if self.tb_writer is not None:
                self.tb_writer.add_scalar("train/loss_epoch", train_loss, epoch)
                self.tb_writer.add_scalar("train/lr", lr, epoch)
                self.tb_writer.add_scalar("train/epoch_time_s", epoch_time, epoch)

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

        if self.tb_writer is not None:
            self.tb_writer.close()

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
