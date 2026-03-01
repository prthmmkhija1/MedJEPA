"""
The training loop: where the model actually LEARNS.

This is like a gym workout routine:
1. Show the model a batch of images (exercise)
2. The model makes predictions (attempt)
3. Compute how wrong it was (feedback)
4. Adjust the model weights to be less wrong next time (improve)
5. Repeat thousands of times

Supports both single-GPU and multi-GPU (DDP) training.
"""

import warnings
import torch
import torch.nn as nn
import torch.distributed as dist
import math
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, WeightedRandomSampler, DistributedSampler
from pathlib import Path
import time
import json
import os
import numpy as np
from typing import Optional
from medjepa.utils.device import get_device

# OneCycleLR internally calls step() during __init__, triggering a false-positive
# "scheduler before optimizer" warning in PyTorch.  Suppress it.
warnings.filterwarnings(
    "ignore",
    message=".*lr_scheduler.step\\(\\) before optimizer.step\\(\\).*",
    category=UserWarning,
)
try:
    from tqdm import tqdm
except ImportError:
    tqdm = None
try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    SummaryWriter = None


# ----------------------------------------------------------------
# CUDA Stream Prefetcher: overlap CPU→GPU transfer with compute
# ----------------------------------------------------------------

class CUDAPrefetcher:
    """Prefetch the next batch onto the GPU using a side CUDA stream.

    While the current batch is being processed on the default stream,
    the *next* batch is asynchronously copied to the GPU on a separate
    stream.  This hides most of the CPU→GPU transfer latency.
    """

    def __init__(self, loader, device: torch.device):
        self.loader = loader
        self.device = device
        self.stream = torch.cuda.Stream(device=device)
        self._base_iter = None
        self._next_batch = None

    def __iter__(self):
        self._base_iter = iter(self.loader)
        self._preload()
        return self

    def _preload(self):
        try:
            batch = next(self._base_iter)
        except StopIteration:
            self._next_batch = None
            return
        with torch.cuda.stream(self.stream):
            if isinstance(batch, (list, tuple)):
                self._next_batch = type(batch)(
                    t.to(self.device, non_blocking=True) if torch.is_tensor(t) else t
                    for t in batch
                )
            elif torch.is_tensor(batch):
                self._next_batch = batch.to(self.device, non_blocking=True)
            else:
                self._next_batch = batch

    def __next__(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        if self._next_batch is None:
            raise StopIteration
        batch = self._next_batch
        # Ensure tensors record the dependency on the prefetch stream
        if torch.is_tensor(batch):
            batch.record_stream(torch.cuda.current_stream())
        elif isinstance(batch, (list, tuple)):
            for t in batch:
                if torch.is_tensor(t):
                    t.record_stream(torch.cuda.current_stream())
        self._preload()
        return batch

    def __len__(self):
        return len(self.loader)


# ----------------------------------------------------------------
# Safe collate: handle mixed returns & shape mismatches
# ----------------------------------------------------------------

def _safe_collate(batch):
    """Custom collate that handles shape mismatches and mixed return types.

    - Strips labels from (image, label) tuples so the batch is always
      a plain tensor of images.
    - Skips items whose image tensor has a different shape from the majority.
    - Falls back to default_collate for clean batches (zero overhead).
    """
    # Unwrap: ensure every element is a plain tensor
    images = []
    for item in batch:
        if isinstance(item, (tuple, list)):
            images.append(item[0])
        else:
            images.append(item)

    if not images:
        return torch.empty(0)

    # Determine the expected shape (majority vote)
    target_shape = images[0].shape
    filtered = [img for img in images if img.shape == target_shape]

    if len(filtered) < len(images):
        # Some images had wrong shapes — drop them
        dropped = len(images) - len(filtered)
        if dropped > 0:
            import logging
            logging.getLogger(__name__).warning(
                f"Dropped {dropped}/{len(images)} items with mismatched shapes "
                f"(expected {target_shape})"
            )

    if not filtered:
        return torch.zeros(1, *target_shape)

    return torch.stack(filtered, 0)


# ----------------------------------------------------------------
# DDP helper utilities
# ----------------------------------------------------------------

def setup_ddp(rank: int, world_size: int, backend: str = "nccl"):
    """
    Initialize the distributed process group.

    Args:
        rank: Global rank of this process (0 = master).
        world_size: Total number of processes.
        backend: "nccl" (GPU, recommended), "gloo" (CPU fallback).
    """
    os.environ.setdefault("MASTER_ADDR", "localhost")
    os.environ.setdefault("MASTER_PORT", "12355")
    dist.init_process_group(backend, rank=rank, world_size=world_size)
    if torch.cuda.is_available():
        torch.cuda.set_device(rank)


def cleanup_ddp():
    """Destroy the distributed process group."""
    if dist.is_initialized():
        dist.destroy_process_group()


def is_main_process() -> bool:
    """Return True if this is rank 0 or DDP is not active."""
    if not dist.is_initialized():
        return True
    return dist.get_rank() == 0


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
            config: Configuration dictionary.  DDP keys:
                    - ``distributed``: bool — enable DistributedDataParallel
                    - ``local_rank``: int — GPU device index for this process
                    - ``world_size``: int — total number of processes
            val_dataset: Optional validation data
            sampler: Optional pre-built WeightedRandomSampler (overrides _build_sampler).
                     Ignored when ``distributed=True`` (a DistributedSampler is used).
        """
        self.model = model
        self.config = config

        # ---------- DDP setup ----------
        self.distributed = config.get("distributed", False)
        self.local_rank = config.get("local_rank", 0)
        self.world_size = config.get("world_size", 1)

        # ---------- CUDA performance flags ----------
        if torch.cuda.is_available():
            # Auto-tune convolution algorithms for fixed input sizes
            torch.backends.cudnn.benchmark = True
            # TF32 on A100/Ampere: ~2x matmul throughput with negligible precision loss
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

        if self.distributed:
            self.device = torch.device(f"cuda:{self.local_rank}")
            self.model = self.model.to(self.device)
            self.model = DDP(
                self.model,
                device_ids=[self.local_rank],
                output_device=self.local_rank,
                find_unused_parameters=False,
            )
            # DDP requires DistributedSampler — override any provided sampler
            sampler = DistributedSampler(
                train_dataset,
                num_replicas=self.world_size,
                rank=self.local_rank,
                shuffle=True,
            )
        else:
            self.device = get_device()
            self.model = self.model.to(self.device)

            # Use provided sampler, or build one from labels if available
            if sampler is None:
                sampler = self._build_sampler(train_dataset, config)

        # ---------- torch.compile for faster forward/backward ----------
        # Disable CUDA graphs (triton.cudagraphs=False) — they crash on any
        # dynamic tensor inside the graph (random sketch matrix, variable-size
        # mask indices, etc.).  All Triton kernel auto-tuning still applies.
        if config.get("compile_model", True) and hasattr(torch, "compile"):
            try:
                self.model = torch.compile(
                    self.model,
                    mode="max-autotune",
                    options={"triton.cudagraphs": False},
                )
                print("  torch.compile enabled (max-autotune, no cudagraphs)")
            except Exception as e:
                print(f"  torch.compile unavailable: {e}")

        # DataLoader: feeds batches of images to the model
        _nw = config.get("num_workers", 0)
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=config.get("batch_size", 32),
            shuffle=(sampler is None),  # shuffle only when no sampler
            sampler=sampler,
            num_workers=_nw,
            pin_memory=torch.cuda.is_available(),
            drop_last=True,     # Drop incomplete last batch
            persistent_workers=(_nw > 0),   # keep workers alive between epochs
            prefetch_factor=3 if _nw > 0 else None,  # pre-load next batches
            collate_fn=_safe_collate,  # handle shape mismatches & mixed returns
        )

        if val_dataset:
            self.val_loader = DataLoader(
                val_dataset,
                batch_size=config.get("batch_size", 32),
                shuffle=False,
                num_workers=_nw,
                pin_memory=torch.cuda.is_available(),
                persistent_workers=(_nw > 0),
                prefetch_factor=3 if _nw > 0 else None,
                collate_fn=_safe_collate,
            )
        else:
            self.val_loader = None

        # Optimizer: the algorithm that updates model weights
        # Fused AdamW merges multiple CUDA kernels → fewer launches → faster
        _use_fused = torch.cuda.is_available() and hasattr(torch.optim.AdamW, 'fused')
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.get("learning_rate", 1e-3),
            weight_decay=config.get("weight_decay", 0.05),
            betas=(0.9, 0.999),
            fused=_use_fused,
        )

        # Learning rate scheduler:
        # OneCycleLR: peaks early and decays — 2-3x faster convergence than
        # warmup+cosine for self-supervised ViT training.
        # Falls back to warmup+cosine if total_steps can't be computed.
        self.warmup_epochs = config.get("warmup_epochs", 10)
        num_epochs = config.get("num_epochs", 100)
        _use_onecycle = config.get("use_onecycle_lr", True)

        try:
            if _use_onecycle and hasattr(train_dataset, '__len__'):
                _bs = config.get("batch_size", 32)
                _accum = config.get("gradient_accumulation_steps", 1)
                steps_per_epoch = max(len(train_dataset) // (_bs * _accum), 1)
                total_steps = num_epochs * steps_per_epoch
                self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
                    self.optimizer,
                    max_lr=config.get("learning_rate", 1e-3),
                    total_steps=total_steps,
                    pct_start=self.warmup_epochs / max(num_epochs, 1),
                    anneal_strategy="cos",
                    div_factor=25.0,      # start_lr = max_lr / 25
                    final_div_factor=1e4, # end_lr  = max_lr / 1e4
                )
                self._scheduler_step_per_batch = True
                print(f"  OneCycleLR: {total_steps} total steps, "
                      f"max_lr={config.get('learning_rate', 1e-3):.4f}")
            else:
                raise ValueError("fallback")
        except Exception:
            # Fallback: warmup + cosine
            cosine_epochs = max(num_epochs - self.warmup_epochs, 1)
            self.cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=cosine_epochs, eta_min=1e-6)
            self.warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
                self.optimizer, start_factor=1e-4, end_factor=1.0,
                total_iters=max(self.warmup_epochs, 1))
            self.scheduler = torch.optim.lr_scheduler.SequentialLR(
                self.optimizer,
                schedulers=[self.warmup_scheduler, self.cosine_scheduler],
                milestones=[self.warmup_epochs])
            self._scheduler_step_per_batch = False

        # Mixed precision training (saves GPU memory and boosts throughput)
        # Prefer BFloat16 on Ampere+ GPUs: wider dynamic range → no GradScaler needed
        self.use_amp = config.get("mixed_precision", False) and torch.cuda.is_available()
        self._amp_dtype = torch.float16  # default
        if self.use_amp and torch.cuda.is_bf16_supported():
            self._amp_dtype = torch.bfloat16
        # GradScaler is only needed for float16 (bf16 has enough dynamic range)
        self.scaler = (
            torch.amp.GradScaler("cuda")
            if self.use_amp and self._amp_dtype == torch.float16
            else None
        )
        if self.use_amp:
            print(f"  Mixed precision: {self._amp_dtype}  "
                  f"(GradScaler={'ON' if self.scaler else 'OFF'})")

        # CUDA stream prefetcher: overlap data transfer with GPU compute
        self._use_prefetcher = (
            torch.cuda.is_available()
            and config.get("use_prefetcher", True)
        )

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
        self._grad_accum_steps = config.get("gradient_accumulation_steps", 1)
        self._tb_log_every = config.get("tb_log_every", 50)  # reduce GPU syncs

        # EMA momentum schedule: cosine ramp from ema_start → 1.0 over training.
        # Higher momentum later means the target encoder changes more slowly as
        # training progresses, producing increasingly stable targets.
        self._ema_start = config.get("ema_momentum", 0.996)
        self._ema_end = 1.0
        self._total_ema_steps = (
            config.get("num_epochs", 100)
            * max(len(train_dataset) // max(config.get("batch_size", 32), 1), 1)
        ) if hasattr(train_dataset, '__len__') else 100_000

    def _update_ema(self):
        """Update EMA target encoder with cosine momentum schedule."""
        model = self.model
        # Unwrap DDP / compiled model
        if hasattr(model, 'module'):
            model = model.module
        if hasattr(model, '_orig_mod'):
            model = model._orig_mod
        if not hasattr(model, 'update_ema'):
            return
        # Cosine momentum schedule: m(t) = 1 - (1 - m_start) * (cos(π*t/T) + 1) / 2
        # Ramps from m_start → m_end over training
        t = min(self._global_step / max(self._total_ema_steps, 1), 1.0)
        momentum = self._ema_end - (self._ema_end - self._ema_start) * (math.cos(math.pi * t) + 1) / 2
        model.ema_momentum = momentum
        model.update_ema()

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

        # Wrap data loader with CUDA prefetcher (overlaps transfer + compute)
        base_loader = self.train_loader
        if self._use_prefetcher:
            base_loader = CUDAPrefetcher(self.train_loader, self.device)

        loader = base_loader
        if tqdm is not None:
            loader = tqdm(
                base_loader,
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

            # Move to device (no-op if prefetcher already placed on GPU)
            images = images.to(self.device, non_blocking=True)

            # Signal start of a new step — needed when CUDA graphs are active
            # (no-op when cudagraphs disabled, but harmless to keep)
            if hasattr(torch, "compiler") and hasattr(torch.compiler, "cudagraph_mark_step_begin"):
                torch.compiler.cudagraph_mark_step_begin()

            # Zero gradients (set_to_none is faster than filling with zeros)
            if batch_idx % self._grad_accum_steps == 0:
                self.optimizer.zero_grad(set_to_none=True)

            # Forward pass (with optional mixed precision)
            if self.use_amp:
                with torch.amp.autocast("cuda", dtype=self._amp_dtype):
                    losses = self.model(images)
                    loss = losses["total_loss"] / self._grad_accum_steps

                if self.scaler is not None:
                    # FP16 path: use GradScaler
                    self.scaler.scale(loss).backward()
                    if (batch_idx + 1) % self._grad_accum_steps == 0:
                        self.scaler.unscale_(self.optimizer)
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                        self._update_ema()
                else:
                    # BF16 path: no scaler needed (wider dynamic range)
                    loss.backward()
                    if (batch_idx + 1) % self._grad_accum_steps == 0:
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                        self.optimizer.step()
                        self._update_ema()
            else:
                losses = self.model(images)
                loss = losses["total_loss"] / self._grad_accum_steps

                # Backward pass
                loss.backward()
                if (batch_idx + 1) % self._grad_accum_steps == 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    self.optimizer.step()
                    self._update_ema()

            # Track loss (avoid .item() sync on every step when possible)
            with torch.no_grad():
                loss_val = (loss * self._grad_accum_steps).detach()
            if torch.isfinite(loss_val):
                total_loss += loss_val.item()
                num_batches += 1

            self._global_step += 1

            # OneCycleLR steps per batch; epoch-level schedulers step at epoch end
            # Only step AFTER the first optimizer.step() has happened
            if getattr(self, '_scheduler_step_per_batch', False):
                if (batch_idx + 1) % self._grad_accum_steps == 0 and self._global_step > 1:
                    self.scheduler.step()

            # TensorBoard per-step logging (throttled to reduce GPU sync overhead)
            if (self.tb_writer is not None and num_batches > 0
                    and self._global_step % self._tb_log_every == 0):
                lv = loss_val.item() if torch.is_tensor(loss_val) else loss_val
                self.tb_writer.add_scalar(
                    "train/loss_step", lv, self._global_step)
                self.tb_writer.add_scalar(
                    "train/pred_loss_step",
                    losses["prediction_loss"].item(), self._global_step)
                self.tb_writer.add_scalar(
                    "train/reg_loss_step",
                    losses["regularization_loss"].item(), self._global_step)

            # Update tqdm postfix or print progress (throttled to avoid GPU sync)
            log_every = self.config.get("log_every", 50)
            if (batch_idx + 1) % log_every == 0:
                avg_loss = total_loss / max(num_batches, 1)
                pred_val = losses["prediction_loss"].item()
                reg_val = losses["regularization_loss"].item()
                if tqdm is not None and hasattr(loader, 'set_postfix'):
                    loader.set_postfix(
                        loss=f"{avg_loss:.4f}",
                        pred=f"{pred_val:.4f}",
                        reg=f"{reg_val:.4f}",
                    )
                else:
                    elapsed = time.time() - start_time
                    print(
                        f"  Epoch {epoch+1} | "
                        f"Batch {batch_idx+1}/{len(self.train_loader)} | "
                        f"Loss: {avg_loss:.4f} | "
                        f"Pred Loss: {pred_val:.4f} | "
                        f"Reg Loss: {reg_val:.4f} | "
                        f"Time: {elapsed:.1f}s"
                    )

        # Step epoch-level scheduler (skip for OneCycleLR which steps per batch)
        if not getattr(self, '_scheduler_step_per_batch', False):
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

        if is_main_process():
            print("=" * 60)
            print(f"Starting MedJEPA Training")
            print(f"Device: {self.device}")
            if self.distributed:
                print(f"DDP: rank {self.local_rank} / world_size {self.world_size}")
            print(f"Epochs: {num_epochs}")
            print(f"Batch size: {self.config.get('batch_size', 32)}")
            print(f"Training samples: {len(self.train_loader.dataset)}")
            print("=" * 60)

        for epoch in range(num_epochs):
            epoch_start = time.time()

            # DDP: set epoch so shuffling differs each epoch
            if self.distributed and hasattr(self.train_loader.sampler, 'set_epoch'):
                self.train_loader.sampler.set_epoch(epoch)

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

            if is_main_process():
                print(
                    f"\nEpoch {epoch+1}/{num_epochs} completed | "
                    f"Train Loss: {train_loss:.4f} | "
                    f"LR: {lr:.6f} | "
                    f"Time: {epoch_time:.1f}s\n"
                )

            # Save checkpoint (only on main process)
            if is_main_process():
                if (epoch + 1) % save_every == 0:
                    self.save_checkpoint(epoch, train_loss)

                if train_loss < best_loss:
                    best_loss = train_loss
                    self.save_checkpoint(epoch, train_loss, is_best=True)

        if is_main_process():
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

        # Unwrap DDP module if present
        model_to_save = self.model.module if hasattr(self.model, "module") else self.model
        # Unwrap torch.compile wrapper if present
        if hasattr(model_to_save, '_orig_mod'):
            model_to_save = model_to_save._orig_mod

        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model_to_save.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "loss": loss,
            "config": self.config,
        }
        # Save EMA encoder state if present
        if hasattr(model_to_save, 'ema_encoder') and model_to_save.ema_encoder is not None:
            checkpoint["ema_encoder_state_dict"] = model_to_save.ema_encoder.state_dict()

        torch.save(checkpoint, path)
        print(f"  Saved checkpoint: {path}")

    def load_checkpoint(self, path: str):
        """Load a saved model to resume training or for evaluation."""
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        # Unwrap compiled model if present
        model = self.model
        if hasattr(model, '_orig_mod'):
            model = model._orig_mod
        model.load_state_dict(checkpoint["model_state_dict"])
        # Restore EMA encoder state if present
        if "ema_encoder_state_dict" in checkpoint:
            inner = model.module if hasattr(model, 'module') else model
            if hasattr(inner, 'ema_encoder') and inner.ema_encoder is not None:
                inner.ema_encoder.load_state_dict(checkpoint["ema_encoder_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if "scheduler_state_dict" in checkpoint:
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        print(f"Loaded checkpoint from epoch {checkpoint['epoch']+1}")
        return checkpoint["epoch"]
