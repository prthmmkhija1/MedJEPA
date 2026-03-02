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
        # All images had different shapes — return a 1-sample placeholder.
        # The training loop will see a batch of size 1 which is harmless
        # (single-sample gradients are noisy but finite).
        import logging
        logging.getLogger(__name__).warning(
            f"All {len(images)} items had mismatched shapes; returning 1-sample placeholder."
        )
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
        # On Windows, Triton is NOT available, so max-autotune will fail.
        # Use mode="default" instead (still benefits from torch.compile graph
        # optimizations, just skips Triton kernel tuning).
        import platform as _platform
        _is_windows = _platform.system() == "Windows"
        _compile_mode = "default" if _is_windows else "max-autotune"
        _compile_options = {} if _is_windows else {"triton.cudagraphs": False}
        if config.get("compile_model", True) and hasattr(torch, "compile"):
            try:
                self.model = torch.compile(
                    self.model,
                    mode=_compile_mode,
                    options=_compile_options,
                )
                print(f"  torch.compile enabled (mode={_compile_mode})")
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
        # Fused AdamW merges multiple CUDA kernels → fewer launches → faster.
        # Guard: some CUDA builds lack fused support; fall back gracefully.
        _use_fused = False
        if torch.cuda.is_available():
            try:
                # Test-construct a tiny fused optimizer to see if it works
                _test_p = torch.nn.Linear(1, 1, device='cuda').parameters()
                torch.optim.AdamW(_test_p, fused=True)
                _use_fused = True
            except Exception:
                _use_fused = False
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
                    final_div_factor=config.get("final_div_factor", 100),  # end_lr = max_lr / 100 → 3e-6
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
        self._grad_clip_max_norm = config.get("grad_clip_max_norm", 1.0)

        # EMA momentum schedule: cosine ramp from ema_start → 1.0 over training.
        # Higher momentum later means the target encoder changes more slowly as
        # training progresses, producing increasingly stable targets.
        self._ema_start = config.get("ema_momentum", 0.996)
        self._ema_end = config.get("ema_momentum_end", 1.0)
        self._total_ema_steps = (
            config.get("num_epochs", 100)
            * max(len(train_dataset) // max(config.get("batch_size", 32), 1), 1)
        ) if hasattr(train_dataset, '__len__') else 100_000

        # Health tracking: detect training stagnation
        self._epoch_pred_losses = []  # track pred_loss per epoch
        self._epoch_reg_losses = []   # track reg_loss per epoch

        # Early stopping: stop training when loss stops improving.
        # Saves potentially 30-50% of total wall time by avoiding
        # unnecessary epochs after convergence.
        self._es_patience = config.get("early_stopping_patience", 30)
        self._es_min_delta = config.get("early_stopping_min_delta", 1e-4)
        self._es_epochs_no_improve = 0
        self._es_best_loss = float("inf")

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
                        _gnorm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self._grad_clip_max_norm)
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                        self._update_ema()
                        self._last_grad_norm = _gnorm.item()
                else:
                    # BF16 path: no scaler needed (wider dynamic range)
                    loss.backward()
                    if (batch_idx + 1) % self._grad_accum_steps == 0:
                        _gnorm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self._grad_clip_max_norm)
                        self.optimizer.step()
                        self._update_ema()
                        self._last_grad_norm = _gnorm.item()
            else:
                losses = self.model(images)
                loss = losses["total_loss"] / self._grad_accum_steps

                # Backward pass
                loss.backward()
                if (batch_idx + 1) % self._grad_accum_steps == 0:
                    _gnorm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self._grad_clip_max_norm)
                    self.optimizer.step()
                    self._update_ema()
                    self._last_grad_norm = _gnorm.item()

            # Track loss (avoid .item() sync on every step when possible)
            with torch.no_grad():
                loss_val = (loss * self._grad_accum_steps).detach()
            if torch.isfinite(loss_val):
                total_loss += loss_val.item()
                num_batches += 1

            # Save last batch component losses for epoch-level health check
            self._last_batch_losses = {
                k: v.item() if torch.is_tensor(v) else v
                for k, v in losses.items() if k != "total_loss"
            }

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
                if hasattr(self, '_last_grad_norm'):
                    self.tb_writer.add_scalar(
                        "train/grad_norm", self._last_grad_norm, self._global_step)
                # Granular reg breakdown: var_loss and cov_loss separately
                if "variance_loss" in losses:
                    self.tb_writer.add_scalar(
                        "train/var_loss_step",
                        losses["variance_loss"].item(), self._global_step)
                if "covariance_loss" in losses:
                    self.tb_writer.add_scalar(
                        "train/cov_loss_step",
                        losses["covariance_loss"].item(), self._global_step)

            # Update tqdm postfix or print progress (throttled to avoid GPU sync)
            log_every = self.config.get("log_every", 50)
            if (batch_idx + 1) % log_every == 0:
                avg_loss = total_loss / max(num_batches, 1)
                pred_val = losses["prediction_loss"].item()
                reg_val = losses["regularization_loss"].item()
                var_val = losses.get("variance_loss")
                cov_val = losses.get("covariance_loss")
                var_str = f"{var_val.item():.4f}" if var_val is not None else "?"
                cov_str = f"{cov_val.item():.4f}" if cov_val is not None else "?"
                if tqdm is not None and hasattr(loader, 'set_postfix'):
                    loader.set_postfix(
                        loss=f"{avg_loss:.4f}",
                        pred=f"{pred_val:.4f}",
                        reg=f"{reg_val:.4f}",
                        var=var_str,
                        cov=cov_str,
                    )
                else:
                    elapsed = time.time() - start_time
                    print(
                        f"  Epoch {epoch+1} | "
                        f"Batch {batch_idx+1}/{len(self.train_loader)} | "
                        f"Loss: {avg_loss:.4f} | "
                        f"Pred: {pred_val:.4f} | "
                        f"Reg: {reg_val:.4f} "
                        f"(var={var_str}, cov={cov_str}) | "
                        f"Time: {elapsed:.1f}s"
                    )

        # Step epoch-level scheduler (skip for OneCycleLR which steps per batch)
        if not getattr(self, '_scheduler_step_per_batch', False):
            self.scheduler.step()

        # Clean up any leftover accumulated gradients at epoch boundary.
        # Without this, if len(loader) % grad_accum_steps != 0, dirty
        # gradients from the last partial cycle bleed into the next epoch.
        self.optimizer.zero_grad(set_to_none=True)

        avg_loss = total_loss / max(num_batches, 1)
        return avg_loss

    def train(self, resume_checkpoint: str = None) -> dict:
        """
        Full training loop: run for all epochs.
        Supports resuming from a checkpoint via resume_checkpoint path.
        """
        num_epochs = self.config.get("num_epochs", 100)
        save_every = self.config.get("save_every", 5)
        best_loss = float("inf")
        self._start_epoch = 0

        # Auto-resume from checkpoint
        if resume_checkpoint and Path(resume_checkpoint).exists():
            try:
                resumed_epoch = self.load_checkpoint(resume_checkpoint)
                self._start_epoch = resumed_epoch + 1
                if is_main_process():
                    print(f"\n  Resuming training from epoch {self._start_epoch} "
                          f"(loaded {Path(resume_checkpoint).name})")
            except Exception as e:
                if is_main_process():
                    print(f"  Could not resume from checkpoint: {e}. Starting fresh.")
                self._start_epoch = 0

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

        for epoch in range(self._start_epoch, num_epochs):
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
                # Log EMA momentum to track the schedule
                _model = self.model
                if hasattr(_model, '_orig_mod'):
                    _model = _model._orig_mod
                if hasattr(_model, 'module'):
                    _model = _model.module
                if hasattr(_model, 'ema_momentum'):
                    self.tb_writer.add_scalar(
                        "train/ema_momentum", _model.ema_momentum, epoch)
                if hasattr(self, '_last_grad_norm'):
                    self.tb_writer.add_scalar(
                        "train/grad_norm_epoch", self._last_grad_norm, epoch)

            if is_main_process():
                print(
                    f"\nEpoch {epoch+1}/{num_epochs} completed | "
                    f"Train Loss: {train_loss:.4f} | "
                    f"LR: {lr:.6f} | "
                    f"Time: {epoch_time:.1f}s\n"
                )

            # ---- Training health check (stagnation / collapse detection) ----
            # Uses the last batch's component losses as a proxy for epoch state
            if is_main_process() and hasattr(self, '_last_batch_losses'):
                _lb = self._last_batch_losses
                _pred = _lb.get("prediction_loss", 0)
                _reg = _lb.get("regularization_loss", 0)
                _var = _lb.get("variance_loss", 0)
                self._epoch_pred_losses.append(_pred)
                self._epoch_reg_losses.append(_reg)

                # Warn if reg is near 0 AND pred hasn't increased since epoch 1
                if epoch >= 4 and len(self._epoch_pred_losses) >= 5:
                    recent_pred = self._epoch_pred_losses[-3:]
                    recent_reg = self._epoch_reg_losses[-3:]
                    avg_pred = sum(recent_pred) / len(recent_pred)
                    avg_reg = sum(recent_reg) / len(recent_reg)

                    # With batch-wise variance, reg has a softplus floor ≈ 0.35
                    # (never zero even with perfect variance).  Check pred_loss
                    # directly to detect collapse.
                    if avg_pred < 0.005:
                        # Only warn if pred_loss is truly flat/decreasing (not rising)
                        pred_rising = (len(self._epoch_pred_losses) >= 3 and
                                       self._epoch_pred_losses[-1] > self._epoch_pred_losses[-3])
                        if not pred_rising:
                            print(
                                f"  ⚠ WARNING: Potential collapse detected!\n"
                                f"    pred_loss={avg_pred:.6f} (very low).\n"
                                f"    reg_loss={avg_reg:.6f} | If pred doesn't rise "
                                f"in next epochs, the model may be learning trivial "
                                f"features.\n"
                                f"    Consider: increasing lambda_var, checking data "
                                f"diversity, or lowering EMA momentum."
                            )
                        else:
                            print(
                                f"  ✓ Healthy: pred_loss rising ({avg_pred:.6f}), "
                                f"regularization active ({avg_reg:.6f})"
                            )
                    elif avg_pred > 0.05:
                        print(
                            f"  ✓ Healthy: pred driving learning ({avg_pred:.4f}), "
                            f"reg={avg_reg:.4f}"
                        )

                # Print epoch-level component breakdown
                _gnorm_str = ""
                if hasattr(self, '_last_grad_norm'):
                    _gnorm_str = f" | grad_norm={self._last_grad_norm:.2f}"
                print(
                    f"  Components: pred={_pred:.4f} | reg={_reg:.4f} "
                    f"(var={_var:.4f}, cov={_lb.get('covariance_loss', 0):.4f})"
                    f"{_gnorm_str}"
                )

            # Save checkpoint (only on main process)
            if is_main_process():
                if (epoch + 1) % save_every == 0:
                    self.save_checkpoint(epoch, train_loss)

                if train_loss < best_loss:
                    best_loss = train_loss
                    self.save_checkpoint(epoch, train_loss, is_best=True)

            # ---- Early stopping ----
            if train_loss < self._es_best_loss - self._es_min_delta:
                self._es_best_loss = train_loss
                self._es_epochs_no_improve = 0
            else:
                self._es_epochs_no_improve += 1

            if (self._es_patience > 0
                    and self._es_epochs_no_improve >= self._es_patience
                    and epoch >= max(self.warmup_epochs + self._es_patience, 50)):
                if is_main_process():
                    print(
                        f"\n  Early stopping: no improvement for "
                        f"{self._es_patience} epochs (best={self._es_best_loss:.6f}). "
                        f"Stopping at epoch {epoch+1}/{num_epochs}."
                    )
                break

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

        # Unwrap torch.compile wrapper if present (must be first — compile is outermost)
        model_to_save = self.model
        if hasattr(model_to_save, '_orig_mod'):
            model_to_save = model_to_save._orig_mod
        # Unwrap DDP module if present (DDP is inside compile)
        if hasattr(model_to_save, 'module'):
            model_to_save = model_to_save.module

        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model_to_save.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "loss": loss,
            "config": self.config,
            # Training state needed for correct resume
            "global_step": self._global_step,
            "es_best_loss": self._es_best_loss,
            "es_epochs_no_improve": self._es_epochs_no_improve,
            "history": self.history,
            "epoch_pred_losses": self._epoch_pred_losses,
            "epoch_reg_losses": self._epoch_reg_losses,
        }
        # Save EMA encoder state if present
        if hasattr(model_to_save, 'ema_encoder') and model_to_save.ema_encoder is not None:
            checkpoint["ema_encoder_state_dict"] = model_to_save.ema_encoder.state_dict()

        torch.save(checkpoint, path)
        print(f"  Saved checkpoint: {path}")

        # Cleanup: keep only the last 3 epoch checkpoints + best_model.pt
        if not is_best:
            import re as _re
            def _epoch_num(p):
                m = _re.search(r'epoch_(\d+)', p.name)
                return int(m.group(1)) if m else 0
            all_ckpts = sorted(
                self.checkpoint_dir.glob("checkpoint_epoch_*.pt"),
                key=_epoch_num,
            )
            while len(all_ckpts) > 3:
                old = all_ckpts.pop(0)
                try:
                    old.unlink()
                except OSError:
                    pass

    def load_checkpoint(self, path: str):
        """Load a saved model to resume training or for evaluation."""
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)

        # ---------- Unwrap compile + DDP to get the raw model ----------
        # Structure may be: compile(DDP(raw)), DDP(raw), compile(raw), or raw
        model = self.model
        if hasattr(model, '_orig_mod'):       # unwrap torch.compile
            model = model._orig_mod
        if hasattr(model, 'module'):           # unwrap DDP
            model = model.module

        model.load_state_dict(checkpoint["model_state_dict"])

        # Restore EMA encoder state if present
        if "ema_encoder_state_dict" in checkpoint:
            if hasattr(model, 'ema_encoder') and model.ema_encoder is not None:
                model.ema_encoder.load_state_dict(checkpoint["ema_encoder_state_dict"])

        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        if "scheduler_state_dict" in checkpoint:
            try:
                self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
            except Exception as e:
                print(f"  WARNING: Could not restore scheduler state: {e}")
                print("  Scheduler will continue from its initial state.")

        # ---------- Restore training state ----------
        if "global_step" in checkpoint:
            self._global_step = checkpoint["global_step"]
        if "es_best_loss" in checkpoint:
            self._es_best_loss = checkpoint["es_best_loss"]
        if "es_epochs_no_improve" in checkpoint:
            self._es_epochs_no_improve = checkpoint["es_epochs_no_improve"]
        if "history" in checkpoint:
            self.history = checkpoint["history"]
        if "epoch_pred_losses" in checkpoint:
            self._epoch_pred_losses = checkpoint["epoch_pred_losses"]
        if "epoch_reg_losses" in checkpoint:
            self._epoch_reg_losses = checkpoint["epoch_reg_losses"]

        print(f"Loaded checkpoint from epoch {checkpoint['epoch']+1}")
        return checkpoint["epoch"]
