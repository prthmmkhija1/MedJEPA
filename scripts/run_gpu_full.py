#!/usr/bin/env python3
"""
=============================================================
MedJEPA — Full GPU Pipeline (6 Datasets)
=============================================================

ONE script to run the complete MedJEPA pipeline:

  PHASE 1:  Pre-train LeJEPA on combined 2D data from all 6 datasets
  PHASE 2:  Pre-train V-JEPA on 3D volumetric data (BraTS + Decathlon)
  PHASE 3:  Evaluate — Linear Probe, Few-Shot, Segmentation (Dice)

Datasets:
  2D:  HAM10000 (skin), APTOS (retina), PCam (histopath), ChestXray14 (chest)
  3D->2D slices: BraTS (brain MRI), Decathlon (multi-organ CT/MRI)
  3D volumes:   BraTS + Decathlon (for V-JEPA)

Usage (from the MedJEPA root directory):
  python scripts/run_gpu_full.py

Optional flags:
  --epochs 100          (default: 100)
  --batch_size 64       (default: 64)
  --lr 0.001            (default: 0.001)
  --embed_dim 768       (default: 768)
  --encoder_depth 12    (default: 12)
  --predictor_depth 6   (default: 6)
  --skip_pretrain        (skip pretraining, go straight to eval)
  --skip_vjepa           (skip V-JEPA 3D pretraining)
  --checkpoint PATH      (use existing LeJEPA checkpoint for eval)
  --vjepa_checkpoint P   (use existing V-JEPA checkpoint)
  --max_samples N        (limit samples per dataset, for quick testing)
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import yaml  # for loading base_config.yaml

# -- Ensure project root is importable --
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)

import torch
import numpy as np
from torch.utils.data import DataLoader, ConcatDataset, random_split

from medjepa.models.lejepa import LeJEPA
from medjepa.models.vjepa import VJEPA
from medjepa.data.datasets import (
    MedicalImageDataset,
    ChestXray14Dataset,
    BraTSDataset,
    DecathlonDataset,
    VolumetricDataset,
)
from medjepa.training.trainer import MedJEPATrainer
from medjepa.evaluation.linear_probe import LinearProbeEvaluator
from medjepa.evaluation.few_shot import FewShotEvaluator
from medjepa.evaluation.segmentation import (
    SimpleSegmentationHead,
    SegmentationEvaluator,
    dice_score,
)
from medjepa.utils.device import get_device, get_device_info


# ===============================================================
# Dataset Configurations
# ===============================================================

# --- 2D datasets using MedicalImageDataset ---
DATASETS_2D = {
    "ham10000": {
        "type": "standard",
        "data_dir": "data/raw/ham10000",
        "metadata_csv": "data/raw/ham10000/HAM10000_metadata.csv",
        "image_column": "image_id",
        "label_column": "dx",
        "file_extension": ".jpg",
        "num_classes": 7,
        "task": "classification",
        "description": "Skin lesion classification (7 types)",
    },
    "aptos2019": {
        "type": "standard",
        "data_dir": "data/raw/aptos2019/train_images",
        "metadata_csv": "data/raw/aptos2019/train.csv",
        "image_column": "id_code",
        "label_column": "diagnosis",
        "file_extension": ".png",
        "num_classes": 5,
        "task": "classification",
        "description": "Diabetic retinopathy grading (5 levels)",
    },
    "pcam": {
        "type": "standard",
        "data_dir": "data/raw/pcam/train",
        "metadata_csv": "data/raw/pcam/train_labels.csv",
        "image_column": "id",
        "label_column": "label",
        "file_extension": ".tif",
        "num_classes": 2,
        "task": "classification",
        "description": "Histopathology cancer detection (binary)",
    },
    "chestxray14": {
        "type": "chestxray14",
        "data_dir": "data/raw/chestxray14",
        "num_classes": 14,
        "task": "classification",
        "description": "Chest X-ray pathology (14 diseases)",
    },
}

# --- 3D datasets that produce 2D slices for LeJEPA ---
DATASETS_3D = {
    "brats": {
        "type": "brats",
        "data_dir": "data/raw/brats",
        "modality": "flair",
        "slices_per_volume": 10,
        "num_classes": 2,
        "task": "segmentation",
        "description": "Brain tumor segmentation (BraTS 2021)",
    },
}

# Detect available Decathlon tasks
DECATHLON_BASE = Path("data/raw/decathlon")
for _task_dir in sorted(DECATHLON_BASE.glob("Task*")) if DECATHLON_BASE.exists() else []:
    if (_task_dir / "dataset.json").exists():
        _task_name = _task_dir.name
        _key = f"decathlon_{_task_name}"
        DATASETS_3D[_key] = {
            "type": "decathlon",
            "data_dir": str(DECATHLON_BASE),
            "task_name": _task_name,
            "slices_per_volume": 10,
            "num_classes": 2,
            "task": "segmentation",
            "description": f"Medical Decathlon: {_task_name}",
        }


def parse_args():
    p = argparse.ArgumentParser(description="MedJEPA Full GPU Pipeline (6 Datasets)")
    # Config file (defaults come from YAML, CLI overrides)
    p.add_argument("--config", type=str, default="configs/base_config.yaml",
                   help="Path to YAML config file (CLI flags override YAML values)")
    # Model
    p.add_argument("--image_size", type=int, default=None)
    p.add_argument("--patch_size", type=int, default=None)
    p.add_argument("--embed_dim", type=int, default=None)
    p.add_argument("--encoder_depth", type=int, default=None)
    p.add_argument("--predictor_depth", type=int, default=None)
    p.add_argument("--mask_ratio", type=float, default=None)
    p.add_argument("--lambda_reg", type=float, default=None)
    # V-JEPA 3D
    p.add_argument("--volume_size", type=int, nargs=3, default=None,
                   help="3D volume size for V-JEPA (D H W)")
    p.add_argument("--vjepa_epochs", type=int, default=None,
                   help="V-JEPA pre-training epochs")
    p.add_argument("--vjepa_batch_size", type=int, default=None,
                   help="V-JEPA batch size (3D volumes are large)")
    # Training
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--batch_size", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--warmup_epochs", type=int, default=None)
    p.add_argument("--num_workers", type=int, default=None)
    # Workflow
    p.add_argument("--skip_pretrain", action="store_true",
                   help="Skip LeJEPA pretraining; jump to evaluation")
    p.add_argument("--skip_vjepa", action="store_true",
                   help="Skip V-JEPA 3D pretraining")
    p.add_argument("--checkpoint", type=str, default=None,
                   help="Path to existing LeJEPA checkpoint")
    p.add_argument("--vjepa_checkpoint", type=str, default=None,
                   help="Path to existing V-JEPA checkpoint")
    p.add_argument("--checkpoint_dir", type=str, default=None)
    p.add_argument("--results_dir", type=str, default=None)
    # Data limits
    p.add_argument("--max_samples", type=int, default=None,
                   help="Limit samples per dataset (for quick testing)")
    # Logging
    p.add_argument("--log_every", type=int, default=None)
    p.add_argument("--save_every", type=int, default=None)
    p.add_argument("--no_tensorboard", action="store_true",
                   help="Disable TensorBoard logging")

    raw = p.parse_args()

    # ---- Load YAML config as defaults, CLI overrides ----
    yaml_cfg = {}
    if raw.config and Path(raw.config).exists():
        with open(raw.config) as f:
            yaml_cfg = yaml.safe_load(f) or {}
        print(f"Loaded config from {raw.config}")

    # Flatten YAML nested dict (model.embed_dim -> embed_dim, etc.)
    flat = {}
    _yaml_mapping = {
        "model": {
            "image_size": "image_size", "patch_size": "patch_size",
            "embed_dim": "embed_dim", "encoder_depth": "encoder_depth",
            "predictor_depth": "predictor_depth",
        },
        "masking": {"mask_ratio": "mask_ratio"},
        "training": {
            "batch_size": "batch_size", "num_epochs": "epochs",
            "learning_rate": "lr", "warmup_epochs": "warmup_epochs",
            "lambda_reg": "lambda_reg",
        },
        "data": {"num_workers": "num_workers"},
        "logging": {
            "log_every": "log_every", "save_every": "save_every",
            "checkpoint_dir": "checkpoint_dir",
        },
    }
    for section, keys in _yaml_mapping.items():
        sec = yaml_cfg.get(section, {})
        for yaml_key, arg_key in keys.items():
            if yaml_key in sec:
                flat[arg_key] = sec[yaml_key]

    # Hard-coded fallbacks (if not in YAML and not on CLI)
    _final_defaults = {
        "image_size": 224, "patch_size": 16, "embed_dim": 768,
        "encoder_depth": 12, "predictor_depth": 6, "mask_ratio": 0.75,
        "lambda_reg": 1.0, "volume_size": [128, 128, 64],
        "vjepa_epochs": 50, "vjepa_batch_size": 4,
        "epochs": 100, "batch_size": 64, "lr": 1e-3,
        "warmup_epochs": 10, "num_workers": 4,
        "checkpoint_dir": "checkpoints", "results_dir": "results",
        "log_every": 10, "save_every": 5,
    }

    # Merge: CLI > YAML > defaults
    ns = vars(raw)
    for key, default in _final_defaults.items():
        if ns.get(key) is None:
            ns[key] = flat.get(key, default)

    return argparse.Namespace(**ns)


# ===============================================================
# Helpers
# ===============================================================

def banner(text: str):
    print("\n" + "=" * 60)
    print(text)
    print("=" * 60)


def load_2d_dataset(name: str, cfg: dict, image_size: int,
                    with_labels: bool, max_samples=None):
    """Load a 2D dataset for pre-training (no labels) or eval (with labels)."""
    ds_type = cfg["type"]

    if ds_type == "standard":
        label_col = cfg["label_column"] if with_labels else None
        ds = MedicalImageDataset(
            image_dir=cfg["data_dir"],
            metadata_csv=cfg["metadata_csv"],
            image_column=cfg["image_column"],
            label_column=label_col,
            file_extension=cfg["file_extension"],
            target_size=(image_size, image_size),
        )
    elif ds_type == "chestxray14":
        ds = ChestXray14Dataset(
            data_dir=cfg["data_dir"],
            target_size=(image_size, image_size),
            with_labels=with_labels,
            max_samples=max_samples,
        )
    else:
        raise ValueError(f"Unknown 2D dataset type: {ds_type}")

    # Apply max_samples limit for standard datasets
    if max_samples and ds_type == "standard" and len(ds) > max_samples:
        indices = torch.randperm(len(ds))[:max_samples].tolist()
        ds = torch.utils.data.Subset(ds, indices)

    print(f"  {name}: {len(ds)} images")
    return ds


def load_3d_slice_dataset(name: str, cfg: dict, image_size: int,
                          with_labels: bool, max_samples=None):
    """Load a 3D dataset as 2D slices for LeJEPA pre-training."""
    ds_type = cfg["type"]
    slices = cfg.get("slices_per_volume", 10)

    if ds_type == "brats":
        ds = BraTSDataset(
            data_dir=cfg["data_dir"],
            modality=cfg.get("modality", "flair"),
            target_size=(image_size, image_size),
            slices_per_volume=slices,
            with_labels=with_labels,
            max_subjects=max_samples // slices if max_samples else None,
        )
    elif ds_type == "decathlon":
        ds = DecathlonDataset(
            data_dir=cfg["data_dir"],
            task=cfg["task_name"],
            target_size=(image_size, image_size),
            slices_per_volume=slices,
            with_labels=with_labels,
        )
    else:
        raise ValueError(f"Unknown 3D dataset type: {ds_type}")

    print(f"  {name}: {len(ds)} slices")
    return ds


def check_dataset_exists(cfg: dict) -> bool:
    """Check if a dataset's data directory exists and has content."""
    data_dir = Path(cfg["data_dir"])
    if not data_dir.exists():
        return False
    children = list(data_dir.iterdir())
    return len(children) > 0


# ===============================================================
# PHASE 1 -- LeJEPA Pre-Training (2D, all datasets combined)
# ===============================================================
def run_lejepa_pretraining(args):
    banner("PHASE 1: LeJEPA Self-Supervised Pre-Training (All 2D Data)")
    print("Combining 2D images from all available datasets...\n")

    datasets = []
    dataset_sizes = {}

    # Load 2D datasets (unlabeled)
    for name, cfg in DATASETS_2D.items():
        if not check_dataset_exists(cfg):
            print(f"  SKIP {name}: data not found at {cfg['data_dir']}")
            continue
        try:
            ds = load_2d_dataset(name, cfg, args.image_size,
                                 with_labels=False, max_samples=args.max_samples)
            datasets.append(ds)
            dataset_sizes[name] = len(ds)
        except Exception as e:
            print(f"  SKIP {name}: {e}")

    # Load 3D datasets as 2D slices (unlabeled)
    for name, cfg in DATASETS_3D.items():
        if not check_dataset_exists(cfg):
            print(f"  SKIP {name}: data not found at {cfg['data_dir']}")
            continue
        try:
            ds = load_3d_slice_dataset(name, cfg, args.image_size,
                                       with_labels=False, max_samples=args.max_samples)
            datasets.append(ds)
            dataset_sizes[name] = len(ds)
        except Exception as e:
            print(f"  SKIP {name}: {e}")

    if not datasets:
        print("ERROR: No datasets found! Run 'python scripts/download_data.py' first.")
        sys.exit(1)

    combined = ConcatDataset(datasets)
    print(f"\nCombined training set: {len(combined)} images")
    for name, size in dataset_sizes.items():
        pct = 100 * size / len(combined)
        print(f"  {name:25s}: {size:>7d} ({pct:5.1f}%)")

    # Build LeJEPA model
    model = LeJEPA(
        image_size=args.image_size,
        patch_size=args.patch_size,
        embed_dim=args.embed_dim,
        encoder_depth=args.encoder_depth,
        predictor_depth=args.predictor_depth,
        mask_ratio=args.mask_ratio,
        lambda_reg=args.lambda_reg,
    )
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nLeJEPA parameters: {total_params:,} ({total_params / 1e6:.1f}M)")

    # Training config
    config = {
        "embed_dim": args.embed_dim,
        "encoder_depth": args.encoder_depth,
        "predictor_depth": args.predictor_depth,
        "image_size": args.image_size,
        "patch_size": args.patch_size,
        "mask_ratio": args.mask_ratio,
        "lambda_reg": args.lambda_reg,
        "batch_size": args.batch_size,
        "num_epochs": args.epochs,
        "learning_rate": args.lr,
        "warmup_epochs": args.warmup_epochs,
        "weight_decay": 0.05,
        "num_workers": args.num_workers,
        "log_every": args.log_every,
        "save_every": args.save_every,
        "checkpoint_dir": args.checkpoint_dir,
        "mixed_precision": torch.cuda.is_available(),
        "use_tensorboard": not getattr(args, "no_tensorboard", False),
        "use_weighted_sampler": True,
    }

    # Train
    trainer = MedJEPATrainer(model=model, train_dataset=combined, config=config)
    history = trainer.train()

    # Save history
    ckpt_dir = Path(args.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    with open(ckpt_dir / "lejepa_training_history.json", "w") as f:
        json.dump(history, f, indent=2)

    best_ckpt = ckpt_dir / "best_model.pt"
    print(f"\nLeJEPA pre-training complete. Best model -> {best_ckpt}")
    return str(best_ckpt)


# ===============================================================
# PHASE 2 -- V-JEPA Pre-Training (3D volumes)
# ===============================================================
def run_vjepa_pretraining(args):
    banner("PHASE 2: V-JEPA Self-Supervised Pre-Training (3D Volumes)")

    volume_size = tuple(args.volume_size)
    nifti_paths = []

    # Collect all 3D NIfTI volume paths
    # BraTS
    brats_dir = Path("data/raw/brats")
    if brats_dir.exists():
        for sdir in sorted(brats_dir.iterdir()):
            if sdir.is_dir() and sdir.name.startswith("BraTS"):
                flair = sdir / f"{sdir.name}_flair.nii.gz"
                if flair.exists():
                    nifti_paths.append(str(flair))
        print(f"  BraTS: {len(nifti_paths)} volumes")

    # Decathlon
    decathlon_count = 0
    if DECATHLON_BASE.exists():
        for task_dir in sorted(DECATHLON_BASE.glob("Task*")):
            meta_path = task_dir / "dataset.json"
            if meta_path.exists():
                import json as json_mod
                with open(meta_path) as f:
                    meta = json_mod.load(f)
                for entry in meta.get("training", []):
                    img_path = task_dir / entry["image"].lstrip("./")
                    if img_path.exists():
                        nifti_paths.append(str(img_path))
                        decathlon_count += 1
        print(f"  Decathlon: {decathlon_count} volumes")

    if not nifti_paths:
        print("No 3D volumes found. Skipping V-JEPA pre-training.")
        return None

    if args.max_samples and len(nifti_paths) > args.max_samples:
        rng = np.random.RandomState(42)
        idx = rng.choice(len(nifti_paths), args.max_samples, replace=False)
        nifti_paths = [nifti_paths[i] for i in sorted(idx)]

    print(f"\nTotal 3D volumes: {len(nifti_paths)}")

    # Create volumetric dataset
    vol_dataset = VolumetricDataset(
        nifti_paths=nifti_paths,
        volume_size=volume_size,
    )
    print(f"Valid volumes: {len(vol_dataset)}")

    if len(vol_dataset) == 0:
        print("No valid volumes loaded. Skipping V-JEPA.")
        return None

    # Build V-JEPA model
    model = VJEPA(
        volume_size=volume_size,
        patch_size=(16, 16, 8),
        in_channels=1,
        embed_dim=args.embed_dim,
        depth=args.encoder_depth,
        num_heads=12,
        predictor_dim=args.embed_dim // 2,
        predictor_depth=args.predictor_depth,
        mask_ratio=args.mask_ratio,
        lambda_reg=args.lambda_reg,
    )
    total_params = sum(p.numel() for p in model.parameters())
    print(f"V-JEPA parameters: {total_params:,} ({total_params / 1e6:.1f}M)")

    # Training loop (manual, since MedJEPATrainer expects 2D)
    device = get_device()
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.05)

    # Warmup + cosine annealing (same schedule as LeJEPA)
    warmup_ep = min(args.warmup_epochs, args.vjepa_epochs // 2)
    cosine_ep = max(args.vjepa_epochs - warmup_ep, 1)
    warmup_sched = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=1e-4, end_factor=1.0,
        total_iters=max(warmup_ep, 1))
    cosine_sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cosine_ep)
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer, schedulers=[warmup_sched, cosine_sched],
        milestones=[warmup_ep])

    loader = DataLoader(
        vol_dataset,
        batch_size=args.vjepa_batch_size,
        shuffle=True,
        num_workers=min(args.num_workers, 2),
        pin_memory=torch.cuda.is_available(),
    )

    scaler = torch.amp.GradScaler('cuda') if torch.cuda.is_available() else None
    best_loss = float("inf")
    ckpt_dir = Path(args.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # TensorBoard for V-JEPA
    tb_writer = None
    if not getattr(args, "no_tensorboard", False):
        try:
            from torch.utils.tensorboard import SummaryWriter
            tb_writer = SummaryWriter(log_dir=str(Path("runs") / "vjepa"))
            print("  TensorBoard logging -> runs/vjepa")
        except ImportError:
            pass

    print(f"\nStarting V-JEPA training for {args.vjepa_epochs} epochs...")
    for epoch in range(args.vjepa_epochs):
        model.train()
        epoch_losses = []

        for batch_idx, volumes in enumerate(loader):
            volumes = volumes.to(device)
            optimizer.zero_grad()

            if scaler:
                with torch.amp.autocast('cuda'):
                    losses = model(volumes)
                    loss = losses["total_loss"]
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                losses = model(volumes)
                loss = losses["total_loss"]
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            epoch_losses.append(loss.item())

            if (batch_idx + 1) % args.log_every == 0:
                avg = np.mean(epoch_losses[-args.log_every:])
                print(f"  Epoch {epoch+1}/{args.vjepa_epochs} "
                      f"Batch {batch_idx+1}/{len(loader)} "
                      f"Loss: {avg:.4f}")

        scheduler.step()
        avg_loss = np.mean(epoch_losses)
        lr = optimizer.param_groups[0]["lr"]
        print(f"Epoch {epoch+1} avg loss: {avg_loss:.4f}  LR: {lr:.6f}")

        if tb_writer is not None:
            tb_writer.add_scalar("vjepa/loss_epoch", avg_loss, epoch)
            tb_writer.add_scalar("vjepa/lr", lr, epoch)

        if avg_loss < best_loss:
            best_loss = avg_loss
            save_path = ckpt_dir / "best_vjepa_model.pt"
            torch.save({
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "loss": best_loss,
                "config": {
                    "embed_dim": args.embed_dim,
                    "encoder_depth": args.encoder_depth,
                    "predictor_depth": args.predictor_depth,
                    "volume_size": list(volume_size),
                },
            }, save_path)

    if tb_writer is not None:
        tb_writer.close()

    print(f"\nV-JEPA pre-training complete. Best loss: {best_loss:.4f}")
    return str(ckpt_dir / "best_vjepa_model.pt")


# ===============================================================
# PHASE 3 -- Evaluation
# ===============================================================
def run_evaluation(args, lejepa_ckpt: str, vjepa_ckpt: str = None):
    banner("PHASE 3: Evaluation (Linear Probe + Few-Shot + Segmentation)")

    device = get_device()
    all_results = {}

    # -- Load LeJEPA model --
    print(f"\nLoading LeJEPA checkpoint: {lejepa_ckpt}")
    ckpt = torch.load(lejepa_ckpt, map_location=device, weights_only=False)
    ckpt_config = ckpt.get("config", {})

    embed_dim = ckpt_config.get("embed_dim", args.embed_dim)
    encoder_depth = ckpt_config.get("encoder_depth", args.encoder_depth)
    predictor_depth = ckpt_config.get("predictor_depth", args.predictor_depth)
    image_size = ckpt_config.get("image_size", args.image_size)
    patch_size = ckpt_config.get("patch_size", args.patch_size)

    lejepa = LeJEPA(
        image_size=image_size,
        patch_size=patch_size,
        embed_dim=embed_dim,
        encoder_depth=encoder_depth,
        predictor_depth=predictor_depth,
    )
    lejepa.load_state_dict(ckpt["model_state_dict"])
    lejepa = lejepa.to(device)
    lejepa.eval()
    print("LeJEPA loaded!\n")

    # -- Evaluate 2D datasets --
    for name, cfg in DATASETS_2D.items():
        if not check_dataset_exists(cfg):
            print(f"  SKIP {name}: data not found")
            continue

        banner(f"Evaluating: {name.upper()} -- {cfg['description']}")

        try:
            ds = load_2d_dataset(name, cfg, args.image_size,
                                 with_labels=True, max_samples=args.max_samples)
        except Exception as e:
            print(f"  Error loading {name}: {e}")
            continue

        # Check if we have labels
        has_labels = hasattr(ds, 'labels') and ds.labels is not None
        if not has_labels:
            inner = ds.dataset if hasattr(ds, 'dataset') else ds
            has_labels = hasattr(inner, 'labels') and inner.labels is not None

        if not has_labels:
            print(f"  No labels for {name}, skipping evaluation.")
            continue

        # 80/20 train/test split
        train_size = int(0.8 * len(ds))
        test_size = len(ds) - train_size
        if test_size < 1:
            test_size = 1
            train_size = len(ds) - 1
        train_ds, test_ds = random_split(ds, [train_size, test_size])

        train_loader = DataLoader(
            train_ds, batch_size=args.batch_size,
            num_workers=args.num_workers,
            pin_memory=torch.cuda.is_available(),
        )
        test_loader = DataLoader(
            test_ds, batch_size=args.batch_size,
            num_workers=args.num_workers,
            pin_memory=torch.cuda.is_available(),
        )

        num_classes = cfg["num_classes"]

        # Linear Probe
        print(f"\n[{name}] Linear Probing ...")
        lp = LinearProbeEvaluator(
            pretrained_model=lejepa,
            num_classes=num_classes,
            embed_dim=embed_dim,
        )
        train_feats, train_labels = lp.extract_features(train_loader)
        test_feats, test_labels = lp.extract_features(test_loader)
        print(f"  Train features: {train_feats.shape}")
        print(f"  Test  features: {test_feats.shape}")

        lp.train_probe(train_feats, train_labels)
        lp_results = lp.evaluate(test_feats, test_labels)
        print(f"  Linear Probe Accuracy: {lp_results['accuracy']:.4f}")
        if lp_results.get("auc"):
            print(f"  Linear Probe AUC:      {lp_results['auc']:.4f}")

        # Few-Shot / Data Efficiency
        print(f"\n[{name}] Few-Shot (Data Efficiency) ...")
        fs = FewShotEvaluator(pretrained_model=lejepa)
        fs_results = fs.evaluate_data_efficiency(
            train_feats, train_labels,
            test_feats, test_labels,
            fractions=[0.01, 0.05, 0.1, 0.25, 0.5, 1.0],
        )

        all_results[name] = {
            "type": "classification",
            "description": cfg["description"],
            "num_classes": num_classes,
            "linear_probing": {
                "accuracy": lp_results["accuracy"],
                "auc": lp_results.get("auc"),
            },
            "few_shot": fs_results,
        }

    # -- Evaluate 3D datasets (as 2D slices through LeJEPA) --
    for name, cfg in DATASETS_3D.items():
        if not check_dataset_exists(cfg):
            print(f"  SKIP {name}: data not found")
            continue

        banner(f"Evaluating: {name.upper()} -- {cfg['description']}")

        try:
            ds = load_3d_slice_dataset(name, cfg, args.image_size,
                                       with_labels=True, max_samples=args.max_samples)
        except Exception as e:
            print(f"  Error loading {name}: {e}")
            continue

        if len(ds) == 0:
            print(f"  No data for {name}, skipping.")
            continue

        # Check labels
        has_labels = hasattr(ds, 'labels') and ds.labels is not None
        if not has_labels:
            print(f"  No labels for {name}, evaluating without classification.")
            all_results[name] = {
                "type": "3d_slices",
                "description": cfg["description"],
                "num_slices": len(ds),
                "note": "No labels available for classification",
            }
            continue

        # Split and evaluate like 2D
        train_size = int(0.8 * len(ds))
        test_size = len(ds) - train_size
        if test_size < 1:
            test_size = 1
            train_size = len(ds) - 1
        train_ds, test_ds = random_split(ds, [train_size, test_size])

        train_loader = DataLoader(
            train_ds, batch_size=args.batch_size,
            num_workers=args.num_workers,
            pin_memory=torch.cuda.is_available(),
        )
        test_loader = DataLoader(
            test_ds, batch_size=args.batch_size,
            num_workers=args.num_workers,
            pin_memory=torch.cuda.is_available(),
        )

        num_classes = cfg.get("num_classes", 2)

        print(f"\n[{name}] Linear Probing (slice-level) ...")
        lp = LinearProbeEvaluator(
            pretrained_model=lejepa,
            num_classes=num_classes,
            embed_dim=embed_dim,
        )
        train_feats, train_labels = lp.extract_features(train_loader)
        test_feats, test_labels = lp.extract_features(test_loader)
        print(f"  Train features: {train_feats.shape}")
        print(f"  Test  features: {test_feats.shape}")

        lp.train_probe(train_feats, train_labels)
        lp_results = lp.evaluate(test_feats, test_labels)
        print(f"  Linear Probe Accuracy: {lp_results['accuracy']:.4f}")

        # Few-shot
        print(f"\n[{name}] Few-Shot ...")
        fs = FewShotEvaluator(pretrained_model=lejepa)
        fs_results = fs.evaluate_data_efficiency(
            train_feats, train_labels,
            test_feats, test_labels,
            fractions=[0.01, 0.05, 0.1, 0.25, 0.5, 1.0],
        )

        all_results[name] = {
            "type": "3d_as_2d_slices",
            "description": cfg["description"],
            "num_slices": len(ds),
            "num_classes": num_classes,
            "linear_probing": {
                "accuracy": lp_results["accuracy"],
                "auc": lp_results.get("auc"),
            },
            "few_shot": fs_results,
        }

    # -- Segmentation evaluation (Dice Score on BraTS 2D slices via LeJEPA) --
    banner("Segmentation Evaluation (Dice Score)")
    brats_cfg = DATASETS_3D.get("brats")
    brats_data_exists = brats_cfg and check_dataset_exists(brats_cfg)

    if brats_data_exists:
        try:
            from medjepa.data.datasets import NIfTISliceDataset
            import nibabel as nib

            # Build a paired (image, mask) slice dataset from BraTS
            brats_dir = Path(brats_cfg["data_dir"])
            subject_dirs = sorted([
                d for d in brats_dir.iterdir()
                if d.is_dir() and d.name.startswith("BraTS")
            ])

            class _BraTSSegSliceDataset(torch.utils.data.Dataset):
                """Yields (image_3ch, mask) pairs from BraTS FLAIR + seg."""
                def __init__(self, subjects, img_size=224, slices_per_vol=10):
                    self.samples = []  # (flair_path, seg_path, slice_idx)
                    self.img_size = img_size
                    for sdir in subjects:
                        flair = sdir / f"{sdir.name}_flair.nii.gz"
                        seg   = sdir / f"{sdir.name}_seg.nii.gz"
                        if flair.exists() and seg.exists():
                            vol = nib.load(str(flair)).get_fdata()
                            n_slices = vol.shape[2]
                            lo = int(n_slices * 0.2)
                            hi = int(n_slices * 0.8)
                            step = max((hi - lo) // slices_per_vol, 1)
                            for si in range(lo, hi, step):
                                self.samples.append((str(flair), str(seg), si))
                    self._cache = {}

                def _load(self, path):
                    if path not in self._cache:
                        self._cache[path] = nib.load(path).get_fdata().astype(np.float32)
                    return self._cache[path]

                def __len__(self):
                    return len(self.samples)

                def __getitem__(self, idx):
                    flair_p, seg_p, si = self.samples[idx]
                    vol = self._load(flair_p)
                    seg = self._load(seg_p)
                    slc = vol[:, :, si]
                    msk = seg[:, :, si]

                    # Normalize image to 0-1
                    smin, smax = slc.min(), slc.max()
                    if smax - smin > 0:
                        slc = (slc - smin) / (smax - smin)
                    else:
                        slc = np.zeros_like(slc)

                    from PIL import Image as PILImage
                    sz = (self.img_size, self.img_size)
                    pil_img = PILImage.fromarray((slc * 255).astype(np.uint8))
                    pil_img = pil_img.resize(sz, PILImage.LANCZOS)
                    slc = np.array(pil_img, dtype=np.float32) / 255.0

                    # Binary mask: any tumor label > 0
                    msk_bin = (msk > 0).astype(np.uint8)
                    pil_msk = PILImage.fromarray(msk_bin * 255)
                    pil_msk = pil_msk.resize(sz, PILImage.NEAREST)
                    msk_t = torch.from_numpy(
                        (np.array(pil_msk) > 127).astype(np.int64)
                    )

                    image = np.stack([slc, slc, slc], axis=-1)
                    image = torch.from_numpy(image).permute(2, 0, 1)
                    return image, msk_t

            seg_ds = _BraTSSegSliceDataset(
                subject_dirs, img_size=image_size, slices_per_vol=10,
            )
            print(f"  BraTS seg slices: {len(seg_ds)}")

            if len(seg_ds) > 0:
                train_n = int(0.8 * len(seg_ds))
                test_n = len(seg_ds) - train_n
                if test_n < 1:
                    test_n = 1
                    train_n = len(seg_ds) - 1
                train_seg, test_seg = random_split(seg_ds, [train_n, test_n])
                train_seg_loader = DataLoader(
                    train_seg, batch_size=args.batch_size,
                    num_workers=args.num_workers,
                    pin_memory=torch.cuda.is_available(),
                )
                test_seg_loader = DataLoader(
                    test_seg, batch_size=args.batch_size,
                    num_workers=args.num_workers,
                    pin_memory=torch.cuda.is_available(),
                )

                seg_evaluator = SegmentationEvaluator(
                    pretrained_model=lejepa,
                    embed_dim=embed_dim,
                    num_classes=2,
                    image_size=image_size,
                    patch_size=patch_size,
                    lr=1e-3,
                    epochs=20,
                )

                print("  Training segmentation head on frozen LeJEPA features...")
                seg_evaluator.train_seg_head(train_seg_loader)

                print("  Evaluating Dice score...")
                dice_results = seg_evaluator.evaluate(test_seg_loader)
                print(f"  Mean Dice Score: {dice_results['mean_dice']:.4f}")
                for cls, d in dice_results["per_class_dice"].items():
                    lbl = "Background" if cls == 0 else "Tumor"
                    print(f"    Class {cls} ({lbl}): {d:.4f}")

                all_results["brats_segmentation"] = {
                    "type": "segmentation",
                    "model": "LeJEPA + SegHead",
                    "dataset": "BraTS 2021",
                    "mean_dice": dice_results["mean_dice"],
                    "per_class_dice": dice_results["per_class_dice"],
                    "num_test_slices": dice_results["num_samples"],
                }
            else:
                print("  No paired seg slices found; skipping Dice eval.")

        except Exception as e:
            print(f"  Segmentation eval failed: {e}")
            import traceback; traceback.print_exc()
    else:
        print("  BraTS data not found, skipping segmentation evaluation.")

    # -- Decathlon segmentation Dice (for each detected task) --
    for deca_key, deca_cfg in DATASETS_3D.items():
        if deca_cfg.get("type") != "decathlon":
            continue
        if not check_dataset_exists(deca_cfg):
            continue

        task_name = deca_cfg["task_name"]
        banner(f"Segmentation Evaluation -- Decathlon {task_name}")
        try:
            import nibabel as nib
            import json as json_mod
            task_dir = Path(deca_cfg["data_dir"]) / task_name
            meta_path = task_dir / "dataset.json"
            if not meta_path.exists():
                print(f"  No dataset.json for {task_name}; skipping.")
                continue
            with open(meta_path) as _f:
                meta = json_mod.load(_f)

            class _DecaSegSliceDataset(torch.utils.data.Dataset):
                """Yields (image_3ch, mask) pairs from a Decathlon task."""
                def __init__(self, entries, tdir, img_size=224, slices_per_vol=10):
                    self.samples = []
                    self.img_size = img_size
                    self._cache = {}
                    for entry in entries:
                        img_p = tdir / entry["image"].lstrip("./")
                        lbl_rel = entry.get("label", "")
                        lbl_p = tdir / lbl_rel.lstrip("./") if lbl_rel else None
                        if img_p.exists() and lbl_p and lbl_p.exists():
                            vol = nib.load(str(img_p)).get_fdata()
                            n = vol.shape[2] if vol.ndim >= 3 else 1
                            lo, hi = int(n * 0.2), int(n * 0.8)
                            step = max((hi - lo) // slices_per_vol, 1)
                            for si in range(lo, hi, step):
                                self.samples.append((str(img_p), str(lbl_p), si))

                def _load(self, path):
                    if path not in self._cache:
                        self._cache[path] = nib.load(path).get_fdata().astype(np.float32)
                    return self._cache[path]

                def __len__(self):
                    return len(self.samples)

                def __getitem__(self, idx):
                    img_p, lbl_p, si = self.samples[idx]
                    vol = self._load(img_p)
                    seg = self._load(lbl_p)
                    # Handle multi-channel volumes (e.g. T2+ADC)
                    if vol.ndim == 4:
                        slc = vol[:, :, si, 0]
                    elif vol.ndim >= 3:
                        slc = vol[:, :, si]
                    else:
                        slc = vol
                    if seg.ndim >= 3:
                        msk = seg[:, :, si]
                    else:
                        msk = seg

                    smin, smax = slc.min(), slc.max()
                    if smax - smin > 0:
                        slc = (slc - smin) / (smax - smin)
                    else:
                        slc = np.zeros_like(slc)

                    from PIL import Image as PILImage
                    sz = (self.img_size, self.img_size)
                    pil_img = PILImage.fromarray((slc * 255).astype(np.uint8))
                    pil_img = pil_img.resize(sz, PILImage.LANCZOS)
                    slc = np.array(pil_img, dtype=np.float32) / 255.0

                    msk_bin = (msk > 0).astype(np.uint8)
                    pil_msk = PILImage.fromarray(msk_bin * 255)
                    pil_msk = pil_msk.resize(sz, PILImage.NEAREST)
                    msk_t = torch.from_numpy(
                        (np.array(pil_msk) > 127).astype(np.int64))

                    image = np.stack([slc, slc, slc], axis=-1)
                    image = torch.from_numpy(image).permute(2, 0, 1)
                    return image, msk_t

            deca_seg_ds = _DecaSegSliceDataset(
                meta.get("training", []), task_dir,
                img_size=image_size, slices_per_vol=10,
            )
            print(f"  {task_name} seg slices: {len(deca_seg_ds)}")

            if len(deca_seg_ds) > 0:
                tr_n = int(0.8 * len(deca_seg_ds))
                te_n = len(deca_seg_ds) - tr_n
                if te_n < 1:
                    te_n = 1; tr_n = len(deca_seg_ds) - 1
                tr_ds, te_ds = random_split(deca_seg_ds, [tr_n, te_n])
                tr_loader = DataLoader(tr_ds, batch_size=args.batch_size,
                                       num_workers=args.num_workers,
                                       pin_memory=torch.cuda.is_available())
                te_loader = DataLoader(te_ds, batch_size=args.batch_size,
                                       num_workers=args.num_workers,
                                       pin_memory=torch.cuda.is_available())

                seg_ev = SegmentationEvaluator(
                    pretrained_model=lejepa, embed_dim=embed_dim,
                    num_classes=2, image_size=image_size,
                    patch_size=patch_size, lr=1e-3, epochs=20,
                )
                print(f"  Training seg head for {task_name}...")
                seg_ev.train_seg_head(tr_loader)
                print(f"  Evaluating Dice for {task_name}...")
                deca_dice = seg_ev.evaluate(te_loader)
                print(f"  Mean Dice: {deca_dice['mean_dice']:.4f}")
                for cls, d in deca_dice["per_class_dice"].items():
                    print(f"    Class {cls}: {d:.4f}")

                all_results[f"{deca_key}_segmentation"] = {
                    "type": "segmentation",
                    "model": "LeJEPA + SegHead",
                    "dataset": task_name,
                    "mean_dice": deca_dice["mean_dice"],
                    "per_class_dice": deca_dice["per_class_dice"],
                    "num_test_slices": deca_dice["num_samples"],
                }
            else:
                print(f"  No paired slices for {task_name}; skipping.")

        except Exception as e:
            print(f"  Decathlon {task_name} seg eval failed: {e}")
            import traceback; traceback.print_exc()

    # -- Save all results --
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    out_path = results_dir / "evaluation_results.json"

    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    # -- Print summary --
    banner("ALL RESULTS SUMMARY")
    print(f"{'Dataset':25s} | {'Type':15s} | {'LP Acc':>8s} | {'AUC':>8s} | {'Dice':>8s}")
    print("-" * 78)
    for name, res in all_results.items():
        lp = res.get("linear_probing", {})
        acc = lp.get("accuracy", None)
        auc = lp.get("auc", None)
        dice = res.get("mean_dice", None)
        rtype = res.get("type", "?")
        acc_str = f"{acc:.4f}" if acc is not None else "N/A"
        auc_str = f"{auc:.4f}" if auc is not None else "N/A"
        dice_str = f"{dice:.4f}" if dice is not None else "N/A"
        print(f"{name:25s} | {rtype:15s} | {acc_str:>8s} | {auc_str:>8s} | {dice_str:>8s}")

        if res.get("few_shot"):
            for fs_entry in res["few_shot"]:
                frac = fs_entry.get("fraction", "?")
                facc = fs_entry.get("accuracy", 0)
                print(f"  {'':23s} |  {frac*100:5.1f}% data -> Acc: {facc:.4f}")

    print(f"\nResults saved to: {out_path}")
    return all_results


# ===============================================================
# Main
# ===============================================================
def main():
    args = parse_args()
    start = time.time()

    # Reproducibility
    seed = 42
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    banner("MedJEPA -- Full GPU Pipeline (6 Datasets)")
    get_device_info()

    # Report available datasets
    print("\nAvailable datasets:")
    for name in DATASETS_2D:
        exists = check_dataset_exists(DATASETS_2D[name])
        status = "FOUND" if exists else "NOT FOUND"
        print(f"  [2D] {name:25s} : {status}")
    for name in DATASETS_3D:
        exists = check_dataset_exists(DATASETS_3D[name])
        status = "FOUND" if exists else "NOT FOUND"
        print(f"  [3D] {name:25s} : {status}")

    # Phase 1: LeJEPA Pre-Training
    if args.skip_pretrain:
        lejepa_ckpt = args.checkpoint or str(
            Path(args.checkpoint_dir) / "best_model.pt"
        )
        print(f"\nSkipping LeJEPA pre-training. Using: {lejepa_ckpt}")
    else:
        lejepa_ckpt = run_lejepa_pretraining(args)

    # Phase 2: V-JEPA Pre-Training
    vjepa_ckpt = args.vjepa_checkpoint
    if not args.skip_vjepa and not vjepa_ckpt:
        vjepa_ckpt = run_vjepa_pretraining(args)
    elif args.skip_vjepa:
        print("\nSkipping V-JEPA pre-training.")
        if not vjepa_ckpt:
            default_vjepa = Path(args.checkpoint_dir) / "best_vjepa_model.pt"
            if default_vjepa.exists():
                vjepa_ckpt = str(default_vjepa)
                print(f"  Found existing V-JEPA checkpoint: {vjepa_ckpt}")

    # Phase 3: Evaluation
    run_evaluation(args, lejepa_ckpt, vjepa_ckpt)

    elapsed = time.time() - start
    hours, rem = divmod(elapsed, 3600)
    mins, secs = divmod(rem, 60)
    banner(f"DONE -- Total time: {int(hours)}h {int(mins)}m {int(secs)}s")


if __name__ == "__main__":
    main()
