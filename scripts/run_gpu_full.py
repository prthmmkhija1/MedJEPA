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
  --batch_size 256      (default: 256, A100-40GB tuned)
  --lr 0.0028           (default: 2.8e-3, linearly scaled for bs=256)
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
from medjepa.data.augmentations import MedJEPAAugmentation
from medjepa.data.datasets import (
    MedicalImageDataset,
    ChestXray14Dataset,
    BraTSDataset,
    DecathlonDataset,
    PreExtractedSliceDataset,
    VolumetricDataset,
    RamCachedDataset,
)
from medjepa.training.trainer import MedJEPATrainer
from medjepa.evaluation.linear_probe import LinearProbeEvaluator
from medjepa.evaluation.few_shot import FewShotEvaluator
from medjepa.evaluation.segmentation import (
    SimpleSegmentationHead,
    SegmentationEvaluator,
    dice_score,
)
from medjepa.evaluation.fine_tune import FineTuneEvaluator, ImageNetBaselineEvaluator
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
    p.add_argument("--gradient_accumulation_steps", type=int, default=1,
                   help="Accumulate gradients over N mini-batches (effective batch = batch_size * N)")
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
    p.add_argument("--cache_dataset", action="store_true",
                   help="Cache entire dataset in RAM after first epoch (zero I/O for epoch 2+)")
    p.add_argument("--cache_gb", type=float, default=8.0,
                   help="Max RAM (GiB) to use for dataset cache (default: 8.0)")
    p.add_argument("--no_split_encoding", action="store_true",
                   help="Disable split encoding (run full encoder, slower)")
    p.add_argument("--gradient_checkpointing", action="store_true",
                   help="Trade ~33%% extra compute for ~2x memory savings (allows bigger batch)")
    p.add_argument("--no_prefetcher", action="store_true",
                   help="Disable CUDA stream prefetcher")

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
        "epochs": 100, "batch_size": 256, "lr": 2.8e-3,
        "warmup_epochs": 10, "num_workers": 8,
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


# Pre-extracted slices base directory
PREEXTRACTED_BASE = Path("data/processed/nifti_slices")


def _find_preextracted_dir(name: str, ds_type: str, cfg: dict):
    """Return Path to pre-extracted slice folder, or None if not available."""
    if not PREEXTRACTED_BASE.exists():
        return None
    if ds_type == "brats":
        d = PREEXTRACTED_BASE / "brats"
    elif ds_type == "decathlon":
        task_name = cfg.get("task_name", "")
        d = PREEXTRACTED_BASE / f"decathlon_{task_name}"
    else:
        return None
    if d.exists() and any(d.glob("*.npy")):
        return d
    return None


def load_3d_slice_dataset(name: str, cfg: dict, image_size: int,
                          with_labels: bool, max_samples=None):
    """Load a 3D dataset as 2D slices for LeJEPA pre-training.

    Automatically uses pre-extracted .npy slices when available
    (10-50x faster than on-the-fly NIfTI decompression).
    """
    ds_type = cfg["type"]
    slices = cfg.get("slices_per_volume", 10)

    # ── Fast path: use pre-extracted slices if they exist ──
    pre_dir = _find_preextracted_dir(name, ds_type, cfg)
    if pre_dir is not None:
        manifest_path = pre_dir.parent / "manifest.json"
        entries = None
        if manifest_path.exists():
            with open(manifest_path) as f:
                manifest = json.load(f)
            # Lookup key: "brats" or "decathlon_Task02_Heart"
            key = name  # e.g. "decathlon_Task02_Heart" or "brats"
            entries = manifest.get(key)
        ds = PreExtractedSliceDataset(
            slice_dir=str(pre_dir),
            entries=entries,
            with_labels=with_labels,
            transform=None,
        )
        if len(ds) > 0:
            print(f"  {name}: {len(ds)} PRE-EXTRACTED slices (fast path)")
            return ds
        # Fallback to original if pre-extracted dir is empty

    # ── Slow path: on-the-fly NIfTI loading ──
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
    dataset_sources = {}  # maps dataset name -> source group for balancing

    # Load 2D datasets (unlabeled) — each is its own source
    for name, cfg in DATASETS_2D.items():
        if not check_dataset_exists(cfg):
            print(f"  SKIP {name}: data not found at {cfg['data_dir']}")
            continue
        try:
            ds = load_2d_dataset(name, cfg, args.image_size,
                                 with_labels=False, max_samples=args.max_samples)
            if len(ds) == 0:
                print(f"  SKIP {name}: loaded but empty (0 images)")
                continue
            datasets.append(ds)
            dataset_sizes[name] = len(ds)
            dataset_sources[name] = name  # own source
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
            if len(ds) == 0:
                print(f"  SKIP {name}: loaded but empty (0 slices)")
                continue
            datasets.append(ds)
            dataset_sizes[name] = len(ds)
            # Group all Decathlon sub-tasks under one "decathlon" source
            dataset_sources[name] = "decathlon" if name.startswith("decathlon_") else name
        except Exception as e:
            print(f"  SKIP {name}: {e}")

    if not datasets:
        print("ERROR: No datasets found! Run 'python scripts/download_data.py' first.")
        sys.exit(1)

    # Safety net: remove any datasets that ended up with 0 samples
    empty = [n for n, s in dataset_sizes.items() if s == 0]
    if empty:
        print(f"  Removing {len(empty)} empty dataset(s): {empty}")
        # Rebuild lists without empty datasets (safer than in-place deletion)
        filtered_datasets = []
        filtered_sizes = {}
        filtered_sources = {}
        for ds, (name, size) in zip(datasets, dataset_sizes.items()):
            if size > 0:
                filtered_datasets.append(ds)
                filtered_sizes[name] = size
                filtered_sources[name] = dataset_sources[name]
        datasets = filtered_datasets
        dataset_sizes = filtered_sizes
        dataset_sources = filtered_sources

    if not datasets:
        print("ERROR: All datasets are empty! Check your data directories.")
        sys.exit(1)

    combined = ConcatDataset(datasets)
    print(f"\nCombined training set: {len(combined)} images")
    for name, size in dataset_sizes.items():
        pct = 100 * size / len(combined)
        print(f"  {name:25s}: {size:>7d} ({pct:5.1f}%)")

    # --------------- Hierarchical 2-level balanced sampler ---------------
    # Level 1: Each SOURCE (ham, aptos, pcam, chestxray14, brats, decathlon)
    #          gets equal total weight = 1 / num_sources.
    # Level 2: Within a multi-task source (decathlon), each sub-task gets
    #          equal share of that source's weight.
    # Result: No source dominates, no sub-task dominates within a source.
    # -------------------------------------------------------------------
    print("\nBuilding hierarchical balanced sampler (equal weight per source)...")
    names_list = list(dataset_sizes.keys())
    unique_sources = sorted(set(dataset_sources.values()))
    num_sources = len(unique_sources)

    # Count how many sub-datasets belong to each source
    source_member_count = {}
    for src in unique_sources:
        source_member_count[src] = sum(
            1 for n in names_list if dataset_sources[n] == src
        )

    print(f"  Sources ({num_sources}): {', '.join(unique_sources)}")
    for src in unique_sources:
        members = [n for n in names_list if dataset_sources[n] == src]
        print(f"    {src}: {len(members)} sub-dataset(s) — {members}")

    sample_weights = []
    for i, ds in enumerate(datasets):
        name = names_list[i]
        src = dataset_sources[name]
        ds_size = len(ds)
        if ds_size == 0:
            continue
        n_members = source_member_count[src]
        # w = (1/num_sources) * (1/n_members_in_source) * (1/ds_size)
        # → every source equal, every sub-task within that source equal
        w = 1.0 / (num_sources * n_members * ds_size)
        sample_weights.extend([w] * ds_size)

    balanced_sampler = torch.utils.data.WeightedRandomSampler(
        weights=torch.tensor(sample_weights, dtype=torch.float64),
        num_samples=len(combined),
        replacement=True,
    )
    print(f"  Balanced sampler ready — {num_sources} sources, {len(combined)} total samples")

    # Optionally wrap with RAM cache (eliminates all disk I/O from epoch 2 onward)
    if getattr(args, 'cache_dataset', False):
        print(f"  Wrapping dataset in RamCachedDataset (max {args.cache_gb:.1f} GB)...")
        combined = RamCachedDataset(combined, max_gb=args.cache_gb, verbose=True)
        print("  RAM cache ready.")

    # Build LeJEPA model
    _split_encoding = not getattr(args, "no_split_encoding", False)
    _grad_ckpt = getattr(args, "gradient_checkpointing", False)
    aug = MedJEPAAugmentation(image_size=args.image_size)
    model = LeJEPA(
        image_size=args.image_size,
        patch_size=args.patch_size,
        embed_dim=args.embed_dim,
        encoder_depth=args.encoder_depth,
        predictor_depth=args.predictor_depth,
        mask_ratio=args.mask_ratio,
        lambda_reg=args.lambda_reg,
        split_encoding=_split_encoding,
        gradient_checkpointing=_grad_ckpt,
        use_ema=True,
        ema_momentum=0.996,
        augmentation=aug,
    )
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nLeJEPA parameters: {total_params:,} ({total_params / 1e6:.1f}M)")
    print(f"  Split encoding:         {'ON (fast)' if _split_encoding else 'OFF'}")
    print(f"  Gradient checkpointing: {'ON (saves memory)' if _grad_ckpt else 'OFF'}")

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
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "compile_model": True,
        "use_prefetcher": not getattr(args, "no_prefetcher", False),
    }

    # Train
    trainer = MedJEPATrainer(model=model, train_dataset=combined, config=config,
                             sampler=balanced_sampler)
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

    # ── Collect 3D NIfTI paths, tracking source for balanced sampling ──
    source_paths = {}  # source_name -> [path, ...]

    # BraTS
    brats_dir = Path("data/raw/brats")
    brats_list = []
    if brats_dir.exists():
        # Auto-extract .tar archives if present
        import tarfile
        for tf_path in list(brats_dir.glob("*.tar*")) + list(brats_dir.glob("*.tgz")):
            marker = tf_path.with_suffix(tf_path.suffix + ".extracted")
            if not marker.exists():
                try:
                    print(f"  Extracting {tf_path.name} ...")
                    with tarfile.open(str(tf_path), "r:*") as tar:
                        tar.extractall(path=str(brats_dir))
                    marker.touch()
                except Exception as e:
                    print(f"  WARNING: Failed to extract {tf_path.name}: {e}")

        # Collect subject dirs, looking inside container folders too
        candidate_dirs = sorted([
            d for d in brats_dir.iterdir()
            if d.is_dir() and d.name.startswith("BraTS")
        ])
        subject_dirs = []
        for d in candidate_dirs:
            flair = d / f"{d.name}_flair.nii.gz"
            if flair.exists():
                subject_dirs.append(d)
            else:
                # Container folder — look one level deeper
                nested = sorted([
                    sd for sd in d.iterdir()
                    if sd.is_dir() and sd.name.startswith("BraTS")
                ]) if d.is_dir() else []
                subject_dirs.extend(nested)

        for sdir in subject_dirs:
            flair = sdir / f"{sdir.name}_flair.nii.gz"
            if flair.exists():
                brats_list.append(str(flair))
    if brats_list:
        source_paths["brats"] = brats_list
        print(f"  BraTS: {len(brats_list)} volumes")

    # Decathlon — each task is a sub-entry under one "decathlon" source
    decathlon_task_paths = {}  # task_name -> [path, ...]
    if DECATHLON_BASE.exists():
        for task_dir in sorted(DECATHLON_BASE.glob("Task*")):
            meta_path = task_dir / "dataset.json"
            if meta_path.exists():
                import json as json_mod
                with open(meta_path) as f:
                    meta = json_mod.load(f)
                task_list = []
                for entry in meta.get("training", []):
                    img_path = task_dir / entry["image"].lstrip("./")
                    if img_path.exists():
                        task_list.append(str(img_path))

                # Fallback: if imagesTr entries are missing, try imagesTs
                if not task_list:
                    images_tr = task_dir / "imagesTr"
                    images_ts = task_dir / "imagesTs"
                    tr_files = sorted(images_tr.glob("*.nii.gz")) if images_tr.exists() else []
                    ts_files = sorted(images_ts.glob("*.nii.gz")) if images_ts.exists() else []
                    if tr_files:
                        task_list = [str(p) for p in tr_files]
                    elif ts_files:
                        print(f"  Decathlon/{task_dir.name}: imagesTr empty, "
                              f"using {len(ts_files)} volumes from imagesTs/")
                        task_list = [str(p) for p in ts_files]

                if task_list:
                    decathlon_task_paths[task_dir.name] = task_list
                    print(f"  Decathlon/{task_dir.name}: {len(task_list)} volumes")

    # Flatten all paths + build per-sample source weights
    nifti_paths = []
    sample_source_weights = []
    num_sources = (1 if source_paths.get("brats") else 0) + (1 if decathlon_task_paths else 0)

    if num_sources == 0:
        print("No 3D volumes found. Skipping V-JEPA pre-training.")
        return None

    # BraTS volumes
    if "brats" in source_paths:
        brats = source_paths["brats"]
        # weight = 1/num_sources * 1/len(brats)  (BraTS is one source)
        w = 1.0 / (num_sources * len(brats))
        for p in brats:
            nifti_paths.append(p)
            sample_source_weights.append(w)

    # Decathlon volumes — hierarchical: source share / num_tasks / task_size
    if decathlon_task_paths:
        n_tasks = len(decathlon_task_paths)
        for tname, tpaths in decathlon_task_paths.items():
            w = 1.0 / (num_sources * n_tasks * len(tpaths))
            for p in tpaths:
                nifti_paths.append(p)
                sample_source_weights.append(w)

    if args.max_samples and len(nifti_paths) > args.max_samples:
        rng = np.random.RandomState(42)
        idx = rng.choice(len(nifti_paths), args.max_samples, replace=False)
        nifti_paths = [nifti_paths[i] for i in sorted(idx)]
        sample_source_weights = [sample_source_weights[i] for i in sorted(idx)]

    print(f"\nTotal 3D volumes: {len(nifti_paths)}  Sources: {num_sources}")

    # Create volumetric dataset
    vol_dataset = VolumetricDataset(
        nifti_paths=nifti_paths,
        volume_size=volume_size,
    )

    # VolumetricDataset may drop invalid volumes — rebuild weights to match
    if hasattr(vol_dataset, 'valid_indices'):
        sample_source_weights = [sample_source_weights[i] for i in vol_dataset.valid_indices]
    elif len(vol_dataset) < len(nifti_paths):
        # Fallback: use uniform weights if we can't track which were dropped
        sample_source_weights = [1.0 / len(vol_dataset)] * len(vol_dataset)

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

    # Build balanced sampler for 3D volumes (BraTS vs Decathlon sources)
    vjepa_sampler = torch.utils.data.WeightedRandomSampler(
        weights=torch.tensor(sample_source_weights, dtype=torch.float64),
        num_samples=len(vol_dataset),
        replacement=True,
    )
    print(f"  V-JEPA balanced sampler ready — {num_sources} sources, {len(vol_dataset)} volumes")

    loader = DataLoader(
        vol_dataset,
        batch_size=args.vjepa_batch_size,
        sampler=vjepa_sampler,
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
        use_ema=True,
    )
    lejepa.load_state_dict(ckpt["model_state_dict"], strict=False)
    # Load EMA encoder weights if available (used for inference)
    if "ema_encoder_state_dict" in ckpt:
        lejepa.ema_encoder.load_state_dict(ckpt["ema_encoder_state_dict"])
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
        _multi_label = (name == "chestxray14")
        lp = LinearProbeEvaluator(
            pretrained_model=lejepa,
            num_classes=num_classes,
            embed_dim=embed_dim,
            multi_label=_multi_label,
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

        # N-Shot Evaluation (5-shot, 10-shot, 20-shot per class)
        print(f"\n[{name}] N-Shot Classification (5/10/20-shot) ...")
        n_shot_results = {}
        unique_classes = torch.unique(train_labels)
        for n_shot in [5, 10, 20]:
            # Sample n_shot examples per class from training features
            support_idx = []
            for cls in unique_classes:
                cls_idx = (train_labels == cls).nonzero(as_tuple=True)[0]
                if len(cls_idx) >= n_shot:
                    perm = cls_idx[torch.randperm(len(cls_idx))[:n_shot]]
                else:
                    perm = cls_idx  # use all available if fewer than n_shot
                support_idx.append(perm)
            support_idx = torch.cat(support_idx)
            support_feats = train_feats[support_idx]
            support_labs = train_labels[support_idx]

            from sklearn.neighbors import KNeighborsClassifier
            from sklearn.metrics import accuracy_score
            k = min(5, len(support_labs))
            knn = KNeighborsClassifier(n_neighbors=max(1, k))
            knn.fit(support_feats.numpy(), support_labs.numpy())
            preds = knn.predict(test_feats.numpy())
            acc = accuracy_score(test_labels.numpy(), preds)
            n_shot_results[f"{n_shot}-shot"] = {
                "accuracy": acc,
                "num_support": len(support_labs),
            }
            print(f"  {n_shot}-shot: Accuracy = {acc:.4f} ({len(support_labs)} support samples)")

        # Supervised Baseline (random init model, same linear probe)
        print(f"\n[{name}] Supervised Baseline (random init) ...")
        baseline_model = LeJEPA(
            image_size=image_size,
            patch_size=patch_size,
            embed_dim=embed_dim,
            encoder_depth=encoder_depth,
            predictor_depth=predictor_depth,
        ).to(device)
        baseline_model.eval()

        baseline_lp = LinearProbeEvaluator(
            pretrained_model=baseline_model,
            num_classes=num_classes,
            embed_dim=embed_dim,
        )
        baseline_train_feats, baseline_train_labels = baseline_lp.extract_features(train_loader)
        baseline_test_feats, baseline_test_labels = baseline_lp.extract_features(test_loader)
        baseline_lp.train_probe(baseline_train_feats, baseline_train_labels)
        baseline_results = baseline_lp.evaluate(baseline_test_feats, baseline_test_labels)
        print(f"  Baseline Accuracy: {baseline_results['accuracy']:.4f}")
        print(f"  MedJEPA  Accuracy: {lp_results['accuracy']:.4f}")
        improvement = lp_results['accuracy'] - baseline_results['accuracy']
        print(f"  Improvement:       {improvement:+.4f}")

        del baseline_model  # free memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # ImageNet Pretrained Baseline (ViT-B/16)
        print(f"\n[{name}] ImageNet ViT-B/16 Baseline ...")
        inet_results = {"accuracy": None, "auc": None}
        try:
            inet_eval = ImageNetBaselineEvaluator(
                num_classes=num_classes, backbone="vit_b_16",
            )
            inet_train_feats, inet_train_labs = inet_eval.extract_features(train_loader)
            inet_test_feats, inet_test_labs = inet_eval.extract_features(test_loader)
            inet_eval.train_probe(inet_train_feats, inet_train_labs)
            inet_results = inet_eval.evaluate(inet_test_feats, inet_test_labs)
            print(f"  ImageNet Baseline Accuracy: {inet_results['accuracy']:.4f}")
            print(f"  MedJEPA LP Accuracy:        {lp_results['accuracy']:.4f}")
            inet_imp = lp_results['accuracy'] - inet_results['accuracy']
            print(f"  MedJEPA vs ImageNet:        {inet_imp:+.4f}")
            del inet_eval
        except Exception as e:
            print(f"  ImageNet baseline failed: {e}")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Full Fine-Tuning (end-to-end, encoder + classification head)
        print(f"\n[{name}] Full Fine-Tuning ...")
        ft_results = {"accuracy": None, "auc": None}
        try:
            ft_model = LeJEPA(
                image_size=image_size, patch_size=patch_size,
                embed_dim=embed_dim, encoder_depth=encoder_depth,
                predictor_depth=predictor_depth, use_ema=True,
            )
            ft_model.load_state_dict(ckpt["model_state_dict"], strict=False)
            if "ema_encoder_state_dict" in ckpt:
                ft_model.ema_encoder.load_state_dict(ckpt["ema_encoder_state_dict"])
            ft_model = ft_model.to(device)

            _multi_label_ft = (name == "chestxray14")
            ft_eval = FineTuneEvaluator(
                pretrained_model=ft_model,
                num_classes=num_classes,
                embed_dim=embed_dim,
                num_epochs=20,
            )
            ft_history = ft_eval.train(train_loader, test_loader)
            ft_results = ft_eval.evaluate(test_loader)
            print(f"  Fine-Tune Accuracy: {ft_results['accuracy']:.4f}")
            if ft_results.get("auc"):
                print(f"  Fine-Tune AUC:      {ft_results['auc']:.4f}")
            del ft_model, ft_eval
        except Exception as e:
            print(f"  Fine-tuning failed: {e}")
            import traceback; traceback.print_exc()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        all_results[name] = {
            "type": "classification",
            "description": cfg["description"],
            "num_classes": num_classes,
            "linear_probing": {
                "accuracy": lp_results["accuracy"],
                "auc": lp_results.get("auc"),
            },
            "few_shot": fs_results,
            "n_shot": n_shot_results,
            "supervised_baseline": {
                "accuracy": baseline_results["accuracy"],
                "auc": baseline_results.get("auc"),
            },
            "imagenet_baseline": {
                "accuracy": inet_results["accuracy"],
                "auc": inet_results.get("auc"),
            },
            "fine_tuning": {
                "accuracy": ft_results["accuracy"],
                "auc": ft_results.get("auc"),
            },
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

        # N-Shot Evaluation for 3D slice datasets
        print(f"\n[{name}] N-Shot Classification (5/10/20-shot) ...")
        n_shot_results = {}
        unique_classes = torch.unique(train_labels)
        for n_shot in [5, 10, 20]:
            support_idx = []
            for cls in unique_classes:
                cls_idx = (train_labels == cls).nonzero(as_tuple=True)[0]
                if len(cls_idx) >= n_shot:
                    perm = cls_idx[torch.randperm(len(cls_idx))[:n_shot]]
                else:
                    perm = cls_idx
                support_idx.append(perm)
            support_idx = torch.cat(support_idx)
            support_feats = train_feats[support_idx]
            support_labs = train_labels[support_idx]

            from sklearn.neighbors import KNeighborsClassifier
            from sklearn.metrics import accuracy_score
            k = min(5, len(support_labs))
            knn = KNeighborsClassifier(n_neighbors=max(1, k))
            knn.fit(support_feats.numpy(), support_labs.numpy())
            preds = knn.predict(test_feats.numpy())
            acc = accuracy_score(test_labels.numpy(), preds)
            n_shot_results[f"{n_shot}-shot"] = {
                "accuracy": acc,
                "num_support": len(support_labs),
            }
            print(f"  {n_shot}-shot: Accuracy = {acc:.4f}")

        # Supervised Baseline for 3D
        print(f"\n[{name}] Supervised Baseline (random init) ...")
        baseline_model = LeJEPA(
            image_size=image_size,
            patch_size=patch_size,
            embed_dim=embed_dim,
            encoder_depth=encoder_depth,
            predictor_depth=predictor_depth,
        ).to(device)
        baseline_model.eval()
        baseline_lp = LinearProbeEvaluator(
            pretrained_model=baseline_model,
            num_classes=num_classes,
            embed_dim=embed_dim,
        )
        bl_train_feats, bl_train_labels = baseline_lp.extract_features(train_loader)
        bl_test_feats, bl_test_labels = baseline_lp.extract_features(test_loader)
        baseline_lp.train_probe(bl_train_feats, bl_train_labels)
        bl_results = baseline_lp.evaluate(bl_test_feats, bl_test_labels)
        print(f"  Baseline Accuracy: {bl_results['accuracy']:.4f}")
        print(f"  MedJEPA  Accuracy: {lp_results['accuracy']:.4f}")
        del baseline_model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

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
            "n_shot": n_shot_results,
            "supervised_baseline": {
                "accuracy": bl_results["accuracy"],
                "auc": bl_results.get("auc"),
            },
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
                            n_slices = nib.load(str(flair)).shape[2]
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
                    msk_t = torch.tensor(
                        (np.array(pil_msk) > 127).astype(np.int64)
                    )

                    image = np.stack([slc, slc, slc], axis=-1)
                    image = torch.tensor(image, dtype=torch.float32).permute(2, 0, 1)
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
                            shape = nib.load(str(img_p)).shape
                            n = shape[2] if len(shape) >= 3 else 1
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
                    msk_t = torch.tensor(
                        (np.array(pil_msk) > 127).astype(np.int64))

                    image = np.stack([slc, slc, slc], axis=-1)
                    image = torch.tensor(image, dtype=torch.float32).permute(2, 0, 1)
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

    # -- Cross-Institutional Validation (Domain Generalization) --
    banner("Cross-Institutional Validation (Domain Generalization)")
    print("Testing representation quality across different medical institutions/datasets")
    print("A good SSL encoder produces separable clusters regardless of data source.\n")

    cross_results = {}
    dataset_features = {}

    for ci_name, ci_cfg in DATASETS_2D.items():
        if not check_dataset_exists(ci_cfg):
            continue
        try:
            ci_ds = load_2d_dataset(ci_name, ci_cfg, image_size,
                                    with_labels=True, max_samples=args.max_samples)
            _has_labels = hasattr(ci_ds, 'labels') and ci_ds.labels is not None
            if not _has_labels:
                _inner = ci_ds.dataset if hasattr(ci_ds, 'dataset') else ci_ds
                _has_labels = hasattr(_inner, 'labels') and _inner.labels is not None
            if not _has_labels:
                continue
            ci_loader = DataLoader(
                ci_ds, batch_size=args.batch_size,
                num_workers=args.num_workers,
                pin_memory=torch.cuda.is_available(),
            )
            _lp = LinearProbeEvaluator(
                pretrained_model=lejepa, num_classes=ci_cfg["num_classes"],
                embed_dim=embed_dim,
            )
            ci_feats, ci_labs = _lp.extract_features(ci_loader)
            dataset_features[ci_name] = {
                "features": ci_feats, "labels": ci_labs,
                "num_classes": ci_cfg["num_classes"],
            }
            print(f"  {ci_name}: extracted {ci_feats.shape[0]} feature vectors")
        except Exception as e:
            print(f"  {ci_name}: skipped ({e})")

    if len(dataset_features) >= 2:
        from sklearn.metrics import silhouette_score

        # 1. Per-dataset silhouette score (representation quality)
        print("\n--- Per-Dataset Silhouette Scores ---")
        for ci_name, ci_data in dataset_features.items():
            try:
                _n = min(3000, len(ci_data["labels"]))
                _idx = np.random.choice(len(ci_data["labels"]), _n, replace=False)
                sil = silhouette_score(
                    ci_data["features"][_idx].numpy(),
                    ci_data["labels"][_idx].numpy(),
                )
                cross_results[f"{ci_name}_silhouette"] = float(sil)
                print(f"  {ci_name}: Silhouette = {sil:.4f}")
            except Exception as e:
                print(f"  {ci_name}: could not compute silhouette ({e})")

        # 2. Cross-dataset kNN transfer (compatible label spaces)
        print("\n--- Cross-Dataset Transfer (kNN) ---")
        ci_ds_names = list(dataset_features.keys())
        for ci_src in ci_ds_names:
            for ci_tgt in ci_ds_names:
                if ci_src == ci_tgt:
                    continue
                src_d = dataset_features[ci_src]
                tgt_d = dataset_features[ci_tgt]
                if src_d["num_classes"] != tgt_d["num_classes"]:
                    continue
                try:
                    from sklearn.neighbors import KNeighborsClassifier
                    from sklearn.metrics import accuracy_score as _acc
                    _k = min(5, len(src_d["labels"]))
                    knn = KNeighborsClassifier(n_neighbors=max(1, _k))
                    knn.fit(src_d["features"].numpy(), src_d["labels"].numpy())
                    _preds = knn.predict(tgt_d["features"].numpy())
                    _accuracy = float(_acc(tgt_d["labels"].numpy(), _preds))
                    pair_key = f"{ci_src}->{ci_tgt}"
                    cross_results[pair_key] = {
                        "accuracy": _accuracy, "type": "knn_transfer",
                    }
                    print(f"  {pair_key}: kNN Accuracy = {_accuracy:.4f}")
                except Exception as e:
                    print(f"  {ci_src}->{ci_tgt}: failed ({e})")

        # 3. Domain invariance test — can a kNN distinguish which dataset
        #    a feature vector came from?  Low accuracy = domain-invariant.
        print("\n--- Domain Invariance Test ---")
        try:
            all_ci_feats = []
            all_source_ids = []
            for _i, (_ci_name, _ci_data) in enumerate(dataset_features.items()):
                _n = min(1000, len(_ci_data["features"]))
                _idx = np.random.choice(len(_ci_data["features"]), _n, replace=False)
                all_ci_feats.append(_ci_data["features"][_idx])
                all_source_ids.extend([_i] * _n)
            all_ci_feats = torch.cat(all_ci_feats).numpy()
            all_source_ids = np.array(all_source_ids)

            from sklearn.neighbors import KNeighborsClassifier
            from sklearn.model_selection import cross_val_score
            knn_domain = KNeighborsClassifier(n_neighbors=5)
            domain_scores = cross_val_score(
                knn_domain, all_ci_feats, all_source_ids, cv=3,
            )
            domain_acc = float(np.mean(domain_scores))
            random_chance = 1.0 / len(dataset_features)
            invariance = 1.0 - (domain_acc - random_chance) / max(1.0 - random_chance, 1e-6)
            cross_results["domain_invariance"] = {
                "domain_classification_accuracy": domain_acc,
                "random_chance": random_chance,
                "invariance_score": float(np.clip(invariance, 0.0, 1.0)),
            }
            print(f"  Domain classification accuracy: {domain_acc:.4f} "
                  f"(random chance: {random_chance:.4f})")
            print(f"  Domain invariance score: {float(np.clip(invariance, 0, 1)):.4f} "
                  f"(1.0 = perfectly invariant, 0.0 = fully separable)")
        except Exception as e:
            print(f"  Domain invariance test failed: {e}")

        all_results["cross_institutional"] = cross_results
    else:
        print("  Need >= 2 datasets for cross-institutional validation. Skipping.")

    # -- Save all results --
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    out_path = results_dir / "evaluation_results.json"

    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    # -- Print summary --
    banner("ALL RESULTS SUMMARY")
    print(f"{'Dataset':25s} | {'Type':15s} | {'LP Acc':>8s} | {'Baseline':>8s} | {'ImageNet':>8s} | {'FineTune':>8s} | {'AUC':>8s} | {'Dice':>8s}")
    print("-" * 120)
    for name, res in all_results.items():
        if name == "cross_institutional":
            continue  # printed separately below
        lp = res.get("linear_probing", {})
        acc = lp.get("accuracy", None)
        bl = res.get("supervised_baseline", {})
        bl_acc = bl.get("accuracy", None)
        inet = res.get("imagenet_baseline", {})
        inet_acc = inet.get("accuracy", None)
        ft = res.get("fine_tuning", {})
        ft_acc = ft.get("accuracy", None)
        auc = lp.get("auc", None)
        dice = res.get("mean_dice", None)
        rtype = res.get("type", "?")
        acc_str = f"{acc:.4f}" if acc is not None else "N/A"
        bl_str = f"{bl_acc:.4f}" if bl_acc is not None else "N/A"
        inet_str = f"{inet_acc:.4f}" if inet_acc is not None else "N/A"
        ft_str = f"{ft_acc:.4f}" if ft_acc is not None else "N/A"
        auc_str = f"{auc:.4f}" if auc is not None else "N/A"
        dice_str = f"{dice:.4f}" if dice is not None else "N/A"
        print(f"{name:25s} | {rtype:15s} | {acc_str:>8s} | {bl_str:>8s} | {inet_str:>8s} | {ft_str:>8s} | {auc_str:>8s} | {dice_str:>8s}")

        if res.get("n_shot"):
            for shot_key, shot_val in res["n_shot"].items():
                print(f"  {'':23s} |  {shot_key:>8s} -> Acc: {shot_val['accuracy']:.4f}")

        if res.get("few_shot"):
            for fs_entry in res["few_shot"]:
                frac = fs_entry.get("fraction", "?")
                facc = fs_entry.get("accuracy", 0)
                print(f"  {'':23s} |  {frac*100:5.1f}% data -> Acc: {facc:.4f}")

    # Print cross-institutional results
    ci = all_results.get("cross_institutional", {})
    if ci:
        print("\n--- Cross-Institutional Validation ---")
        for ci_key, ci_val in ci.items():
            if isinstance(ci_val, dict) and "accuracy" in ci_val:
                print(f"  {ci_key:35s} | Accuracy: {ci_val['accuracy']:.4f}")
            elif isinstance(ci_val, dict) and "invariance_score" in ci_val:
                print(f"  Domain Invariance Score: {ci_val['invariance_score']:.4f}")
            elif isinstance(ci_val, float):
                print(f"  {ci_key:35s} | Silhouette: {ci_val:.4f}")

    print(f"\nResults saved to: {out_path}")
    return all_results


# ===============================================================
# Main
# ===============================================================
def main():
    args = parse_args()
    start = time.time()

    # Reproducibility — seeds only; leave cuDNN benchmark=True (set in trainer)
    seed = 42
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        # TF32 global setting: "high" enables TF32 matmuls for maximum A100 throughput
        # (10-bit mantissa vs float32 23-bit — negligible error for self-supervised SSL)
        torch.set_float32_matmul_precision("high")

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
