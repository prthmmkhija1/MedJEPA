#!/usr/bin/env python3
"""
=============================================================
MedJEPA — Full GPU Pipeline (HAM10000 + APTOS + PCam)
=============================================================

ONE script to upload to JupyterLab GPU and run everything:
  1. Install dependencies
  2. Pre-train on ALL three datasets combined
  3. Evaluate (Linear Probe + Few-Shot) on each dataset

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
  --checkpoint PATH      (use existing checkpoint for eval)
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

# ── Ensure project root is importable ──────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)

import torch
from torch.utils.data import DataLoader, ConcatDataset, random_split

from medjepa.models.lejepa import LeJEPA
from medjepa.data.datasets import MedicalImageDataset
from medjepa.training.trainer import MedJEPATrainer
from medjepa.evaluation.linear_probe import LinearProbeEvaluator
from medjepa.evaluation.few_shot import FewShotEvaluator
from medjepa.utils.device import get_device, get_device_info


# ═══════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════
DATASETS = {
    "ham10000": {
        "data_dir": "data/raw/ham10000",
        "metadata_csv": "data/raw/ham10000/HAM10000_metadata.csv",
        "image_column": "image_id",
        "label_column": "dx",
        "file_extension": ".jpg",
        "num_classes": 7,
    },
    "aptos2019": {
        "data_dir": "data/raw/aptos2019/train_images",
        "metadata_csv": "data/raw/aptos2019/train.csv",
        "image_column": "id_code",
        "label_column": "diagnosis",
        "file_extension": ".png",
        "num_classes": 5,
    },
    "pcam": {
        "data_dir": "data/raw/pcam/train",
        "metadata_csv": "data/raw/pcam/train_labels.csv",
        "image_column": "id",
        "label_column": "label",
        "file_extension": ".tif",
        "num_classes": 2,
    },
}


def parse_args():
    p = argparse.ArgumentParser(description="MedJEPA Full GPU Pipeline")
    # Model
    p.add_argument("--image_size", type=int, default=224)
    p.add_argument("--patch_size", type=int, default=16)
    p.add_argument("--embed_dim", type=int, default=768)
    p.add_argument("--encoder_depth", type=int, default=12)
    p.add_argument("--predictor_depth", type=int, default=6)
    p.add_argument("--mask_ratio", type=float, default=0.75)
    p.add_argument("--lambda_reg", type=float, default=1.0)
    # Training
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--num_workers", type=int, default=4)
    # Workflow
    p.add_argument("--skip_pretrain", action="store_true",
                   help="Skip pretraining; jump to evaluation")
    p.add_argument("--checkpoint", type=str, default=None,
                   help="Path to existing checkpoint (for --skip_pretrain)")
    p.add_argument("--checkpoint_dir", type=str, default="checkpoints")
    p.add_argument("--results_dir", type=str, default="results")
    # Logging
    p.add_argument("--log_every", type=int, default=10)
    p.add_argument("--save_every", type=int, default=5)
    return p.parse_args()


# ═══════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════
def load_dataset(name: str, cfg: dict, image_size: int, with_labels: bool):
    """Load a single dataset. Set label_column=None for self-supervised."""
    label_col = cfg["label_column"] if with_labels else None
    ds = MedicalImageDataset(
        image_dir=cfg["data_dir"],
        metadata_csv=cfg["metadata_csv"],
        image_column=cfg["image_column"],
        label_column=label_col,
        file_extension=cfg["file_extension"],
        target_size=(image_size, image_size),
    )
    print(f"  {name}: {len(ds)} images")
    return ds


def banner(text: str):
    print("\n" + "=" * 60)
    print(text)
    print("=" * 60)


# ═══════════════════════════════════════════════════════════
# PHASE 1 — Pre-training (self-supervised, no labels)
# ═══════════════════════════════════════════════════════════
def run_pretraining(args):
    banner("PHASE 1: Self-Supervised Pre-Training (HAM + APTOS + PCam)")

    # 1a. Load all three datasets WITHOUT labels (self-supervised)
    print("\nLoading datasets (unlabeled) ...")
    datasets = []
    for name, cfg in DATASETS.items():
        ds = load_dataset(name, cfg, args.image_size, with_labels=False)
        datasets.append(ds)

    combined = ConcatDataset(datasets)
    print(f"\nCombined training set: {len(combined)} images")

    # 1b. Build model
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
    print(f"Model parameters: {total_params:,} ({total_params / 1e6:.1f}M)")

    # 1c. Training config
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
        "weight_decay": 0.05,
        "num_workers": args.num_workers,
        "log_every": args.log_every,
        "save_every": args.save_every,
        "checkpoint_dir": args.checkpoint_dir,
        "mixed_precision": torch.cuda.is_available(),
    }

    # 1d. Train!
    trainer = MedJEPATrainer(model=model, train_dataset=combined, config=config)
    history = trainer.train()

    # Save history
    Path(args.checkpoint_dir).mkdir(parents=True, exist_ok=True)
    with open(Path(args.checkpoint_dir) / "training_history.json", "w") as f:
        json.dump(history, f, indent=2)

    best_ckpt = Path(args.checkpoint_dir) / "best_model.pt"
    print(f"\nPre-training complete. Best model → {best_ckpt}")
    return str(best_ckpt)


# ═══════════════════════════════════════════════════════════
# PHASE 2 — Evaluation (per dataset)
# ═══════════════════════════════════════════════════════════
def run_evaluation(args, checkpoint_path: str):
    banner("PHASE 2: Evaluation (Linear Probe + Few-Shot)")

    device = get_device()

    # Load model from checkpoint
    print(f"\nLoading checkpoint: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    ckpt_config = ckpt.get("config", {})

    embed_dim = ckpt_config.get("embed_dim", args.embed_dim)
    encoder_depth = ckpt_config.get("encoder_depth", args.encoder_depth)
    predictor_depth = ckpt_config.get("predictor_depth", args.predictor_depth)

    model = LeJEPA(
        embed_dim=embed_dim,
        encoder_depth=encoder_depth,
        predictor_depth=predictor_depth,
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print("Model loaded!\n")

    all_results = {}

    for name, cfg in DATASETS.items():
        banner(f"Evaluating on: {name.upper()}")

        # Load labeled dataset
        ds = load_dataset(name, cfg, args.image_size, with_labels=True)
        if ds.labels is None:
            print(f"  WARNING: No labels available for {name}, skipping...")
            continue

        # 80/20 train-test split
        train_size = int(0.8 * len(ds))
        test_size = len(ds) - train_size
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

        # ── Linear Probe ─────────────────────────
        print(f"\n[{name}] Linear Probing ...")
        lp = LinearProbeEvaluator(
            pretrained_model=model,
            num_classes=cfg["num_classes"],
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

        # ── Few-Shot / Data Efficiency ────────────
        print(f"\n[{name}] Few-Shot (Data Efficiency) ...")
        fs = FewShotEvaluator(pretrained_model=model)
        fs_results = fs.evaluate_data_efficiency(
            train_feats, train_labels,
            test_feats, test_labels,
            fractions=[0.01, 0.05, 0.1, 0.25, 0.5, 1.0],
        )

        all_results[name] = {
            "linear_probing": {
                "accuracy": lp_results["accuracy"],
                "auc": lp_results.get("auc"),
            },
            "few_shot": fs_results,
        }

    # ── Save everything ───────────────────────────
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    out_path = results_dir / "evaluation_results.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    banner("ALL RESULTS")
    for name, res in all_results.items():
        acc = res["linear_probing"]["accuracy"]
        auc = res["linear_probing"].get("auc", "N/A")
        print(f"  {name:12s}  |  LP Acc: {acc:.4f}  |  AUC: {auc}")
        if res.get("few_shot"):
            for fs in res["few_shot"]:
                frac = fs.get("fraction", "?")
                facc = fs.get("accuracy", 0)
                print(f"        {frac*100:5.1f}% data → Acc: {facc:.4f}")

    print(f"\nResults saved to: {out_path}")
    return all_results


# ═══════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════
def main():
    args = parse_args()
    start = time.time()

    banner("MedJEPA — Full GPU Pipeline")
    get_device_info()

    # Phase 1: Pre-train
    if args.skip_pretrain:
        ckpt = args.checkpoint or str(Path(args.checkpoint_dir) / "best_model.pt")
        print(f"\nSkipping pre-training. Using checkpoint: {ckpt}")
    else:
        ckpt = run_pretraining(args)

    # Phase 2: Evaluate
    run_evaluation(args, ckpt)

    elapsed = time.time() - start
    hours, rem = divmod(elapsed, 3600)
    mins, secs = divmod(rem, 60)
    banner(f"DONE — Total time: {int(hours)}h {int(mins)}m {int(secs)}s")


if __name__ == "__main__":
    main()
