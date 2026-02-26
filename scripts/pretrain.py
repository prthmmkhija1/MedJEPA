"""
Main script to launch pre-training.

Usage:
  python scripts/pretrain.py --dataset ham10000 --batch_size 32 --epochs 50

Run this on HP-GPU, Mac-M4, or Cloud.
On HP-Lite, use --batch_size 4 --epochs 2 for testing only.
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
from medjepa.models.lejepa import LeJEPA
from medjepa.data.datasets import MedicalImageDataset
from medjepa.training.trainer import MedJEPATrainer


def parse_args():
    parser = argparse.ArgumentParser(description="MedJEPA Pre-training")

    # Data
    parser.add_argument("--dataset", type=str, default="ham10000")
    parser.add_argument("--data_dir", type=str, default="data/raw/ham10000")
    parser.add_argument("--file_extension", type=str, default=".jpg")

    # Model
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--patch_size", type=int, default=16)
    parser.add_argument("--embed_dim", type=int, default=768)
    parser.add_argument("--encoder_depth", type=int, default=12)
    parser.add_argument("--predictor_depth", type=int, default=6)
    parser.add_argument("--mask_ratio", type=float, default=0.75)

    # Training
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--lambda_reg", type=float, default=1.0)

    # Hardware
    parser.add_argument("--num_workers", type=int, default=0)
    # Use 0 on Windows, 4 on Mac/Linux

    # Logging
    parser.add_argument("--log_every", type=int, default=10)
    parser.add_argument("--save_every", type=int, default=5)
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints")
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to checkpoint to resume training from")

    return parser.parse_args()


def main():
    args = parse_args()

    print("\n" + "=" * 60)
    print("MedJEPA Pre-Training")
    print("=" * 60)

    # Create dataset
    print(f"\nLoading dataset from: {args.data_dir}")
    dataset = MedicalImageDataset(
        image_dir=args.data_dir,
        file_extension=args.file_extension,
        target_size=(args.image_size, args.image_size),
    )
    print(f"Found {len(dataset)} images")

    # Create model
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
    print(f"Model parameters: {total_params:,} ({total_params/1e6:.1f}M)")

    # Training config (includes model architecture so checkpoints are self-describing)
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

    # Create trainer and start!
    trainer = MedJEPATrainer(
        model=model,
        train_dataset=dataset,
        config=config,
    )

    # Resume from checkpoint if specified
    if args.resume:
        print(f"\nResuming from: {args.resume}")
        trainer.load_checkpoint(args.resume)

    history = trainer.train()

    # Save training history
    import json
    with open(Path(args.checkpoint_dir) / "training_history.json", "w") as f:
        json.dump(history, f, indent=2)

    print("\nDone! Model saved to:", args.checkpoint_dir)


if __name__ == "__main__":
    main()
