"""
Run all evaluations on a pre-trained MedJEPA model.

Usage:
  python scripts/evaluate.py --checkpoint checkpoints/best_model.pt --dataset ham10000

This will run:
1. Linear probing (accuracy on classification tasks)
2. Few-shot learning (accuracy with 5, 10, 20 examples)
3. Data efficiency analysis (1%, 5%, 10%, 25%, 50%, 100% labeled data)
"""

import argparse
import sys
import json
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
from torch.utils.data import DataLoader, random_split
from medjepa.models.lejepa import LeJEPA
from medjepa.data.datasets import MedicalImageDataset
from medjepa.evaluation.linear_probe import LinearProbeEvaluator
from medjepa.evaluation.few_shot import FewShotEvaluator
from medjepa.utils.device import get_device


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--metadata_csv", type=str, default=None)
    parser.add_argument("--label_column", type=str, default="dx")
    parser.add_argument("--num_classes", type=int, default=7)  # HAM10000 has 7 classes
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--embed_dim", type=int, default=None,
                        help="Override embed_dim (use if checkpoint config lacks it)")
    parser.add_argument("--encoder_depth", type=int, default=None,
                        help="Override encoder_depth (use if checkpoint config lacks it)")
    parser.add_argument("--predictor_depth", type=int, default=None,
                        help="Override predictor_depth (use if checkpoint config lacks it)")
    parser.add_argument("--file_extension", type=str, default=".jpg")
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--num_workers", type=int, default=0,
                        help="DataLoader workers (0 for Windows/HP-Lite, 4+ for Linux/GPU)")
    args = parser.parse_args()

    device = get_device()

    # Load model
    print("Loading pre-trained model...")
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    config = checkpoint.get("config", {})

    # CLI overrides take priority over checkpoint config
    embed_dim = args.embed_dim or config.get("embed_dim", 768)
    encoder_depth = args.encoder_depth or config.get("encoder_depth", 12)
    predictor_depth = args.predictor_depth or config.get("predictor_depth", 6)
    print(f"  embed_dim={embed_dim}, encoder_depth={encoder_depth}, predictor_depth={predictor_depth}")

    model = LeJEPA(
        embed_dim=embed_dim,
        encoder_depth=encoder_depth,
        predictor_depth=predictor_depth,
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    print("Model loaded!")

    # Load labeled dataset
    print("\nLoading dataset...")
    dataset = MedicalImageDataset(
        image_dir=args.data_dir,
        metadata_csv=args.metadata_csv,
        label_column=args.label_column,
        file_extension=args.file_extension,
        target_size=(args.image_size, args.image_size),
    )

    # Split into train (80%) and test (20%)
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size,
        num_workers=args.num_workers, pin_memory=torch.cuda.is_available(),
    )
    test_loader = DataLoader(
        test_dataset, batch_size=args.batch_size,
        num_workers=args.num_workers, pin_memory=torch.cuda.is_available(),
    )

    # ==========================================
    # Evaluation 1: Linear Probing
    # ==========================================
    print("\n" + "=" * 50)
    print("EVALUATION 1: Linear Probing")
    print("=" * 50)

    evaluator = LinearProbeEvaluator(
        pretrained_model=model,
        num_classes=args.num_classes,
        embed_dim=embed_dim,
    )

    print("Extracting features...")
    train_features, train_labels = evaluator.extract_features(train_loader)
    test_features, test_labels = evaluator.extract_features(test_loader)
    print(f"Train: {train_features.shape}, Test: {test_features.shape}")

    print("Training linear probe...")
    evaluator.train_probe(train_features, train_labels)

    print("Evaluating...")
    results_lp = evaluator.evaluate(test_features, test_labels)
    print(f"\nLinear Probe Accuracy: {results_lp['accuracy']:.4f}")
    if results_lp.get("auc"):
        print(f"Linear Probe AUC: {results_lp['auc']:.4f}")

    # ==========================================
    # Evaluation 2: Few-Shot Learning
    # ==========================================
    print("\n" + "=" * 50)
    print("EVALUATION 2: Few-Shot Learning")
    print("=" * 50)

    few_shot = FewShotEvaluator(pretrained_model=model)
    results_fs = few_shot.evaluate_data_efficiency(
        train_features, train_labels,
        test_features, test_labels,
        fractions=[0.01, 0.05, 0.1, 0.25, 0.5, 1.0],
    )

    # ==========================================
    # Save Results
    # ==========================================
    results = {
        "linear_probing": {
            "accuracy": results_lp["accuracy"],
            "auc": results_lp.get("auc"),
        },
        "few_shot": results_fs,
    }

    output_path = Path("results") / "evaluation_results.json"
    output_path.parent.mkdir(exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
