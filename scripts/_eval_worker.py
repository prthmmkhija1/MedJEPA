#!/usr/bin/env python3
"""
Subprocess worker for crash-isolated evaluation tasks.

Runs ImageNet baseline or full fine-tuning in a separate process so that
CUDA segfaults / OOM crashes don't kill the main evaluation pipeline.

Called by run_gpu_full.py via subprocess.run().
"""

import argparse
import gc
import json
import os
import sys
from pathlib import Path

# Ensure project root is importable
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset


def run_imagenet_baseline(cfg):
    """Run ImageNet ViT-B/16 baseline evaluation."""
    from medjepa.evaluation.fine_tune import ImageNetBaselineEvaluator

    num_classes = cfg["num_classes"]
    batch_size = cfg["batch_size"]
    num_workers = cfg["num_workers"]
    train_indices = cfg["train_indices"]
    test_indices = cfg["test_indices"]
    image_size = cfg["image_size"]
    max_samples = cfg.get("max_samples")
    multi_label = cfg.get("multi_label", False)
    ds_name = cfg["name"]
    ds_cfg = cfg["dataset_cfg"]

    # Recreate dataset
    ds = _load_dataset(ds_name, ds_cfg, image_size, max_samples)
    train_ds = Subset(ds, train_indices)
    test_ds = Subset(ds, test_indices)

    # num_workers=0: subprocess workers inherit stdout/stderr pipe handles on
    # Windows; if the subprocess is killed (e.g. on timeout), those handles
    # stay open and the parent communicate() deadlocks.  Single-process
    # loading avoids spawning any children inside this subprocess.
    train_loader = DataLoader(
        train_ds, batch_size=batch_size,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )

    inet_eval = ImageNetBaselineEvaluator(
        num_classes=num_classes, backbone="vit_b_16",
        multi_label=multi_label,
    )
    inet_train_feats, inet_train_labs = inet_eval.extract_features(train_loader)
    inet_test_feats, inet_test_labs = inet_eval.extract_features(test_loader)
    inet_eval.train_probe(inet_train_feats, inet_train_labs)
    inet_results = inet_eval.evaluate(inet_test_feats, inet_test_labs)

    print(f"  ImageNet Baseline Accuracy: {inet_results['accuracy']:.4f}")
    if inet_results.get("auc"):
        print(f"  ImageNet Baseline AUC:      {inet_results['auc']:.4f}")

    # Strip non-serializable report
    inet_results.pop("report", None)
    return inet_results


def run_fine_tuning(cfg):
    """Run full fine-tuning evaluation."""
    from medjepa.models.lejepa import LeJEPA
    from medjepa.evaluation.fine_tune import FineTuneEvaluator

    num_classes = cfg["num_classes"]
    batch_size = cfg["batch_size"]
    num_workers = cfg["num_workers"]
    train_indices = cfg["train_indices"]
    test_indices = cfg["test_indices"]
    image_size = cfg["image_size"]
    patch_size = cfg["patch_size"]
    embed_dim = cfg["embed_dim"]
    encoder_depth = cfg["encoder_depth"]
    predictor_depth = cfg["predictor_depth"]
    checkpoint_path = cfg["checkpoint_path"]
    max_samples = cfg.get("max_samples")
    multi_label = cfg.get("multi_label", False)
    ds_name = cfg["name"]
    ds_cfg = cfg["dataset_cfg"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Recreate dataset
    ds = _load_dataset(ds_name, ds_cfg, image_size, max_samples)
    train_ds = Subset(ds, train_indices)
    test_ds = Subset(ds, test_indices)

    # num_workers=0: see note in run_imagenet_baseline above.
    train_loader = DataLoader(
        train_ds, batch_size=batch_size,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )

    # Load checkpoint
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)

    ft_model = LeJEPA(
        image_size=image_size, patch_size=patch_size,
        embed_dim=embed_dim, encoder_depth=encoder_depth,
        predictor_depth=predictor_depth, use_ema=True,
    )
    ft_state = {k: v for k, v in ckpt["model_state_dict"].items()
                if "_sketch_matrix" not in k}
    ft_model.load_state_dict(ft_state, strict=False)
    if "ema_encoder_state_dict" in ckpt:
        ft_model.ema_encoder.load_state_dict(ckpt["ema_encoder_state_dict"])
    ft_model = ft_model.to(device)

    ft_eval = FineTuneEvaluator(
        pretrained_model=ft_model,
        num_classes=num_classes,
        embed_dim=embed_dim,
        num_epochs=20,
        multi_label=multi_label,
    )
    ft_history = ft_eval.train(train_loader, test_loader)
    ft_results = ft_eval.evaluate(test_loader)

    print(f"  Fine-Tune Accuracy: {ft_results['accuracy']:.4f}")
    if ft_results.get("auc"):
        print(f"  Fine-Tune AUC:      {ft_results['auc']:.4f}")

    ft_results.pop("report", None)
    return ft_results


def _load_dataset(name, ds_cfg, image_size, max_samples):
    """Load raw dataset without max_samples subsetting.

    The caller passes absolute indices (resolved through any Subset chain
    from the main process), so we always load the full underlying dataset.
    """
    from medjepa.data.datasets import MedicalImageDataset, ChestXray14Dataset

    ds_type = ds_cfg.get("type", "standard")

    if ds_type == "standard":
        ds = MedicalImageDataset(
            image_dir=ds_cfg["data_dir"],
            metadata_csv=ds_cfg.get("metadata_csv"),
            image_column=ds_cfg.get("image_column", "image"),
            label_column=ds_cfg.get("label_column", "label"),
            file_extension=ds_cfg.get("file_extension", ".jpg"),
            target_size=(image_size, image_size),
        )
    elif ds_type == "chestxray14":
        ds = ChestXray14Dataset(
            data_dir=ds_cfg["data_dir"],
            target_size=(image_size, image_size),
            with_labels=True,
        )
    else:
        raise ValueError(f"Unknown dataset type: {ds_type}")

    return ds


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="JSON config file path")
    parser.add_argument("--result", required=True, help="JSON result file path")
    parser.add_argument("--task", required=True,
                        choices=["imagenet_baseline", "fine_tuning"])
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = json.load(f)

    if args.task == "imagenet_baseline":
        results = run_imagenet_baseline(cfg)
    elif args.task == "fine_tuning":
        results = run_fine_tuning(cfg)
    else:
        results = {"error": f"Unknown task: {args.task}"}

    with open(args.result, "w") as f:
        json.dump(results, f)


if __name__ == "__main__":
    main()
