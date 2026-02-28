#!/usr/bin/env python3
"""
Download all 6 datasets for MedJEPA.

Datasets:
  2D: HAM10000, APTOS 2019, PatchCamelyon, ChestX-ray14
  3D: BraTS 2021, Medical Segmentation Decathlon

Usage:
    python scripts/download_data.py            # download all
    python scripts/download_data.py --only ham  # download just one

Requirements:
    pip install kaggle
    Set up Kaggle API credentials (see below).

Kaggle API setup (one-time):
    1. Go to https://www.kaggle.com/settings  ->  API  ->  Create New Token
    2. This downloads kaggle.json to your local machine
    3. On the GPU server, run:
          mkdir -p ~/.kaggle
          mv kaggle.json ~/.kaggle/kaggle.json
          chmod 600 ~/.kaggle/kaggle.json
"""

import os
import subprocess
import sys
import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_RAW = ROOT / "data" / "raw"


def run(cmd: str):
    print(f"\n$ {cmd}")
    ret = subprocess.run(cmd, shell=True)
    if ret.returncode != 0:
        print(f"WARNING: command failed with exit code {ret.returncode}")
        return False
    return True


# ══════════════════════════════════════════════════════
# 2D Datasets
# ══════════════════════════════════════════════════════

def download_ham10000():
    """HAM10000 — 10,015 dermatoscopic images, 7 skin lesion types."""
    out = DATA_RAW / "ham10000"
    out.mkdir(parents=True, exist_ok=True)
    print("\n[1/6] Downloading HAM10000 (skin lesion)...")
    run(f"kaggle datasets download -d kmader/skin-cancer-mnist-ham10000 -p {out} --unzip")
    part1 = out / "HAM10000_images_part_1"
    part2 = out / "HAM10000_images_part_2"
    part1.mkdir(exist_ok=True)
    part2.mkdir(exist_ok=True)
    print(f"  Done -> {out}")


def download_aptos():
    """APTOS 2019 — 3,662 retinal images, 5 diabetic retinopathy grades."""
    out = DATA_RAW / "aptos2019"
    out.mkdir(parents=True, exist_ok=True)
    print("\n[2/6] Downloading APTOS 2019 (diabetic retinopathy)...")
    run(f"kaggle competitions download -c aptos2019-blindness-detection -p {out}")
    import glob
    for z in glob.glob(str(out / "*.zip")):
        run(f"unzip -o {z} -d {out}")
        os.remove(z)
    print(f"  Done -> {out}")


def download_pcam():
    """PatchCamelyon — 327,680 histopathology patches, binary cancer detection."""
    out = DATA_RAW / "pcam"
    out.mkdir(parents=True, exist_ok=True)
    print("\n[3/6] Downloading PatchCamelyon (histopathology)...")
    run(f"kaggle competitions download -c histopathologic-cancer-detection -p {out}")
    import glob
    for z in glob.glob(str(out / "*.zip")):
        run(f"unzip -o {z} -d {out}")
        os.remove(z)
    print(f"  Done -> {out}")


def download_chestxray14():
    """
    NIH ChestX-ray14 — 112,120 frontal X-rays, 14 disease labels.
    Downloads from Kaggle: nih-chest-xrays/data
    """
    out = DATA_RAW / "chestxray14"
    out.mkdir(parents=True, exist_ok=True)
    images_dir = out / "images"
    images_dir.mkdir(exist_ok=True)
    print("\n[4/6] Downloading ChestX-ray14 (chest X-rays)...")

    # Try Kaggle dataset (most reliable)
    success = run(
        f"kaggle datasets download -d nih-chest-xrays/data -p {out} --unzip"
    )
    if not success:
        # Fallback: sample dataset
        print("  Full dataset download failed. Trying sample dataset...")
        run(
            f"kaggle datasets download -d nih-chest-xrays/sample -p {out} --unzip"
        )

    # The Kaggle download puts images in images_001/ through images_012/ subfolders.
    # Flatten them into images/ for easier access
    for subdir in sorted(out.glob("images_*")):
        if subdir.is_dir():
            for img in subdir.glob("images/*.png"):
                dest = images_dir / img.name
                if not dest.exists():
                    img.rename(dest)
            # Clean up empty dirs
            try:
                import shutil
                shutil.rmtree(subdir)
            except Exception:
                pass

    print(f"  Done -> {out}")


# ══════════════════════════════════════════════════════
# 3D Datasets
# ══════════════════════════════════════════════════════

def download_brats():
    """
    BraTS 2021 — Brain Tumor Segmentation.
    Multi-parametric MRI: FLAIR, T1, T1ce, T2 + segmentation masks.
    Downloads from Kaggle: dschettler8845/brats-2021-task1
    """
    out = DATA_RAW / "brats"
    out.mkdir(parents=True, exist_ok=True)
    print("\n[5/6] Downloading BraTS 2021 (brain tumor segmentation)...")

    success = run(
        f"kaggle datasets download -d dschettler8845/brats-2021-task1 -p {out} --unzip"
    )
    if not success:
        print("  BraTS download failed. Trying alternative source...")
        run(
            f"kaggle datasets download -d awsaf49/brats2020-training-data -p {out} --unzip"
        )

    # Check that we have subject folders
    subjects = [d for d in out.iterdir() if d.is_dir() and "BraTS" in d.name]
    print(f"  Found {len(subjects)} BraTS subjects")
    print(f"  Done -> {out}")


def download_decathlon():
    """
    Medical Segmentation Decathlon — 10 segmentation tasks.
    Downloads from: http://medicaldecathlon.com/ (via Google Drive) or Kaggle.

    Available tasks (locally downloaded will be detected):
        Task01_BrainTumour, Task02_Heart, Task03_Liver, Task04_Hippocampus,
        Task05_Prostate, Task06_Lung, Task07_Pancreas, Task08_HepaticVessel,
        Task09_Spleen, Task10_Colon
    """
    out = DATA_RAW / "decathlon"
    out.mkdir(parents=True, exist_ok=True)
    print("\n[6/6] Downloading Medical Segmentation Decathlon...")

    # Try to download from Kaggle (community dataset uploads)
    kaggle_sources = [
        "awsaf49/medical-segmentation-decathlon",
        "aryashah2k/medical-segmentation-decathlon-dataset",
    ]

    downloaded = False
    for src in kaggle_sources:
        print(f"  Trying Kaggle source: {src}")
        if run(f"kaggle datasets download -d {src} -p {out} --unzip"):
            downloaded = True
            break

    if not downloaded:
        # Try individual task downloads via gdown (Google Drive)
        print("  Kaggle failed. Trying direct download via gdown...")
        try:
            import importlib
            importlib.import_module("gdown")
        except ImportError:
            run("pip install gdown")

        # Google Drive IDs for Decathlon tasks (official mirrors)
        gdrive_tasks = {
            "Task02_Heart":      "1wEB2I6S6tQBVEPxir8cA7kFB7go4hBME",
            "Task04_Hippocampus": "1RzPB1_bqzQhlWvU-YGvZzhx2omcDh38C",
            "Task05_Prostate":   "1Ff7c21UksxyT4JfETjaarmuKEjdqe1-a",
            "Task09_Spleen":     "1jzeNU1EKnK81PyTsrx0ujfNl-t0Jo8uE",
        }
        for task_name, gdrive_id in gdrive_tasks.items():
            task_dir = out / task_name
            if task_dir.exists() and (task_dir / "dataset.json").exists():
                print(f"  {task_name} already exists, skipping.")
                continue
            print(f"  Downloading {task_name}...")
            tar_path = out / f"{task_name}.tar"
            run(f"gdown --id {gdrive_id} -O {tar_path}")
            if tar_path.exists():
                run(f"tar -xf {tar_path} -C {out}")
                tar_path.unlink()

    # Report what we have
    tasks = [d.name for d in out.iterdir() if d.is_dir() and d.name.startswith("Task")]
    print(f"  Found {len(tasks)} Decathlon tasks: {', '.join(sorted(tasks))}")
    print(f"  Done -> {out}")


# ══════════════════════════════════════════════════════
# Setup & main
# ══════════════════════════════════════════════════════

def check_kaggle():
    ret = subprocess.run("kaggle --version", shell=True, capture_output=True)
    if ret.returncode != 0:
        print("kaggle not found. Installing...")
        subprocess.run("pip install kaggle", shell=True, check=True)

    creds = Path.home() / ".kaggle" / "kaggle.json"
    if not creds.exists():
        # Windows fallback
        creds_win = Path(os.environ.get("USERPROFILE", "")) / ".kaggle" / "kaggle.json"
        if not creds_win.exists():
            print("\nWARNING: Kaggle credentials not found.")
            print("  1. Go to https://www.kaggle.com/settings -> API -> Create New Token")
            print("  2. Place kaggle.json in ~/.kaggle/kaggle.json")
            print("  Downloads requiring Kaggle API will fail.\n")


def main():
    parser = argparse.ArgumentParser(description="Download MedJEPA datasets")
    parser.add_argument(
        "--only",
        type=str,
        default=None,
        help="Download only this dataset: ham|aptos|pcam|chestxray|brats|decathlon",
    )
    args = parser.parse_args()

    check_kaggle()

    downloaders = {
        "ham":       download_ham10000,
        "aptos":     download_aptos,
        "pcam":      download_pcam,
        "chestxray": download_chestxray14,
        "brats":     download_brats,
        "decathlon": download_decathlon,
    }

    if args.only:
        key = args.only.lower()
        if key not in downloaders:
            print(f"Unknown dataset: {key}")
            print(f"Available: {', '.join(downloaders.keys())}")
            sys.exit(1)
        downloaders[key]()
    else:
        for fn in downloaders.values():
            fn()

    print("\n" + "=" * 50)
    print("Dataset download complete!")
    print(f"Location: {DATA_RAW}")
    print("=" * 50)
    print("\nNext step:")
    print("  python scripts/run_gpu_full.py")


if __name__ == "__main__":
    main()
