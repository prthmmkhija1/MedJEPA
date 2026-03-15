#!/usr/bin/env python3
"""
Download datasets for MedJEPA.

Datasets (2D):
  HAM10000       — 10,015 dermatoscopic images, 7 skin lesion types
  APTOS 2019     — 3,662 retinal images, 5 diabetic retinopathy grades
  PatchCamelyon  — 327,680 histopathology patches, binary cancer detection
  ChestX-ray14   — 112,120 frontal X-rays, 14 disease labels (~42 GB)
  ISIC 2019      — 25,331 dermoscopic images, 8 categories
  CheXpert       — 224,316 chest radiographs (small version from Kaggle)

Datasets (3D):
  BraTS 2021     — 1,251 multi-parametric brain MRI volumes
  Decathlon      — 10 segmentation tasks (multi-organ CT/MRI)

Usage:
    python scripts/download_data.py --small     # HAM10000 only (~3 GB, Kaggle-safe)
    python scripts/download_data.py --only ham  # download just one
    python scripts/download_data.py             # all except ChestXray14
    python scripts/download_data.py --all       # include ChestXray14 (~42 GB)

Kaggle API setup (one-time, not needed inside Kaggle notebooks):
    1. Go to https://www.kaggle.com/settings  ->  API  ->  Create New Token
    2. Place kaggle.json in ~/.kaggle/kaggle.json
    3. chmod 600 ~/.kaggle/kaggle.json
"""

import os
import subprocess
import sys
import argparse
import glob
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_RAW = ROOT / "data" / "raw"


def run(cmd: str):
    """Run a shell command, return True on success."""
    print(f"\n$ {cmd}")
    ret = subprocess.run(cmd, shell=True)
    if ret.returncode != 0:
        print(f"  WARNING: command failed (exit code {ret.returncode})")
        return False
    return True


def kaggle_download(slug: str, out: Path, is_competition=False):
    """Try to download from Kaggle. Returns True on success."""
    kind = "competitions" if is_competition else "datasets"
    return run(f"kaggle {kind} download -d {slug} -p {out} --unzip")


def try_sources(sources: list, out: Path):
    """Try multiple Kaggle sources in order. Returns True if any succeeds."""
    for slug, is_comp in sources:
        label = "competition" if is_comp else "dataset"
        print(f"  Trying {label}: {slug}")
        if kaggle_download(slug, out, is_competition=is_comp):
            return True
        print(f"  Failed, trying next source...")
    return False


# ══════════════════════════════════════════════════════
# 2D Datasets
# ══════════════════════════════════════════════════════

def download_ham10000():
    """HAM10000 — 10,015 dermatoscopic images, 7 skin lesion types (~3 GB)."""
    out = DATA_RAW / "ham10000"
    out.mkdir(parents=True, exist_ok=True)
    print("\n[1/8] Downloading HAM10000 (skin lesion)...")

    sources = [
        ("kmader/skin-cancer-mnist-ham10000", False),
    ]
    if not try_sources(sources, out):
        print("  ERROR: HAM10000 download failed.")
        return

    part1 = out / "HAM10000_images_part_1"
    part2 = out / "HAM10000_images_part_2"
    part1.mkdir(exist_ok=True)
    part2.mkdir(exist_ok=True)
    print(f"  Done -> {out}")


def download_aptos():
    """APTOS 2019 — 3,662 retinal images, 5 diabetic retinopathy grades (~3 GB).

    Sources tried in order:
      1. Dataset mirror (no competition rules needed)
      2. Original competition (needs rules accepted on kaggle.com first)
    """
    out = DATA_RAW / "aptos2019"
    out.mkdir(parents=True, exist_ok=True)
    print("\n[2/8] Downloading APTOS 2019 (diabetic retinopathy)...")

    sources = [
        # Dataset mirrors — no 403 issues
        ("benjaminwarner/aptos2019-blindness-detection", False),
        ("mariaherrerot/aptos2019", False),
        # Original competition — may 403 if rules not accepted
        ("aptos2019-blindness-detection", True),
    ]
    if not try_sources(sources, out):
        print("  ERROR: APTOS download failed. Manual fix:")
        print("    1. Go to https://www.kaggle.com/competitions/aptos2019-blindness-detection/rules")
        print("    2. Accept the rules")
        print("    3. Re-run this script")
        return

    # Unzip any inner zips (competition downloads are double-zipped)
    for z in glob.glob(str(out / "*.zip")):
        run(f"unzip -o -q {z} -d {out}")
        os.remove(z)
    print(f"  Done -> {out}")


def download_pcam():
    """PatchCamelyon — 327,680 histopathology patches, binary cancer (~8 GB).

    Sources tried in order:
      1. Dataset uploads (no competition rules needed)
      2. Original competition (needs rules accepted on kaggle.com first)
    """
    out = DATA_RAW / "pcam"
    out.mkdir(parents=True, exist_ok=True)
    print("\n[3/8] Downloading PatchCamelyon (histopathology)...")

    sources = [
        # Dataset mirrors — no 403 issues
        ("andrewmvd/metastatic-tissue-classification-patchcamelyon", False),
        # Original competition — may 403
        ("histopathologic-cancer-detection", True),
    ]
    if not try_sources(sources, out):
        print("  ERROR: PCam download failed. Manual fix:")
        print("    1. Go to https://www.kaggle.com/competitions/histopathologic-cancer-detection/rules")
        print("    2. Accept the rules")
        print("    3. Re-run this script")
        return

    # Unzip any inner zips
    for z in glob.glob(str(out / "*.zip")):
        run(f"unzip -o -q {z} -d {out}")
        os.remove(z)

    # The dataset mirror puts files in a subfolder; flatten if needed
    subfolder = out / "metastatic-tissue-classification-patchcamelyon"
    if subfolder.is_dir():
        import shutil
        for item in subfolder.iterdir():
            dest = out / item.name
            if not dest.exists():
                shutil.move(str(item), str(dest))
        shutil.rmtree(subfolder, ignore_errors=True)

    print(f"  Done -> {out}")


def download_chestxray14():
    """
    NIH ChestX-ray14 — 112,120 frontal X-rays, 14 disease labels (~42 GB).
    WARNING: Very large. Skipped by default. Use --all or --only chestxray.
    """
    out = DATA_RAW / "chestxray14"
    out.mkdir(parents=True, exist_ok=True)
    images_dir = out / "images"
    images_dir.mkdir(exist_ok=True)
    print("\n[4/8] Downloading ChestX-ray14 (chest X-rays, ~42 GB)...")

    sources = [
        ("nih-chest-xrays/data", False),
        ("nih-chest-xrays/sample", False),  # fallback: 5,606 image sample
    ]
    if not try_sources(sources, out):
        print("  ERROR: ChestXray14 download failed.")
        return

    # Flatten images_001/…/images_012/ subfolders into images/
    for subdir in sorted(out.glob("images_*")):
        if subdir.is_dir():
            for img in subdir.glob("images/*.png"):
                dest = images_dir / img.name
                if not dest.exists():
                    img.rename(dest)
            try:
                import shutil
                shutil.rmtree(subdir)
            except Exception:
                pass

    print(f"  Done -> {out}")


def download_isic():
    """ISIC 2019 — 25,331 dermoscopic images, 8 diagnostic categories (~9 GB)."""
    out = DATA_RAW / "isic2019"
    out.mkdir(parents=True, exist_ok=True)
    print("\n[5/8] Downloading ISIC 2019 (dermoscopy)...")

    sources = [
        ("andrewmvd/isic-2019", False),
        ("nroman/2019-dermatology-concepts-for-skin-cancer-detection", False),
    ]
    if not try_sources(sources, out):
        print("  ERROR: ISIC download failed.")
        return

    print(f"  Done -> {out}")


def download_chexpert():
    """CheXpert — small version (~11k images) from Kaggle.
    For the full 224k dataset, apply at:
    https://stanfordaimi.azurewebsites.net/datasets/8cbd9ed4-2eb9-4565-affc-111cf4f7ebe2
    """
    out = DATA_RAW / "chexpert"
    out.mkdir(parents=True, exist_ok=True)
    print("\n[6/8] Downloading CheXpert-small (chest X-ray)...")

    sources = [
        ("ashery/chexpert", False),
        ("amanullahasraf/covid19-pneumonia-normal-chest-xray-pa-dataset", False),
    ]
    if not try_sources(sources, out):
        print("  ERROR: CheXpert download failed.")
        return

    print(f"  Done -> {out}")


# ══════════════════════════════════════════════════════
# 3D Datasets
# ══════════════════════════════════════════════════════

def download_brats():
    """
    BraTS 2021 — Brain Tumor Segmentation (~2-5 GB).
    Multi-parametric MRI: FLAIR, T1, T1ce, T2 + segmentation masks.
    """
    out = DATA_RAW / "brats"
    out.mkdir(parents=True, exist_ok=True)
    print("\n[7/8] Downloading BraTS 2021 (brain tumor segmentation)...")

    sources = [
        ("dschettler8845/brats-2021-task1", False),
        ("awsaf49/brats2020-training-data", False),
    ]
    if not try_sources(sources, out):
        print("  ERROR: BraTS download failed.")
        return

    subjects = [d for d in out.iterdir() if d.is_dir() and "BraTS" in d.name]
    print(f"  Found {len(subjects)} BraTS subjects")
    print(f"  Done -> {out}")


def download_decathlon():
    """
    Medical Segmentation Decathlon — 10 segmentation tasks.
    Source: http://medicaldecathlon.com/ (via Google Drive) or Kaggle.
    """
    out = DATA_RAW / "decathlon"
    out.mkdir(parents=True, exist_ok=True)
    print("\n[8/8] Downloading Medical Segmentation Decathlon...")

    # Try Kaggle first
    kaggle_sources = [
        ("awsaf49/medical-segmentation-decathlon", False),
        ("aryashah2k/medical-segmentation-decathlon-dataset", False),
    ]
    if try_sources(kaggle_sources, out):
        tasks = [d.name for d in out.iterdir() if d.is_dir() and d.name.startswith("Task")]
        print(f"  Found {len(tasks)} Decathlon tasks: {', '.join(sorted(tasks))}")
        print(f"  Done -> {out}")
        return

    # Fallback: download individual tasks via gdown (Google Drive)
    print("  Kaggle failed. Trying direct download via gdown...")
    try:
        import importlib
        importlib.import_module("gdown")
    except ImportError:
        run("pip install gdown")

    # Google Drive IDs for Decathlon tasks (official mirrors from medicaldecathlon.com)
    gdrive_tasks = {
        "Task01_BrainTumour":   "1A2IU8Sgea1h3fYLpYtFb2v7NYdMjW765",
        "Task02_Heart":         "1wEB2I6S6tQBVEPxir8cA7kFB7go4hBME",
        "Task03_Liver":         "1jyVGmCPkFkBJeZ8xEinCtaY0HzN1HmBy",
        "Task04_Hippocampus":   "1RzPB1_bqzQhlWvU-YGvZzhx2omcDh38C",
        "Task05_Prostate":      "1Ff7c21UksxyT4JfETjaarmuKEjdqe1-a",
        "Task06_Lung":          "1I1LR7XjyEZ-VBQ-Xruh31V7xExMjlVvi",
        "Task07_Pancreas":      "1YZQFAoXXCSS-DeRwZaauaRTLvyCFkPf8",
        "Task08_HepaticVessel":  "1qVrpV7vmhIsUxFiH189LmAn0ALbAPrgS",
        "Task09_Spleen":        "1jzeNU1EKnK81PyTsrx0ujfNl-t0Jo8uE",
        "Task10_Colon":         "1m7tMpE9qEcQGQjL_BdMD-Mvgmc44hG1Y",
    }
    for task_name, gdrive_id in gdrive_tasks.items():
        task_dir = out / task_name
        if task_dir.exists() and (task_dir / "dataset.json").exists():
            print(f"  {task_name} already exists, skipping.")
            continue
        tar_path = out / f"{task_name}.tar"
        # Verify existing tar if present
        if tar_path.exists():
            result = subprocess.run(
                f"tar -tf {tar_path} > /dev/null 2>&1", shell=True
            )
            if result.returncode != 0:
                print(f"  {task_name}.tar is corrupted — removing...")
                tar_path.unlink()
        # Download if needed
        if not tar_path.exists():
            print(f"  Downloading {task_name}...")
            run(f"gdown --id {gdrive_id} -O {tar_path} --fuzzy")
        # Extract
        if tar_path.exists():
            verify = subprocess.run(
                f"tar -tf {tar_path} > /dev/null 2>&1", shell=True
            )
            if verify.returncode == 0:
                run(f"tar -xf {tar_path} -C {out}")
                tar_path.unlink()
            else:
                print(f"  ERROR: {task_name}.tar failed integrity check.")
                print(f"  Manual: gdown --id {gdrive_id} -O {tar_path} --fuzzy")

    tasks = [d.name for d in out.iterdir() if d.is_dir() and d.name.startswith("Task")]
    print(f"  Found {len(tasks)} Decathlon tasks: {', '.join(sorted(tasks))}")
    print(f"  Done -> {out}")


# ══════════════════════════════════════════════════════
# Setup & main
# ══════════════════════════════════════════════════════

def check_kaggle():
    """Make sure the Kaggle CLI is installed and credentials exist."""
    ret = subprocess.run("kaggle --version", shell=True, capture_output=True)
    if ret.returncode != 0:
        print("kaggle CLI not found. Installing...")
        subprocess.run([sys.executable, "-m", "pip", "install", "kaggle"], check=True)

    creds = Path.home() / ".kaggle" / "kaggle.json"
    if not creds.exists():
        creds_win = Path(os.environ.get("USERPROFILE", "")) / ".kaggle" / "kaggle.json"
        if not creds_win.exists():
            print("\nWARNING: Kaggle credentials not found.")
            print("  1. Go to https://www.kaggle.com/settings -> API -> Create New Token")
            print("  2. Place kaggle.json in ~/.kaggle/kaggle.json")
            print("  (Inside Kaggle notebooks, credentials are automatic.)\n")


def main():
    parser = argparse.ArgumentParser(
        description="Download MedJEPA datasets",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Available datasets:
  ham        HAM10000 — skin lesions (7 classes, ~3 GB)
  aptos     APTOS 2019 — retinal images (5 grades, ~3 GB)
  pcam       PatchCamelyon — histopathology (binary, ~8 GB)
  chestxray  ChestX-ray14 — chest X-rays (14 labels, ~42 GB, skipped by default)
  isic       ISIC 2019 — dermoscopy (8 categories, ~9 GB)
  chexpert   CheXpert-small — chest X-rays (~1 GB)
  brats      BraTS 2021 — brain MRI (segmentation, ~2-5 GB)
  decathlon  Med. Seg. Decathlon — 10 tasks (variable size)
        """,
    )
    parser.add_argument(
        "--only", type=str, default=None,
        help="Download only this dataset (see list above)",
    )
    parser.add_argument(
        "--all", action="store_true",
        help="Include ChestXray14 (~42 GB). Without this flag it is skipped.",
    )
    parser.add_argument(
        "--small", action="store_true",
        help="Kaggle-safe preset: HAM10000 only (~3 GB). Fits in 20 GB disk.",
    )
    args = parser.parse_args()

    check_kaggle()

    # Default set: everything except the 42 GB ChestXray14
    downloaders_default = {
        "ham":       download_ham10000,
        "aptos":     download_aptos,
        "pcam":      download_pcam,
        "isic":      download_isic,
        "chexpert":  download_chexpert,
        "brats":     download_brats,
        "decathlon": download_decathlon,
    }

    downloaders_all = {
        **downloaders_default,
        "chestxray": download_chestxray14,
    }

    if args.small:
        # Only HAM10000 — safe for Kaggle's 20 GB disk after code+packages take ~4 GB
        print("NOTE: --small mode: downloading HAM10000 only (~3 GB).")
        download_ham10000()
    elif args.only:
        key = args.only.lower()
        if key not in downloaders_all:
            print(f"Unknown dataset: {key}")
            print(f"Available: {', '.join(downloaders_all.keys())}")
            sys.exit(1)
        downloaders_all[key]()
    elif args.all:
        for fn in downloaders_all.values():
            fn()
    else:
        # Skip ChestXray14 by default (too large for Kaggle's ~70 GB disk)
        print("NOTE: Skipping ChestXray14 (~42 GB). Use --all to include it.")
        for fn in downloaders_default.values():
            fn()

    print("\n" + "=" * 50)
    print("Dataset download complete!")
    print(f"Location: {DATA_RAW}")
    print("=" * 50)

    # Report what we have
    if DATA_RAW.exists():
        print("\nAvailable datasets:")
        for d in sorted(DATA_RAW.iterdir()):
            if d.is_dir():
                n = sum(len(f) for _, _, f in os.walk(d))
                print(f"  {d.name:30s}  ({n:,} files)")

    print("\nNext step:")
    print("  python scripts/run_gpu_full.py")


if __name__ == "__main__":
    main()
