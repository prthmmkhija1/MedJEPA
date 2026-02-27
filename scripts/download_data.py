#!/usr/bin/env python3
"""
Download all three datasets from Kaggle.

Usage:
    python scripts/download_data.py

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
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_RAW = ROOT / "data" / "raw"


def run(cmd: str):
    print(f"\n$ {cmd}")
    ret = subprocess.run(cmd, shell=True)
    if ret.returncode != 0:
        print(f"ERROR: command failed with exit code {ret.returncode}")
        sys.exit(1)


def download_ham10000():
    out = DATA_RAW / "ham10000"
    out.mkdir(parents=True, exist_ok=True)
    print("\n[1/3] Downloading HAM10000 (skin lesion)...")
    run(f"kaggle datasets download -d kmader/skin-cancer-mnist-ham10000 -p {out} --unzip")
    # Rename images folder to expected names if needed
    part1 = out / "HAM10000_images_part_1"
    part2 = out / "HAM10000_images_part_2"
    part1.mkdir(exist_ok=True)
    part2.mkdir(exist_ok=True)
    print(f"  Done -> {out}")


def download_aptos():
    out = DATA_RAW / "aptos2019"
    out.mkdir(parents=True, exist_ok=True)
    print("\n[2/3] Downloading APTOS 2019 (diabetic retinopathy)...")
    run(f"kaggle competitions download -c aptos2019-blindness-detection -p {out}")
    # Unzip
    import glob
    for z in glob.glob(str(out / "*.zip")):
        run(f"unzip -o {z} -d {out}")
        os.remove(z)
    print(f"  Done -> {out}")


def download_pcam():
    out = DATA_RAW / "pcam"
    out.mkdir(parents=True, exist_ok=True)
    print("\n[3/3] Downloading PatchCamelyon / PCam (histopathology)...")
    run(f"kaggle competitions download -c histopathologic-cancer-detection -p {out}")
    import glob
    for z in glob.glob(str(out / "*.zip")):
        run(f"unzip -o {z} -d {out}")
        os.remove(z)
    print(f"  Done -> {out}")


def check_kaggle():
    ret = subprocess.run("kaggle --version", shell=True, capture_output=True)
    if ret.returncode != 0:
        print("kaggle not found. Installing...")
        subprocess.run("pip install kaggle", shell=True, check=True)

    creds = Path.home() / ".kaggle" / "kaggle.json"
    if not creds.exists():
        print("\nERROR: Kaggle credentials not found.")
        print("  1. Go to https://www.kaggle.com/settings -> API -> Create New Token")
        print("  2. Upload kaggle.json to the GPU server")
        print("  3. Run:")
        print("       mkdir -p ~/.kaggle")
        print("       mv kaggle.json ~/.kaggle/kaggle.json")
        print("       chmod 600 ~/.kaggle/kaggle.json")
        print("  Then re-run this script.")
        sys.exit(1)


def main():
    check_kaggle()
    download_ham10000()
    download_aptos()
    download_pcam()

    print("\n" + "=" * 50)
    print("All datasets downloaded!")
    print(f"Location: {DATA_RAW}")
    print("=" * 50)
    print("\nNow run:")
    print("  python scripts/run_gpu_full.py")


if __name__ == "__main__":
    main()
