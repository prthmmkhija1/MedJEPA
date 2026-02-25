"""
Extract ChestXray14 archives images_005 through images_012 into 
data/raw/chestxray14/images/ (flat, PNG files only).
Images 001-004 are already extracted (34,999 images).
"""

import tarfile
import os
from pathlib import Path
import time

SRC_DIR = Path("data/raw/chestxray14/CXR8/images")
DEST_DIR = Path("data/raw/chestxray14/images")
DEST_DIR.mkdir(parents=True, exist_ok=True)

# Only extract 005-012 (001-004 already done)
archives = sorted(SRC_DIR.glob("images_0*.tar.gz"))
to_extract = [a for a in archives if int(a.name.split("_")[1].split(".")[0]) >= 5]

print(f"Destination: {DEST_DIR.resolve()}")
print(f"Already extracted: {len(list(DEST_DIR.glob('*.png')))} images")
print(f"Archives to extract: {[a.name for a in to_extract]}")
print()

total_new = 0
for archive in to_extract:
    print(f"Extracting {archive.name} ({archive.stat().st_size / 1e9:.2f} GB)...")
    start = time.time()
    count = 0
    with tarfile.open(archive, "r:gz") as tar:
        for member in tar.getmembers():
            if member.name.endswith(".png") and member.isfile():
                # Flatten: extract just the filename
                basename = os.path.basename(member.name)
                # Check if already exists (skip duplicates)
                dest_path = DEST_DIR / basename
                if dest_path.exists():
                    count += 1
                    continue
                # Extract to temp name then rename
                member.name = basename
                tar.extract(member, DEST_DIR)
                count += 1
    elapsed = time.time() - start
    total_new += count
    print(f"  -> {count} images in {elapsed:.1f}s")

final_count = len(list(DEST_DIR.glob("*.png")))
print(f"\nDone! Total images in {DEST_DIR}: {final_count}")
print(f"Expected: 112,120 images for full ChestX-ray14 dataset")
