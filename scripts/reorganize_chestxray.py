"""
Reorganize ChestXray14 so the structure is clean:
  chestxray14/
    images/          <- all PNGs flat
    Data_Entry_2017.csv
    BBox_List_2017.csv
    test_list.txt
    train_val_list.txt
"""
import shutil
from pathlib import Path

base = Path("data/raw/chestxray14")
cxr8 = base / "CXR8"
all_img = base / "all_images"

# 1. Rename all_images -> images
target_images = base / "images"
if all_img.exists() and not target_images.exists():
    all_img.rename(target_images)
    print(f"Renamed all_images -> images")
elif target_images.exists():
    print("images/ already exists")

# 2. Copy key CSVs/txt from CXR8/ up to chestxray14/
files_to_move = [
    "Data_Entry_2017_v2020.csv",
    "BBox_List_2017.csv",
    "test_list.txt",
    "train_val_list.txt",
    "README_CHESTXRAY.pdf",
]

for fname in files_to_move:
    src = cxr8 / fname
    dst = base / fname
    if src.exists() and not dst.exists():
        shutil.copy2(src, dst)
        print(f"Copied {fname}")
    elif dst.exists():
        print(f"{fname} already in place")
    else:
        print(f"{fname} not found in CXR8/")

# 3. Count final images
img_count = len(list(target_images.glob("*.png"))) if target_images.exists() else 0
print(f"\nFinal: {img_count} images in chestxray14/images/")
print("DONE!")
