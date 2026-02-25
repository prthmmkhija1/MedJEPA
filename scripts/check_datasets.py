"""Check status of all 4 datasets."""
from pathlib import Path

# HAM10000
h1 = len(list(Path("data/raw/ham10000/HAM10000_images_part_1").glob("*.jpg")))
h2 = len(list(Path("data/raw/ham10000/HAM10000_images_part_2").glob("*.jpg")))
meta = Path("data/raw/ham10000/HAM10000_metadata.csv").exists()
print(f"HAM10000: part1={h1}, part2={h2}, total={h1+h2}, metadata={meta}")

# APTOS
a_train = len(list(Path("data/raw/aptos2019/train_images").glob("*.png")))
a_test = len(list(Path("data/raw/aptos2019/test_images").glob("*.png")))
a_csv = Path("data/raw/aptos2019/train.csv").exists()
print(f"APTOS: train={a_train}, test={a_test}, train.csv={a_csv}")

# PCam
p_train = Path("data/raw/pcam/train")
p_test = Path("data/raw/pcam/test")
p_labels = Path("data/raw/pcam/train_labels.csv").exists()
print(f"PCam: train_dir={p_train.exists()}, test_dir={p_test.exists()}, labels={p_labels}")
if p_train.exists():
    tc = len(list(p_train.glob("*.tif")))
    print(f"  train images: {tc}")
if p_test.exists():
    tc2 = len(list(p_test.glob("*.tif")))
    print(f"  test images: {tc2}")

# ChestXray14
cx_img = Path("data/raw/chestxray14/images")
cx_csv = Path("data/raw/chestxray14/Data_Entry_2017_v2020.csv").exists()
print(f"ChestXray14: images_dir={cx_img.exists()}, csv={cx_csv}")
if cx_img.exists():
    cx_count = len(list(cx_img.glob("*.png")))
    print(f"  extracted images: {cx_count}")
tgz = len(list(Path("data/raw/chestxray14/CXR8/images").glob("*.tar.gz")))
print(f"  tar.gz archives available: {tgz}")
