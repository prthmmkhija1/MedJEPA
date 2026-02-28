"""
Detect and remove corrupted image files from datasets.

Run BEFORE training to avoid crashes mid-epoch.

Usage:
    # Dry run (just report, don't delete):
    python scripts/clean_corrupted_images.py

    # Actually delete corrupted files:
    python scripts/clean_corrupted_images.py --delete

    # Scan a specific folder only:
    python scripts/clean_corrupted_images.py --data-dir data/raw/aptos2019/train_images

    # Move corrupted files to a quarantine folder instead of deleting:
    python scripts/clean_corrupted_images.py --quarantine
"""

import argparse
import shutil
import sys
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import List, Tuple, Optional

# ---------------------------------------------------------------------------
# Image extensions to scan
# ---------------------------------------------------------------------------
IMAGE_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff",
    ".gif", ".webp",
}

# NIfTI / DICOM handled separately
NIFTI_EXTENSIONS = {".nii", ".gz"}   # .nii and .nii.gz
DICOM_EXTENSIONS = {".dcm"}


def check_standard_image(path: str) -> Tuple[str, Optional[str]]:
    """
    Try to fully load a standard image file.
    Returns (path, error_message) — error_message is None if OK.
    """
    from PIL import Image
    try:
        with Image.open(path) as img:
            img.verify()                # lightweight structural check
        # verify() can miss some issues, so re-open and force full decode
        with Image.open(path) as img:
            img.load()                  # actually read all pixel data
        return (path, None)
    except Exception as e:
        return (path, str(e))


def check_nifti_file(path: str) -> Tuple[str, Optional[str]]:
    """Check a NIfTI (.nii / .nii.gz) file."""
    try:
        import nibabel as nib
        nii = nib.load(path)
        _ = nii.get_fdata()             # force full read
        return (path, None)
    except Exception as e:
        return (path, str(e))


def check_dicom_file(path: str) -> Tuple[str, Optional[str]]:
    """Check a DICOM (.dcm) file."""
    try:
        import pydicom
        ds = pydicom.dcmread(path)
        _ = ds.pixel_array              # force pixel decode
        return (path, None)
    except Exception as e:
        return (path, str(e))


def gather_files(data_dir: Path) -> dict:
    """Collect all scannable files grouped by type."""
    files = {"standard": [], "nifti": [], "dicom": []}
    for p in data_dir.rglob("*"):
        if not p.is_file():
            continue
        suffix = p.suffix.lower()
        if suffix in IMAGE_EXTENSIONS:
            files["standard"].append(str(p))
        elif suffix in DICOM_EXTENSIONS:
            files["dicom"].append(str(p))
        elif suffix == ".nii" or p.name.endswith(".nii.gz"):
            files["nifti"].append(str(p))
    return files


def scan_files(
    file_list: List[str],
    check_fn,
    workers: int,
    desc: str,
) -> List[Tuple[str, str]]:
    """
    Run check_fn on every file in file_list in parallel.
    Returns list of (path, error) for corrupted files only.
    """
    corrupted = []
    total = len(file_list)
    if total == 0:
        return corrupted

    print(f"\n  Scanning {total} {desc} files ({workers} workers)...")
    done = 0

    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(check_fn, f): f for f in file_list}
        for future in as_completed(futures):
            done += 1
            path, err = future.result()
            if err is not None:
                corrupted.append((path, err))
            # progress
            if done % 500 == 0 or done == total:
                print(f"    [{done}/{total}]  corrupted so far: {len(corrupted)}")

    return corrupted


def main():
    parser = argparse.ArgumentParser(
        description="Find (and optionally remove) corrupted images from datasets."
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data/raw",
        help="Root folder to scan (default: data/raw)",
    )
    parser.add_argument(
        "--delete",
        action="store_true",
        help="Actually delete corrupted files (default: dry-run only).",
    )
    parser.add_argument(
        "--quarantine",
        action="store_true",
        help="Move corrupted files to data/corrupted/ instead of deleting.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Number of parallel workers (default: 4).",
    )
    parser.add_argument(
        "--skip-nifti",
        action="store_true",
        help="Skip NIfTI files (they are slow to fully decode).",
    )
    parser.add_argument(
        "--skip-dicom",
        action="store_true",
        help="Skip DICOM files.",
    )
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        print(f"ERROR: {data_dir} does not exist.")
        sys.exit(1)

    print(f"=== Corrupted Image Scanner ===")
    print(f"Scanning: {data_dir.resolve()}")
    print(f"Mode:     {'DELETE' if args.delete else 'QUARANTINE' if args.quarantine else 'DRY RUN (no files will be touched)'}")

    # ---- gather files ----
    files = gather_files(data_dir)
    total_files = sum(len(v) for v in files.values())
    print(f"\nFound {total_files} files to check:")
    print(f"  Standard images : {len(files['standard'])}")
    print(f"  DICOM files     : {len(files['dicom'])}")
    print(f"  NIfTI files     : {len(files['nifti'])}")

    # ---- scan ----
    all_corrupted: List[Tuple[str, str]] = []

    all_corrupted.extend(
        scan_files(files["standard"], check_standard_image, args.workers, "standard image")
    )

    if not args.skip_dicom and files["dicom"]:
        all_corrupted.extend(
            scan_files(files["dicom"], check_dicom_file, args.workers, "DICOM")
        )

    if not args.skip_nifti and files["nifti"]:
        all_corrupted.extend(
            scan_files(files["nifti"], check_nifti_file, args.workers, "NIfTI")
        )

    # ---- report ----
    print(f"\n{'='*60}")
    print(f"RESULT: {len(all_corrupted)} corrupted file(s) found out of {total_files}")
    print(f"{'='*60}")

    if not all_corrupted:
        print("All files are clean!")
        return

    for path, err in all_corrupted:
        print(f"  CORRUPTED: {path}")
        print(f"             Reason: {err}")

    # ---- action ----
    if args.delete:
        print(f"\nDeleting {len(all_corrupted)} corrupted file(s)...")
        for path, _ in all_corrupted:
            Path(path).unlink()
            print(f"  Deleted: {path}")
        print("Done.")

    elif args.quarantine:
        quarantine_dir = Path("data/corrupted")
        quarantine_dir.mkdir(parents=True, exist_ok=True)
        print(f"\nMoving {len(all_corrupted)} corrupted file(s) to {quarantine_dir}/...")
        for path, _ in all_corrupted:
            src = Path(path)
            # preserve relative structure inside quarantine
            try:
                rel = src.relative_to(data_dir)
            except ValueError:
                rel = src.name
            dst = quarantine_dir / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(src), str(dst))
            print(f"  Moved: {path} -> {dst}")
        print("Done.")

    else:
        print(f"\nThis was a DRY RUN. To remove these files, re-run with --delete or --quarantine.")


if __name__ == "__main__":
    main()
