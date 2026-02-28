"""
Pre-extract 2D slices from NIfTI volumes and save as .npy files.

This is a ONE-TIME step that speeds up training by 10-50x because:
  - No more gzip decompression during training
  - No more loading full 3D volumes for a single 2D slice
  - .npy files load in microseconds vs seconds for .nii.gz

Usage (run once before training):
    python scripts/preextract_slices.py

Output structure:
    data/processed/nifti_slices/
        brats/
            BraTS2021_00000_flair_s05.npy
            BraTS2021_00000_flair_s05_label.npy   (if labels exist)
            ...
        decathlon_Task02_Heart/
            ...
        manifest.json   ← index file used by the fast dataset class
"""

import argparse
import json
import sys
import time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np


# ---------- helpers (run in worker processes) ----------

def _extract_one_volume(args_tuple):
    """
    Extract slices from a single NIfTI volume.
    Returns list of dicts with slice metadata, or (path, error_str) on failure.
    """
    nifti_path, out_dir, prefix, slices_per_volume, save_labels, label_path = args_tuple
    import nibabel as nib
    from PIL import Image

    TARGET_SIZE = (224, 224)
    results = []

    try:
        vol = nib.load(str(nifti_path)).get_fdata().astype(np.float32)
    except Exception as e:
        return (str(nifti_path), str(e))

    # Handle 4D volumes (take first channel)
    if vol.ndim == 4:
        vol = vol[..., 0]

    n_slices = vol.shape[2] if vol.ndim >= 3 else vol.shape[0]

    # Pick slices from middle 60%
    start = int(n_slices * 0.2)
    end = int(n_slices * 0.8)
    if end - start < slices_per_volume:
        indices = list(range(start, max(end, start + 1)))
    else:
        indices = np.linspace(start, end - 1, slices_per_volume, dtype=int).tolist()

    # Load label volume if needed
    label_vol = None
    if save_labels and label_path and Path(label_path).exists():
        try:
            label_vol = nib.load(str(label_path)).get_fdata()
        except Exception:
            label_vol = None

    for si in indices:
        # Extract slice
        if vol.ndim >= 3:
            slc = vol[:, :, si]
        else:
            slc = vol

        # Normalize to [0, 1]
        smin, smax = slc.min(), slc.max()
        if smax - smin > 0:
            slc = (slc - smin) / (smax - smin)
        else:
            slc = np.zeros_like(slc)

        # Resize
        pil_img = Image.fromarray((slc * 255).astype(np.uint8))
        pil_img = pil_img.resize(TARGET_SIZE, Image.LANCZOS)
        slc_resized = np.array(pil_img, dtype=np.float32) / 255.0

        # Stack to 3 channels (HWC)
        img_3ch = np.stack([slc_resized] * 3, axis=-1)

        # Save
        fname = f"{prefix}_s{si:03d}.npy"
        out_path = Path(out_dir) / fname
        np.save(str(out_path), img_3ch)

        entry = {"file": fname, "source_volume": str(nifti_path)}

        # Save label slice if available
        if label_vol is not None:
            if label_vol.ndim >= 3:
                lbl_slc = label_vol[:, :, si]
            else:
                lbl_slc = label_vol
            # Resize label with nearest-neighbor (preserve integer classes)
            lbl_pil = Image.fromarray(lbl_slc.astype(np.int16))
            lbl_pil = lbl_pil.resize(TARGET_SIZE, Image.NEAREST)
            lbl_resized = np.array(lbl_pil, dtype=np.int16)
            lbl_fname = f"{prefix}_s{si:03d}_label.npy"
            np.save(str(Path(out_dir) / lbl_fname), lbl_resized)
            entry["label_file"] = lbl_fname
            # Binary label: >1% foreground
            fg_frac = (lbl_resized > 0).sum() / max(lbl_resized.size, 1)
            entry["label"] = 1 if fg_frac > 0.01 else 0

        results.append(entry)

    return results


# ---------- dataset-specific collectors ----------

def collect_brats(data_dir, modality="flair"):
    """Collect BraTS volumes + labels."""
    brats_dir = Path(data_dir) / "brats"
    if not brats_dir.exists():
        return [], "brats"
    volumes = []
    for subj_dir in sorted(brats_dir.iterdir()):
        if not subj_dir.is_dir() or not subj_dir.name.startswith("BraTS"):
            continue
        flair = subj_dir / f"{subj_dir.name}_{modality}.nii.gz"
        seg = subj_dir / f"{subj_dir.name}_seg.nii.gz"
        if flair.exists():
            prefix = f"{subj_dir.name}_{modality}"
            label_path = str(seg) if seg.exists() else None
            volumes.append((str(flair), prefix, label_path))
    return volumes, "brats"


def collect_decathlon(data_dir):
    """Collect all Decathlon task volumes + labels."""
    dec_dir = Path(data_dir) / "decathlon"
    if not dec_dir.exists():
        return {}, "decathlon"

    tasks = {}
    for task_dir in sorted(dec_dir.glob("Task*")):
        meta_path = task_dir / "dataset.json"
        if not meta_path.exists():
            continue

        with open(meta_path) as f:
            meta = json.load(f)

        volumes = []
        for entry in meta.get("training", []):
            img_rel = entry["image"]
            img_path = task_dir / img_rel.lstrip("./")
            lbl_rel = entry.get("label", "")
            lbl_path = str(task_dir / lbl_rel.lstrip("./")) if lbl_rel else None

            if img_path.exists():
                stem = img_path.stem.replace(".nii", "")
                prefix = f"{task_dir.name}_{stem}"
                volumes.append((str(img_path), prefix, lbl_path))

        # Fallback: glob imagesTr if dataset.json paths don't match
        if not volumes:
            images_tr = task_dir / "imagesTr"
            if images_tr.exists():
                for nii in sorted(images_tr.glob("*.nii.gz")):
                    stem = nii.stem.replace(".nii", "")
                    prefix = f"{task_dir.name}_{stem}"
                    volumes.append((str(nii), prefix, None))

        if volumes:
            tasks[task_dir.name] = volumes

    return tasks, "decathlon"


def main():
    parser = argparse.ArgumentParser(description="Pre-extract NIfTI slices to .npy")
    parser.add_argument("--data-dir", type=str, default="data/raw",
                        help="Root raw data directory")
    parser.add_argument("--output-dir", type=str, default="data/processed/nifti_slices",
                        help="Where to save extracted slices")
    parser.add_argument("--slices-per-volume", type=int, default=10)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--with-labels", action="store_true", default=True)
    args = parser.parse_args()

    out_base = Path(args.output_dir)
    out_base.mkdir(parents=True, exist_ok=True)

    manifest = {}  # dataset_name -> [slice entries]

    # ── BraTS ──
    brats_vols, _ = collect_brats(args.data_dir)
    if brats_vols:
        brats_out = out_base / "brats"
        brats_out.mkdir(exist_ok=True)
        print(f"\n[BraTS] {len(brats_vols)} volumes, extracting {args.slices_per_volume} slices each...")

        tasks = [
            (path, str(brats_out), prefix, args.slices_per_volume, args.with_labels, lbl)
            for path, prefix, lbl in brats_vols
        ]
        manifest["brats"] = _run_extraction(tasks, args.workers, "BraTS")

    # ── Decathlon (per task) ──
    dec_tasks, _ = collect_decathlon(args.data_dir)
    for task_name, vols in dec_tasks.items():
        task_out = out_base / f"decathlon_{task_name}"
        task_out.mkdir(exist_ok=True)
        print(f"\n[Decathlon/{task_name}] {len(vols)} volumes, extracting {args.slices_per_volume} slices each...")

        tasks = [
            (path, str(task_out), prefix, args.slices_per_volume, args.with_labels, lbl)
            for path, prefix, lbl in vols
        ]
        manifest[f"decathlon_{task_name}"] = _run_extraction(tasks, args.workers, task_name)

    # ── Save manifest ──
    manifest_path = out_base / "manifest.json"
    # Convert to serializable format
    serializable = {}
    for key, entries in manifest.items():
        serializable[key] = entries

    with open(manifest_path, "w") as f:
        json.dump(serializable, f, indent=2)

    total_slices = sum(len(v) for v in manifest.values())
    print(f"\n{'='*60}")
    print(f"Done! {total_slices} slices extracted to {out_base}")
    print(f"Manifest: {manifest_path}")
    print(f"{'='*60}")


def _run_extraction(task_args, workers, desc):
    """Run parallel extraction and return list of slice entries."""
    all_entries = []
    errors = []
    done = 0
    total = len(task_args)
    t0 = time.time()

    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_extract_one_volume, t): t[0] for t in task_args}
        for future in as_completed(futures):
            done += 1
            result = future.result()
            if isinstance(result, tuple):
                # Error
                path, err = result
                errors.append(path)
                print(f"  SKIP (corrupted): {Path(path).name} — {err}")
            else:
                all_entries.extend(result)
            if done % 10 == 0 or done == total:
                elapsed = time.time() - t0
                print(f"  [{done}/{total}] {len(all_entries)} slices, {elapsed:.1f}s")

    if errors:
        print(f"  ⚠ {len(errors)} corrupted volumes skipped for {desc}")

    return all_entries


if __name__ == "__main__":
    main()
