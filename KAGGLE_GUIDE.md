# MedJEPA — Complete Kaggle Finishing Guide

> **Your situation:** The JupyterLab GPU is gone. The trained checkpoint
> (`best_model.pt`) was gitignored and never pushed. You have all the code
> (with bug fixes) and the old results JSON, but no model weights.
>
> **The plan:** Retrain from scratch on a free Kaggle T4 GPU in **two
> sessions** (~7 hours each). Session 1 does pretraining + saves the
> checkpoint. Session 2 loads it and runs full evaluation with the fixed kNN.
>
> This guide covers everything from "open Kaggle" to "push and submit."

---

## What You Have Right Now

| Thing | Status |
|-------|--------|
| All source code (with fixes) | On your PC, ready to push |
| `results/evaluation_results.json` | Old copy with buggy 5-shot numbers |
| `best_model.pt` checkpoint | **GONE** (was gitignored) |
| Visualization PNGs | Not generated yet |
| `.github/`, `CONTRIBUTING.md`, `examples/`, `pyproject.toml` | On your PC, not yet pushed |

## What We Need To Get

| Thing | How |
|-------|-----|
| Trained checkpoint | Retrain on Kaggle (Session 1) |
| Corrected results JSON | Re-evaluate on Kaggle (Session 2) |
| Visualization PNGs | Generate from notebook (Session 2) |
| Published checkpoint | GitHub Release after download |

---

## Before Kaggle: Push Your Code Fixes

All the bug fixes we made (lazy pydicom import, fixed kNN, pyproject.toml,
minimal demo, etc.) need to be on GitHub so Kaggle can clone them.

Open a terminal on your local machine:

```bash
cd f:\Projects\MedJEPA

# Stage everything
git add .gitignore README.md setup.py pyproject.toml
git add medjepa/data/dicom_utils.py medjepa/data/preprocessing.py
git add medjepa/evaluation/__init__.py medjepa/evaluation/few_shot.py
git add scripts/run_gpu_full.py scripts/download_data.py
git add tests/test_core.py
git add notebooks/05_results_analysis.ipynb
git add .github/ CONTRIBUTING.md examples/

# The partial results file was deleted on purpose
git rm --cached results/evaluation_results_partial.json 2>NUL

# Commit
git commit -m "Fix broken imports, add CI, minimal demo, pyproject.toml, and kNN bugfix"

# Push
git push
```

Verify it worked: go to `https://github.com/prthmmkhija1/MedJEPA` and
check that `examples/minimal_medjepa.py` and `.github/workflows/ci.yml` exist.

---

## Session 1 — Pretrain the Model (~6-8 hours)

This session downloads datasets and trains LeJEPA from scratch.

### 1A. Create a Kaggle Notebook

1. Go to [kaggle.com/code](https://www.kaggle.com/code)
2. Click **"New Notebook"**
3. Right sidebar settings:
   - **Accelerator** → **GPU T4 x2** (or T4 x1)
   - **Internet** → **ON**
   - **Persistence** → **Files only**
4. Rename it to: `MedJEPA-Session1-Pretrain`

### 1B. Paste These Cells

---

**Cell 1 — Clone and install**

> **Why `/kaggle/working/`?** Kaggle has two disk areas:
> - Root filesystem (`/`) — only ~5 GB, fills up fast, causes "no space left on device"
> - `/kaggle/working/` — ~70 GB output disk, this is where you must work
>
> Always clone and install here. The `GIT_TMPDIR` line tells git to write
> temp files to the big disk too (otherwise it uses `/tmp` on the small disk).

```python
# ============================================================
# CELL 1: Get the code from GitHub and install it
# ============================================================
import os, shutil, subprocess

# ── STEP 1: move to / first so we can safely delete /kaggle/working/* ─
# (if current dir is inside /kaggle/working/MedJEPA and we delete it,
#  the shell dies with "getcwd: No such file or directory")
os.chdir('/')

# ── STEP 2: show what is eating disk space ────────────────────────────
print("── Disk before cleanup ──────────────────────")
subprocess.run(['df', '-h', '/kaggle/working'], check=False)
subprocess.run('du -sh /kaggle/working/* 2>/dev/null | sort -rh | head -15',
               shell=True, check=False)

# ── STEP 3: wipe ALL of /kaggle/working/ to start fresh ──────────────
# /kaggle/input/ (read-only datasets you added) is NOT touched.
# Everything inside /kaggle/working/ is rebuilt by the cells below.
print("\n── Wiping /kaggle/working/ ──────────────────")
for name in os.listdir('/kaggle/working'):
    full = f'/kaggle/working/{name}'
    try:
        shutil.rmtree(full) if os.path.isdir(full) else os.remove(full)
        print(f"  removed {full}")
    except Exception as e:
        print(f"  skip    {full}: {e}")

print("\n── Disk after cleanup ───────────────────────")
subprocess.run(['df', '-h', '/kaggle/working'], check=False)

# ── STEP 4: clone into /kaggle/working/ (plenty of room now) ─────────
os.makedirs('/kaggle/working/tmp', exist_ok=True)
os.environ['GIT_TMPDIR'] = '/kaggle/working/tmp'   # git temp files → here

!git clone --depth 1 https://github.com/prthmmkhija1/MedJEPA.git /kaggle/working/MedJEPA
%cd /kaggle/working/MedJEPA
!pip install -e ".[medical]" -q --cache-dir /kaggle/working/pip_cache
!pip install gdown -q --cache-dir /kaggle/working/pip_cache
print("Setup done!")
```

---

**Cell 2 — Download datasets**

```python
# ============================================================
# CELL 2: Download datasets using Kaggle API
# ============================================================
# ChestXray14 is ~42 GB — we skip it to save disk space.
# HAM10000 + APTOS + PCam + BraTS (~15 GB total) is enough
# for strong pretraining results.
#
# Takes ~20-30 min.

!python scripts/download_data.py  # skips ChestXray14 by default

# Quick check: what downloaded?
import os
data_dir = 'data/raw'
if os.path.exists(data_dir):
    print("\nDownloaded datasets:")
    for item in sorted(os.listdir(data_dir)):
        path = os.path.join(data_dir, item)
        if os.path.isdir(path):
            n_files = sum(len(f) for _, _, f in os.walk(path))
            print(f"  {item:30s}  ({n_files:,} files)")
```

> **Disk space warning:** Kaggle gives ~70 GB. If ChestXray14 fails, don't
> worry — the model will train on the other datasets. HAM10000 + APTOS +
> PCam + BraTS is enough for good results.

---

**Cell 3 — Train the model (THE BIG STEP)**

```python
# ============================================================
# CELL 3: Full training — Phase 1 (LeJEPA) + Phase 2 (V-JEPA)
# ============================================================
# This trains the model from scratch. Settings are tuned for T4 GPU:
#   - batch_size 64 (T4 has 16GB VRAM, can't do 256 like A100)
#   - gradient_accumulation_steps 4 (effective batch = 64 * 4 = 256)
#   - epochs 50 (half of full 100, but enough for strong features)
#   - V-JEPA included (50 epochs on 3D volumes)
#
# The script automatically:
#   - Combines all available 2D datasets for pretraining
#   - Extracts 2D slices from 3D volumes
#   - Saves checkpoints every 5 epochs
#   - Saves the best model to checkpoints/best_model.pt
#
# TAKES: ~5-7 hours on T4. Watch the loss — it should decrease.

!python scripts/run_gpu_full.py \
    --epochs 50 \
    --batch_size 64 \
    --gradient_accumulation_steps 4 \
    --num_workers 2 \
    --lr 0.0003 \
    --checkpoint_dir /kaggle/working/checkpoints \
    --results_dir /kaggle/working/results \
    --imagenet_backbone resnet50 \
    --ft_epochs 5 \
    --cache_dataset
```

> **What if it crashes?** The script saves checkpoints every 5 epochs.
> If you restart, skip to Session 2's Cell 2 and point `--checkpoint`
> at the latest saved checkpoint file.

---

**Cell 4 — Save checkpoint for download**

```python
# ============================================================
# CELL 4: Copy checkpoint to Kaggle output directory
# ============================================================
# Kaggle's /kaggle/working/ is the output directory. Files here
# persist after the session ends and can be downloaded.

import shutil, os, glob

# Find the best checkpoint
ckpt_candidates = [
    '/kaggle/working/checkpoints/best_model.pt',
    'checkpoints/best_model.pt',
]
# Also grab any checkpoint_epoch_*.pt as backup
ckpt_candidates += sorted(glob.glob('/kaggle/working/checkpoints/checkpoint_epoch_*.pt'))
ckpt_candidates += sorted(glob.glob('checkpoints/checkpoint_epoch_*.pt'))

found_ckpt = None
for path in ckpt_candidates:
    if os.path.exists(path):
        found_ckpt = path
        break

if found_ckpt:
    out_path = '/kaggle/working/best_model.pt'
    shutil.copy(found_ckpt, out_path)
    size_mb = os.path.getsize(out_path) / 1e6
    print(f"Checkpoint saved: {out_path} ({size_mb:.1f} MB)")
    print("Download this from the Output tab!")
else:
    print("ERROR: No checkpoint found. Training may not have completed.")
    print("Check /kaggle/working/checkpoints/ manually:")
    for root, dirs, files in os.walk('/kaggle/working/checkpoints'):
        for f in files:
            print(f"  {os.path.join(root, f)}")

# Also save any partial results
if os.path.exists('/kaggle/working/results/evaluation_results.json'):
    size = os.path.getsize('/kaggle/working/results/evaluation_results.json')
    print(f"\nResults also saved: evaluation_results.json ({size:,} bytes)")
```

### 1C. After Session 1 Ends

1. Click the **"Output"** tab on the right sidebar
2. Download **`best_model.pt`** (this is your trained model, ~300-400 MB)
3. Also download `evaluation_results.json` if it exists (Session 1 may
   run some evaluations before timing out)

### 1D. Re-upload checkpoint as a Kaggle Dataset

We need the checkpoint accessible for Session 2:

1. Go to [kaggle.com/datasets/new](https://www.kaggle.com/datasets/new)
2. Click **"New Dataset"**
3. Name: `medjepa-checkpoint`
4. Drag and drop the `best_model.pt` you just downloaded
5. Visibility: **Private**
6. Click **"Create"** and wait for upload

---

## Session 2 — Evaluate + Visualize (~3-4 hours)

This session loads your trained model and runs all evaluations with the
fixed kNN code.

### 2A. Create a new Kaggle Notebook

1. **"New Notebook"** on Kaggle
2. Settings: **GPU T4**, **Internet ON**
3. **"Add Input"** → search for your `medjepa-checkpoint` dataset → Add it
4. Rename to: `MedJEPA-Session2-Evaluate`

### 2B. Paste These Cells

---

**Cell 1 — Setup (same as Session 1)**

```python
# ============================================================
# CELL 1: Clone and install (same as Session 1)
# ============================================================
import os, shutil, subprocess

os.chdir('/')   # step away from /kaggle/working/* before wiping it

print("── Disk before cleanup ──────────────────────")
subprocess.run(['df', '-h', '/kaggle/working'], check=False)
subprocess.run('du -sh /kaggle/working/* 2>/dev/null | sort -rh | head -15',
               shell=True, check=False)

print("\n── Wiping /kaggle/working/ ──────────────────")
for name in os.listdir('/kaggle/working'):
    full = f'/kaggle/working/{name}'
    try:
        shutil.rmtree(full) if os.path.isdir(full) else os.remove(full)
        print(f"  removed {full}")
    except Exception as e:
        print(f"  skip    {full}: {e}")

print("\n── Disk after cleanup ───────────────────────")
subprocess.run(['df', '-h', '/kaggle/working'], check=False)

os.makedirs('/kaggle/working/tmp', exist_ok=True)
os.environ['GIT_TMPDIR'] = '/kaggle/working/tmp'

!git clone --depth 1 https://github.com/prthmmkhija1/MedJEPA.git /kaggle/working/MedJEPA
%cd /kaggle/working/MedJEPA
!pip install -e ".[medical]" -q --cache-dir /kaggle/working/pip_cache
!pip install gdown -q --cache-dir /kaggle/working/pip_cache
print("Setup done!")
```

---

**Cell 2 — Copy checkpoint from your uploaded dataset**

```python
# ============================================================
# CELL 2: Load the checkpoint you trained in Session 1
# ============================================================
import os, shutil, glob

os.makedirs('checkpoints', exist_ok=True)

# Search common Kaggle input paths
search_paths = glob.glob('/kaggle/input/*/best_model.pt') + \
               glob.glob('/kaggle/input/*/*/best_model.pt') + \
               glob.glob('/kaggle/input/*/checkpoints/best_model.pt')

# Also check for any .pt files
if not search_paths:
    search_paths = glob.glob('/kaggle/input/**/*.pt', recursive=True)

if search_paths:
    src = search_paths[0]
    shutil.copy(src, 'checkpoints/best_model.pt')
    size_mb = os.path.getsize('checkpoints/best_model.pt') / 1e6
    print(f"Checkpoint loaded: {size_mb:.1f} MB from {src}")
else:
    print("ERROR: No checkpoint found in /kaggle/input/")
    print("\nHere's what IS in /kaggle/input/:")
    for root, dirs, files in os.walk('/kaggle/input/'):
        for f in files:
            print(f"  {os.path.join(root, f)}")
    print("\nMake sure you added your medjepa-checkpoint dataset as Input!")
```

---

**Cell 3 — Download datasets (same as Session 1)**

```python
# ============================================================
# CELL 3: Download datasets (same as Session 1)
# ============================================================
!python scripts/download_data.py

import os
data_dir = 'data/raw'
if os.path.exists(data_dir):
    print("\nAvailable datasets:")
    for item in sorted(os.listdir(data_dir)):
        path = os.path.join(data_dir, item)
        if os.path.isdir(path):
            n_files = sum(len(f) for _, _, f in os.walk(path))
            print(f"  {item:30s}  ({n_files:,} files)")
```

---

**Cell 4 — Run full evaluation with fixed kNN (THE KEY STEP)**

```python
# ============================================================
# CELL 4: Run ALL evaluation — with the corrected kNN code
# ============================================================
# This skips pretraining (already done) and runs:
#   - Linear probing (freeze encoder, train a linear head)
#   - Few-shot kNN (NOW FIXED: L2-normalized, cosine similarity)
#   - N-shot (5/10/20 examples per class)
#   - Fine-tuning (unfreeze encoder, train end-to-end)
#   - ImageNet baseline comparison
#   - Segmentation (BraTS Dice score)
#   - Cross-institutional domain analysis
#
# TAKES: ~2-3 hours on T4

!python scripts/run_gpu_full.py \
    --skip_pretrain \
    --skip_vjepa \
    --checkpoint checkpoints/best_model.pt \
    --batch_size 64 \
    --num_workers 2 \
    --results_dir results \
    --checkpoint_dir checkpoints \
    --imagenet_backbone resnet50 \
    --ft_epochs 5
```

> If you get "Out of Memory", add `--skip_imagenet --skip_finetune`
> to skip the memory-heavy evaluations. Linear probe + few-shot are
> the most important ones for the GSoC submission.

---

**Cell 5 — Check the corrected results**

```python
# ============================================================
# CELL 5: Print results — verify 5-shot numbers are fixed
# ============================================================
import json

with open('results/evaluation_results.json') as f:
    results = json.load(f)

print("=" * 70)
print("RESULTS SUMMARY (with corrected kNN)")
print("=" * 70)

# Classification
print("\n--- Linear Probe ---")
for name in ['ham10000', 'aptos2019', 'pcam', 'chestxray14', 'brats']:
    res = results.get(name, {})
    lp = res.get('linear_probing', {})
    if 'accuracy' in lp:
        print(f"  {name:20s}  Acc: {lp['accuracy']:.1%}   AUC: {lp.get('auc', 0):.3f}")

# THE CRITICAL CHECK: 5-shot numbers
print("\n--- N-Shot (should be MUCH higher than before) ---")
for name in ['ham10000', 'aptos2019', 'pcam', 'brats']:
    res = results.get(name, {})
    ns = res.get('n_shot', {})
    if ns:
        parts = []
        for shot in ['5-shot', '10-shot', '20-shot']:
            acc = ns.get(shot, {}).get('accuracy')
            if acc is not None:
                parts.append(f"{shot}={acc:.1%}")
        if parts:
            print(f"  {name:20s}  {',  '.join(parts)}")

# Fine-tune vs ImageNet
print("\n--- Fine-Tune vs ImageNet ---")
for name in ['ham10000', 'aptos2019', 'pcam']:
    res = results.get(name, {})
    ft = res.get('fine_tuning', {}).get('accuracy')
    inet = res.get('imagenet_baseline', {}).get('accuracy')
    if ft and inet:
        winner = "MEDJEPA WINS" if ft > inet else "ImageNet leads"
        print(f"  {name:20s}  FT={ft:.1%}  ImageNet={inet:.1%}  [{winner}]")

# Segmentation
print("\n--- Segmentation ---")
for name, res in results.items():
    if 'mean_dice' in res:
        fg = res.get('per_class_dice', {}).get('1', 0)
        if isinstance(fg, (int, float)) and fg > 0.01:
            print(f"  {name:40s}  Dice={res['mean_dice']:.3f}  FG={fg:.3f}")

print("\n" + "=" * 70)
print("SAVE THESE NUMBERS — you'll update the README with them.")
print("=" * 70)
```

---

**Cell 6 — Run the visualization notebook**

```python
# ============================================================
# CELL 6: Execute notebook 05 to generate all plots
# ============================================================
!pip install nbconvert ipykernel -q
!python -m ipykernel install --user --name python3 2>/dev/null

!jupyter nbconvert --execute --to notebook --inplace \
    --ExecutePreprocessor.timeout=900 \
    --ExecutePreprocessor.kernel_name=python3 \
    notebooks/05_results_analysis.ipynb

# Check what plots were created
import os
print("\nGenerated files:")
for f in sorted(os.listdir('results')):
    size = os.path.getsize(f'results/{f}')
    icon = {'png': 'PLOT', 'json': 'DATA', 'csv': 'TABLE'}.get(f.split('.')[-1], 'FILE')
    print(f"  [{icon:5s}]  {f:50s}  {size/1024:.1f} KB")
```

---

**Cell 7 — Package everything for download**

```python
# ============================================================
# CELL 7: Zip everything for download
# ============================================================
import shutil, os

OUT = '/kaggle/working'

# Results JSON
shutil.copy('results/evaluation_results.json', f'{OUT}/evaluation_results.json')

# All plots and tables
for f in os.listdir('results'):
    if f.endswith(('.png', '.csv')):
        shutil.copy(f'results/{f}', f'{OUT}/{f}')

# Executed notebook (with visible plot outputs)
shutil.copy(
    'notebooks/05_results_analysis.ipynb',
    f'{OUT}/05_results_analysis.ipynb'
)

# Checkpoint for publishing
if os.path.exists('checkpoints/best_model.pt'):
    shutil.copy('checkpoints/best_model.pt', f'{OUT}/best_model.pt')

# Create one zip with everything
shutil.make_archive(f'{OUT}/medjepa_final', 'zip', OUT)

print("=" * 50)
print("ALL DONE!")
print("=" * 50)
print("\nGo to the 'Output' tab and download:")
print("  medjepa_final.zip  (everything in one file)")
print("\nOr download individual files.")
```

---

## After Kaggle — Final Steps on Your PC

### Step 1: Extract the downloaded zip

Extract `medjepa_final.zip` somewhere (e.g. `C:\Users\prath\Downloads\medjepa_final\`).

### Step 2: Copy files into your project

```bash
cd f:\Projects\MedJEPA

# Overwrite the old (buggy) results with the corrected ones
copy "C:\Users\prath\Downloads\medjepa_final\evaluation_results.json" results\evaluation_results.json

# Copy all plot images into results folder
copy "C:\Users\prath\Downloads\medjepa_final\*.png" results\

# Copy the CSV results table
copy "C:\Users\prath\Downloads\medjepa_final\full_results_table.csv" results\

# Copy the executed notebook (now shows plots on GitHub)
copy "C:\Users\prath\Downloads\medjepa_final\05_results_analysis.ipynb" notebooks\05_results_analysis.ipynb

# Keep best_model.pt somewhere safe for the GitHub Release
copy "C:\Users\prath\Downloads\medjepa_final\best_model.pt" C:\Users\prath\Desktop\best_model.pt
```

### Step 3: Update the README few-shot table

Open `results/evaluation_results.json` and look at the `n_shot` sections.
Then edit `README.md` — find the few-shot table (around line 350) and replace
the old numbers:

```markdown
#### Few-Shot Learning (kNN)

| Dataset  | 5-shot | 10-shot | 20-shot | 1% data | 10% data | 100% data |
| -------- | :----: | :-----: | :-----: | :-----: | :------: | :-------: |
| HAM10000 | XX.X%  |  XX.X%  |  XX.X%  |  XX.X%  |  XX.X%   |   XX.X%   |
| ...
```

Replace all `XX.X%` with the real numbers from Cell 5's output.

Also update the "Honest gaps" bullet in the Discussion section. Change this:

> Few-shot n-shot results (5-shot) are weak: HAM10000 5-shot accuracy (8.9%)
> is below random...

To something like:

> Few-shot performance scales with examples: HAM10000 5-shot reaches XX.X%
> (up from the initial 8.9% after fixing the kNN distance metric to use
> L2-normalized cosine similarity).

### Step 4: Add the checkpoint download link to README

Add this right before the "Quick Start" section in `README.md`:

```markdown
## Pre-trained Checkpoint

Download the pre-trained LeJEPA model:
[**best_model.pt** — GitHub Release v0.1.0](https://github.com/prthmmkhija1/MedJEPA/releases/tag/v0.1.0)

\```python
import torch
from medjepa.models.lejepa import LeJEPA

model = LeJEPA(image_size=224, patch_size=16, embed_dim=768,
               encoder_depth=12, predictor_depth=6)
ckpt = torch.load('best_model.pt', map_location='cpu', weights_only=False)
model.load_state_dict(ckpt['model_state_dict'])
model.eval()
\```
```

(Remove the backslashes before the triple backticks — they're just escaping here.)

### Step 5: Commit and push

```bash
cd f:\Projects\MedJEPA

git add results/evaluation_results.json
git add results/*.png results/*.csv
git add notebooks/05_results_analysis.ipynb
git add README.md

git commit -m "Add corrected evaluation results and visualization plots"
git push
```

### Step 6: Publish checkpoint as a GitHub Release

1. Go to `https://github.com/prthmmkhija1/MedJEPA/releases/new`
2. Fill in:
   - **Tag:** `v0.1.0`
   - **Release title:** `MedJEPA v0.1.0 — Pre-trained Checkpoint`
   - **Description:**
     ```
     Pre-trained MedJEPA (LeJEPA) checkpoint.

     Architecture: ViT-B/12 (768-dim, 12 layers, 6-layer predictor)
     Training: 50 epochs on HAM10000 + APTOS + PCam + BraTS
     Loss: SIGReg (prediction + variance + sketched covariance)

     Usage:
       ckpt = torch.load('best_model.pt', map_location='cpu', weights_only=False)
       model.load_state_dict(ckpt['model_state_dict'])
     ```
3. Click **"Attach binaries"** → upload `best_model.pt` from your Desktop
4. Click **"Publish release"**

### Step 7: Final commit with checkpoint link

```bash
cd f:\Projects\MedJEPA
git add README.md
git commit -m "Add pre-trained checkpoint download link"
git push
```

---

## Troubleshooting

### "No space left on device" / "getcwd: No such file or directory"

Two things happen together:
1. `/kaggle/working/` is 100% full from a previous run's datasets/cache
2. If you delete the folder you're currently inside, git/shell dies with
   "getcwd: No such file or directory"

Fix — paste this into a fresh cell and run it **before** Cell 1:

```python
import os, shutil, subprocess

os.chdir('/')   # <-- critical: move out of /kaggle/working/ first

# Show what's eating the disk
subprocess.run(['df', '-h', '/kaggle/working'], check=False)
subprocess.run('du -sh /kaggle/working/* 2>/dev/null | sort -rh', shell=True, check=False)

# Wipe everything in /kaggle/working/ (safe — Cell 2 re-downloads data)
for name in os.listdir('/kaggle/working'):
    full = f'/kaggle/working/{name}'
    shutil.rmtree(full) if os.path.isdir(full) else os.remove(full)
    print(f"removed {full}")

subprocess.run(['df', '-h', '/kaggle/working'], check=False)
print("Disk cleared — now run Cell 1 again")
```

---

### "CUDA out of memory" during training (Session 1)

Reduce batch size and use gradient accumulation:

```python
!python scripts/run_gpu_full.py \
    --epochs 50 \
    --batch_size 32 \
    --gradient_accumulation_steps 8 \
    --num_workers 2 \
    --checkpoint_dir /kaggle/working/checkpoints \
    --results_dir /kaggle/working/results \
    --gradient_checkpointing
```

### "CUDA out of memory" during evaluation (Session 2)

Skip the heaviest evaluations:

```python
!python scripts/run_gpu_full.py \
    --skip_pretrain --skip_vjepa \
    --checkpoint checkpoints/best_model.pt \
    --batch_size 32 \
    --num_workers 2 \
    --skip_imagenet \
    --skip_finetune
```

You'll still get linear probe + few-shot + segmentation, which are the most
important results for the proposal.

### Session times out mid-training

The script saves a checkpoint every 5 epochs. After restart:

1. Find the latest checkpoint in the Output tab (e.g. `checkpoint_epoch_45.pt`)
2. Upload it as a Kaggle Dataset (like Step 1D)
3. Start Session 2 but use that checkpoint:

```python
!python scripts/run_gpu_full.py \
    --skip_pretrain --skip_vjepa \
    --checkpoint checkpoints/checkpoint_epoch_45.pt \
    --batch_size 64 --num_workers 2
```

### "Dataset not found" errors

Perfectly fine. The script skips missing datasets and continues with the rest.
Getting results on 3-4 datasets is enough for the GSoC submission.

### Notebook Cell 6 fails

The visualization notebook needs the results JSON to exist. If Cell 4
completed, this should work. If it errors:

```python
# Run just the plotting parts manually
import json, os
os.makedirs('results', exist_ok=True)

with open('results/evaluation_results.json') as f:
    results = json.load(f)

# At minimum, generate the key bar chart
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

names, accs, baselines = [], [], []
for name, res in results.items():
    lp = res.get('linear_probing', {})
    bl = res.get('supervised_baseline', {})
    if 'accuracy' in lp:
        names.append(name)
        accs.append(lp['accuracy'])
        baselines.append(bl.get('accuracy', 0))

fig, ax = plt.subplots(figsize=(12, 6))
x = np.arange(len(names))
ax.bar(x - 0.2, accs, 0.4, label='MedJEPA', color='steelblue')
ax.bar(x + 0.2, baselines, 0.4, label='Random Init', color='lightcoral')
ax.set_xticks(x)
ax.set_xticklabels(names, rotation=30, ha='right')
ax.set_ylabel('Accuracy')
ax.set_title('MedJEPA vs Random Init Baseline')
ax.legend()
ax.set_ylim(0, 1)
plt.tight_layout()
plt.savefig('results/multi_dataset_linear_probe.png', dpi=150)
print("Saved: results/multi_dataset_linear_probe.png")
```

### "Can I do both sessions in one go?"

Yes, if your Session 1 finishes training in under ~7 hours. The
`run_gpu_full.py` script already runs Phase 1 → Phase 2 → Phase 3 in
sequence. So if you use the Cell 3 command from Session 1, it will train
AND evaluate in one shot. The only risk is the 12-hour Kaggle time limit.

If you want to try it all in one session, use:

```python
# ONE SESSION: train + evaluate (risky — may hit 12h limit)
!python scripts/run_gpu_full.py \
    --epochs 50 \
    --batch_size 64 \
    --gradient_accumulation_steps 4 \
    --num_workers 2 \
    --lr 0.0003 \
    --checkpoint_dir /kaggle/working/checkpoints \
    --results_dir /kaggle/working/results \
    --imagenet_backbone resnet50 \
    --ft_epochs 5 \
    --cache_dataset
```

---

## Final Checklist — Is My Submission Ready?

After completing all steps, verify:

- [ ] `results/evaluation_results.json` has corrected 5-shot numbers (all > 15%)
- [ ] `results/` folder has PNG plot files (at least 3-4 charts)
- [ ] `notebooks/05_results_analysis.ipynb` shows plots when viewed on GitHub
- [ ] GitHub Release `v0.1.0` has `best_model.pt` attached
- [ ] README has checkpoint download link
- [ ] README few-shot table has the corrected numbers
- [ ] All 24 tests pass (`python -m pytest tests/ -v`)
- [ ] `examples/minimal_medjepa.py` exists and runs
- [ ] `.github/workflows/ci.yml` is committed
- [ ] `CONTRIBUTING.md` is committed
- [ ] `pyproject.toml` is committed

---

## Time Estimate

| Step | Where | Time |
|------|-------|:----:|
| Push code fixes to GitHub | Local | 5 min |
| **Session 1:** Clone + download data | Kaggle | 30-45 min |
| **Session 1:** Train model (50 epochs) | Kaggle | **5-7 hours** |
| Download checkpoint, re-upload as dataset | Browser | 10 min |
| **Session 2:** Clone + download data | Kaggle | 30-45 min |
| **Session 2:** Full evaluation (fixed kNN) | Kaggle | **2-3 hours** |
| **Session 2:** Run notebook + package | Kaggle | 15 min |
| Copy files back, update README | Local | 15 min |
| Publish GitHub Release | Browser | 5 min |
| Final commit + push | Local | 5 min |
| **Total** | | **~10-12 hours** |

This fits across two Kaggle sessions (12 hours each).
If you try everything in one session with `--epochs 50`,
it might finish in ~8-10 hours total.
