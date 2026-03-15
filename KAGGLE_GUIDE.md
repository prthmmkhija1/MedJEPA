# MedJEPA — Kaggle Finishing Guide

> **New approach:** Add datasets directly as Kaggle "Inputs" — they mount at
> `/kaggle/input/` and do **not** count against your 20 GB working disk.
> No downloading, no disk-space errors.

---

## How This Works

| Location | Size limit | What goes here |
|---|---|---|
| `/kaggle/working/` | 20 GB | Code clone, pip packages, checkpoints, results |
| `/kaggle/input/` | Unlimited (read-only) | Datasets you add as Inputs |

We add HAM10000, APTOS, and BraTS as Inputs → they appear under `/kaggle/input/`
→ Cell 2 creates symlinks so the training script finds them at `data/raw/`.

---

## Step 0 — Add Datasets to Your Kaggle Account (one-time)

You only need to do this once. Go to each link and click **"+ Add"** / **"Add to notebook"**:

| Dataset | Kaggle slug | Size |
|---|---|---|
| HAM10000 skin lesions | `kmader/skin-cancer-mnist-ham10000` | ~2 GB |
| APTOS 2019 retinal | `benjaminwarner/aptos2019-blindness-detection` | ~3 GB |
| BraTS 2021 brain MRI | `dschettler8845/brats-2021-task1` | ~2 GB |

To search for them: go to [kaggle.com/datasets](https://www.kaggle.com/datasets) and
search the slug (the part after the `/`).

---

## Session 1 — Pretrain (~6-8 hours)

### 1A. Create the Notebook

1. Go to [kaggle.com/code](https://www.kaggle.com/code) → **"New Notebook"**
2. Right sidebar:
   - **Accelerator** → **GPU T4 x1**
   - **Internet** → **ON**
   - **Persistence** → **Files only**
3. Rename to: `MedJEPA-Session1-Pretrain`

### 1B. Add Datasets as Inputs

In the right sidebar click **"Add Input"**, then search and add:

1. `skin-cancer-mnist-ham10000` (by kmader)
2. `aptos2019-blindness-detection` (by benjaminwarner)
3. `brats-2021-task1` (by dschettler8845) — optional, for 3D training

Each will appear as a card under "Inputs". They mount automatically under `/kaggle/input/`.

### 1C. Paste These Cells

---

**Cell 1 — Clone and install**

```python
# ============================================================
# CELL 1: Get the code and install it
# ============================================================
import os, shutil, subprocess

# Move to / before any cleanup to avoid "getcwd" errors
os.chdir('/')

# Clean up only the old code clone (NOT /kaggle/input/ — that's your data)
for stale in ['/kaggle/working/MedJEPA',
              '/kaggle/working/pip_cache',
              '/kaggle/working/tmp']:
    if os.path.exists(stale):
        shutil.rmtree(stale, ignore_errors=True)
        print(f"cleaned {stale}")

# Show free disk (should be ~16-18 GB free now)
subprocess.run(['df', '-h', '/kaggle/working'], check=False)

# Clone into working disk, shallow (saves ~200 MB vs full history)
os.makedirs('/kaggle/working/tmp', exist_ok=True)
os.environ['GIT_TMPDIR'] = '/kaggle/working/tmp'

!git clone --depth 1 https://github.com/prthmmkhija1/MedJEPA.git /kaggle/working/MedJEPA
%cd /kaggle/working/MedJEPA
!pip install -e ".[medical]" -q --cache-dir /kaggle/working/pip_cache
!pip install gdown -q --cache-dir /kaggle/working/pip_cache
print("Done!")
```

---

**Cell 2 — Link datasets from /kaggle/input/**

```python
# ============================================================
# CELL 2: Point data/raw/ at the datasets you added as Inputs
# ============================================================
# /kaggle/input/<name>/ is read-only but we just need to READ it,
# so symlinks work perfectly.

import os

RAW = '/kaggle/working/MedJEPA/data/raw'
os.makedirs(RAW, exist_ok=True)

# Map: Kaggle input folder name → folder name our scripts expect
DATASETS = {
    'skin-cancer-mnist-ham10000':              'ham10000',
    'aptos2019-blindness-detection':           'aptos2019',
    'brats-2021-task1':                        'brats',
    # Uncomment if you also added these:
    # 'metastatic-tissue-classification-patchcamelyon': 'pcam',
}

for kaggle_name, our_name in DATASETS.items():
    src = f'/kaggle/input/{kaggle_name}'
    dst = f'{RAW}/{our_name}'
    if os.path.exists(src):
        if os.path.exists(dst) or os.path.islink(dst):
            os.remove(dst) if os.path.islink(dst) else None
        os.symlink(src, dst)
        n = sum(len(f) for _, _, f in os.walk(src))
        print(f"  linked  {src}  →  data/raw/{our_name}  ({n:,} files)")
    else:
        print(f"  MISSING {src}  — did you add it as an Input?")

print("\ndata/raw/ contents:")
for name in sorted(os.listdir(RAW)):
    print(f"  {name}")
```

> If a dataset shows **MISSING**, go back to the right sidebar → "Add Input"
> and search for it by name, then re-run this cell.

---

**Cell 3 — Train the model**

```python
# ============================================================
# CELL 3: Full training — Phase 1 (LeJEPA) + Phase 2 (V-JEPA)
# ============================================================
# Trains from scratch on all linked datasets.
# Settings tuned for T4 (16 GB VRAM):
#   batch_size 64 + gradient_accumulation 4 = effective batch 256
#   50 epochs ≈ 5-7 hours on T4
#
# Checkpoints saved every 5 epochs to /kaggle/working/checkpoints/

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

> If you get "CUDA out of memory" → change `--batch_size 64` to `--batch_size 32`
> and `--gradient_accumulation_steps 4` to `--gradient_accumulation_steps 8`.

---

**Cell 4 — Save checkpoint to Output**

```python
# ============================================================
# CELL 4: Copy checkpoint to Output tab for download
# ============================================================
import shutil, os, glob

# Find best checkpoint
candidates = (
    glob.glob('/kaggle/working/checkpoints/best_model.pt') +
    sorted(glob.glob('/kaggle/working/checkpoints/checkpoint_epoch_*.pt'))
)

if candidates:
    src = candidates[0]
    dst = '/kaggle/working/best_model.pt'
    shutil.copy(src, dst)
    print(f"Saved: {dst}  ({os.path.getsize(dst)/1e6:.0f} MB)")
    print("Download from the Output tab →")
else:
    print("No checkpoint found — did Cell 3 finish?")
    print("Files in checkpoints/:")
    for f in glob.glob('/kaggle/working/checkpoints/*'):
        print(f"  {f}")
```

### 1D. After Session 1 Ends

1. Click **"Output"** tab (right sidebar) → download **`best_model.pt`**
2. Upload it as a new Kaggle Dataset:
   - [kaggle.com/datasets/new](https://www.kaggle.com/datasets/new)
   - Name: `medjepa-checkpoint`
   - Upload `best_model.pt` → **Create**

---

## Session 2 — Evaluate + Visualize (~3-4 hours)

### 2A. Create a New Notebook

1. **"New Notebook"** → GPU T4, Internet ON
2. **"Add Input"** → add the same 3 datasets as Session 1
3. **"Add Input"** → search `medjepa-checkpoint` (your own dataset) → add it
4. Rename to: `MedJEPA-Session2-Evaluate`

### 2B. Paste These Cells

---

**Cell 1 — Clone and install**

```python
# Same as Session 1 Cell 1
import os, shutil, subprocess

os.chdir('/')
for stale in ['/kaggle/working/MedJEPA',
              '/kaggle/working/pip_cache',
              '/kaggle/working/tmp']:
    if os.path.exists(stale):
        shutil.rmtree(stale, ignore_errors=True)

subprocess.run(['df', '-h', '/kaggle/working'], check=False)

os.makedirs('/kaggle/working/tmp', exist_ok=True)
os.environ['GIT_TMPDIR'] = '/kaggle/working/tmp'

!git clone --depth 1 https://github.com/prthmmkhija1/MedJEPA.git /kaggle/working/MedJEPA
%cd /kaggle/working/MedJEPA
!pip install -e ".[medical]" -q --cache-dir /kaggle/working/pip_cache
!pip install gdown -q --cache-dir /kaggle/working/pip_cache
print("Done!")
```

---

**Cell 2 — Load checkpoint**

```python
# ============================================================
# CELL 2: Copy checkpoint from your medjepa-checkpoint dataset
# ============================================================
import os, shutil, glob

os.makedirs('checkpoints', exist_ok=True)

found = (glob.glob('/kaggle/input/medjepa-checkpoint/best_model.pt') +
         glob.glob('/kaggle/input/*/best_model.pt') +
         glob.glob('/kaggle/input/**/*.pt', recursive=True))

if found:
    shutil.copy(found[0], 'checkpoints/best_model.pt')
    print(f"Loaded: {found[0]}  ({os.path.getsize('checkpoints/best_model.pt')/1e6:.0f} MB)")
else:
    print("ERROR: no checkpoint found in /kaggle/input/")
    print("Contents of /kaggle/input/:")
    for root, dirs, files in os.walk('/kaggle/input/'):
        for f in files:
            print(' ', os.path.join(root, f))
```

---

**Cell 3 — Link datasets (same as Session 1 Cell 2)**

```python
import os

RAW = '/kaggle/working/MedJEPA/data/raw'
os.makedirs(RAW, exist_ok=True)

DATASETS = {
    'skin-cancer-mnist-ham10000':    'ham10000',
    'aptos2019-blindness-detection': 'aptos2019',
    'brats-2021-task1':              'brats',
}

for kaggle_name, our_name in DATASETS.items():
    src = f'/kaggle/input/{kaggle_name}'
    dst = f'{RAW}/{our_name}'
    if os.path.exists(src):
        if os.path.islink(dst):
            os.remove(dst)
        os.symlink(src, dst)
        print(f"  linked  data/raw/{our_name}")
    else:
        print(f"  MISSING {kaggle_name}")
```

---

**Cell 4 — Run full evaluation**

```python
# ============================================================
# CELL 4: Run all evaluations with the corrected kNN
# ============================================================
# Skips pretraining (already done), runs:
#   linear probe, few-shot kNN, fine-tune, segmentation
# Takes ~2-3 hours on T4.

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

---

**Cell 5 — Print results**

```python
import json

with open('results/evaluation_results.json') as f:
    results = json.load(f)

print("=" * 65)
print("RESULTS SUMMARY")
print("=" * 65)

print("\n── Linear Probe ──────────────────────────────────────────")
for name, res in results.items():
    lp = res.get('linear_probing', {})
    if 'accuracy' in lp:
        print(f"  {name:25s}  Acc={lp['accuracy']:.1%}  AUC={lp.get('auc', 0):.3f}")

print("\n── N-Shot ────────────────────────────────────────────────")
for name, res in results.items():
    ns = res.get('n_shot', {})
    if ns:
        shots = [f"{k}={v['accuracy']:.1%}" for k, v in ns.items() if 'accuracy' in v]
        if shots:
            print(f"  {name:25s}  {',  '.join(shots)}")

print("\n── Fine-Tune vs ImageNet ─────────────────────────────────")
for name, res in results.items():
    ft  = res.get('fine_tuning', {}).get('accuracy')
    inet = res.get('imagenet_baseline', {}).get('accuracy')
    if ft and inet:
        win = "MedJEPA wins" if ft > inet else "ImageNet leads"
        print(f"  {name:25s}  FT={ft:.1%}  ImageNet={inet:.1%}  [{win}]")
```

---

**Cell 6 — Generate plots**

```python
!pip install nbconvert ipykernel -q
!python -m ipykernel install --user --name python3 2>/dev/null
!jupyter nbconvert --execute --to notebook --inplace \
    --ExecutePreprocessor.timeout=900 \
    notebooks/05_results_analysis.ipynb
```

---

**Cell 7 — Package everything**

```python
import shutil, os

OUT = '/kaggle/working'
shutil.copy('results/evaluation_results.json', f'{OUT}/evaluation_results.json')
for f in os.listdir('results'):
    if f.endswith(('.png', '.csv')):
        shutil.copy(f'results/{f}', f'{OUT}/{f}')
shutil.copy('notebooks/05_results_analysis.ipynb', f'{OUT}/05_results_analysis.ipynb')
if os.path.exists('checkpoints/best_model.pt'):
    shutil.copy('checkpoints/best_model.pt', f'{OUT}/best_model.pt')

shutil.make_archive(f'{OUT}/medjepa_final', 'zip', OUT)
print("Done! Download medjepa_final.zip from the Output tab.")
```

---

## After Kaggle — Final Steps

### 1. Copy files back to your PC

Extract `medjepa_final.zip` and copy into the project:

```bash
cd f:\Projects\MedJEPA

copy "C:\Users\prath\Downloads\medjepa_final\evaluation_results.json" results\evaluation_results.json
copy "C:\Users\prath\Downloads\medjepa_final\*.png" results\
copy "C:\Users\prath\Downloads\medjepa_final\05_results_analysis.ipynb" notebooks\05_results_analysis.ipynb
copy "C:\Users\prath\Downloads\medjepa_final\best_model.pt" C:\Users\prath\Desktop\best_model.pt
```

### 2. Update README few-shot table

Open `results/evaluation_results.json`, read the `n_shot` numbers, find the
few-shot table in `README.md` and replace the placeholder percentages.

### 3. Commit and push

```bash
git add results/ notebooks/05_results_analysis.ipynb README.md
git commit -m "Add corrected evaluation results and visualization plots"
git push
```

### 4. Publish checkpoint as GitHub Release

1. Go to `https://github.com/prthmmkhija1/MedJEPA/releases/new`
2. Tag: `v0.1.0`, Title: `MedJEPA v0.1.0 — Pre-trained Checkpoint`
3. Attach `best_model.pt` from your Desktop
4. Publish

---

## Troubleshooting

### A dataset shows "MISSING" in Cell 2

You forgot to add it as an Input. In the notebook sidebar:
1. Click **"Add Input"**
2. Search the slug (e.g. `skin-cancer-mnist-ham10000`)
3. Click **"Add"**
4. Re-run Cell 2

### "CUDA out of memory"

Halve the batch size and double accumulation steps:

```python
!python scripts/run_gpu_full.py \
    --epochs 50 --batch_size 32 --gradient_accumulation_steps 8 \
    --num_workers 2 --checkpoint_dir /kaggle/working/checkpoints \
    --results_dir /kaggle/working/results
```

### Session times out mid-training

The script saves every 5 epochs. Find the latest checkpoint in the Output tab
(e.g. `checkpoint_epoch_45.pt`), upload it as a dataset, then in Session 2:

```python
!python scripts/run_gpu_full.py \
    --skip_pretrain --skip_vjepa \
    --checkpoint checkpoints/checkpoint_epoch_45.pt \
    --batch_size 64 --num_workers 2
```

### "getcwd / No such file or directory"

The notebook kernel's working directory was deleted. Fix:

```python
import os; os.chdir('/')
```

Then re-run Cell 1.

---

## Time Estimate

| Step | Time |
|---|:---:|
| Add datasets as Inputs (one-time) | 5 min |
| **Session 1:** Cell 1 clone + install | 5 min |
| **Session 1:** Cell 2 link datasets | <1 min |
| **Session 1:** Cell 3 train 50 epochs | **5-7 hours** |
| Download checkpoint, upload as dataset | 10 min |
| **Session 2:** Cell 1-3 setup | 5 min |
| **Session 2:** Cell 4 full evaluation | **2-3 hours** |
| **Session 2:** Plots + package | 15 min |
| Copy back, update README, publish release | 20 min |
| **Total** | **~10-12 hours** |

---

## Checklist

- [ ] HAM10000, APTOS, BraTS added as Kaggle Inputs
- [ ] Session 1 ran, `best_model.pt` downloaded
- [ ] Checkpoint uploaded as `medjepa-checkpoint` dataset
- [ ] Session 2 ran, `evaluation_results.json` has all metrics
- [ ] PNG plots generated in `results/`
- [ ] README few-shot table updated with real numbers
- [ ] GitHub Release `v0.1.0` has `best_model.pt` attached
- [ ] All 24 tests pass locally (`python -m pytest tests/ -v`)
