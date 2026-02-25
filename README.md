# MedJEPA: Complete End-to-End Project Guide

## (Self-Supervised Medical Image Representation Learning with JEPA)

> **Written for absolute beginners** — no prior ML knowledge assumed.
> This guide tells you EXACTLY what to do, step by step, cell by cell.

---

## Your Setup Reference

Throughout this guide, we'll use these labels for your 3 machines:

| Label       | Machine                      | Key Specs                     | Use For                                    |
| ----------- | ---------------------------- | ----------------------------- | ------------------------------------------ |
| **HP-Lite** | HP 15s (integrated graphics) | CPU only, limited RAM         | Coding, reading papers, light testing      |
| **HP-GPU**  | HP with NVIDIA graphics      | Has CUDA GPU                  | Medium experiments, testing training loops |
| **Mac-M4**  | Mac M4                       | Apple Silicon, unified memory | Development, medium experiments via MPS    |
| **Cloud**   | Google Colab / Lab cluster   | Powerful GPUs (free or paid)  | Full pre-training, heavy evaluation        |

> **Rule of thumb**: Write code on HP-Lite. Test on HP-GPU or Mac-M4. Train for real on Cloud.

---

# TABLE OF CONTENTS

1. [Phase 0: Understanding What We're Building](#phase-0-understanding-what-were-building)
2. [Phase 1: Setting Up Your Machines](#phase-1-setting-up-your-machines)
3. [Phase 2: Reading the Research Papers](#phase-2-reading-the-research-papers)
4. [Phase 3: Project Skeleton & GitHub Setup](#phase-3-project-skeleton--github-setup)
5. [Phase 4: Medical Data — Getting & Understanding It](#phase-4-medical-data--getting--understanding-it)
6. [Phase 5: Data Preprocessing Pipeline](#phase-5-data-preprocessing-pipeline)
7. [Phase 6: Building the Masking System](#phase-6-building-the-masking-system)
8. [Phase 7: Building the LeJEPA Model (2D Images)](#phase-7-building-the-lejepa-model-2d-images)
9. [Phase 8: Building the V-JEPA Extension (3D/Video)](#phase-8-building-the-v-jepa-extension-3dvideo)
10. [Phase 9: Training Infrastructure](#phase-9-training-infrastructure)
11. [Phase 10: Pre-Training the Model](#phase-10-pre-training-the-model)
12. [Phase 11: Evaluation — Disease Classification](#phase-11-evaluation--disease-classification)
13. [Phase 12: Evaluation — Detection & Segmentation](#phase-12-evaluation--detection--segmentation)
14. [Phase 13: Evaluation — Clinical Predictions & Few-Shot](#phase-13-evaluation--clinical-predictions--few-shot)
15. [Phase 14: Interpretability & Visualizations](#phase-14-interpretability--visualizations)
16. [Phase 15: Final Packaging & Open-Source Release](#phase-15-final-packaging--open-source-release)

---

# Phase 0: Understanding What We're Building

## What Is This Project in Plain English?

Hospitals have MILLIONS of medical images (X-rays, CT scans, MRIs, etc.) sitting in their systems.
The problem? Almost none of them are "labeled" — meaning no doctor has sat down and written
"this X-ray shows pneumonia" or "this MRI shows a brain tumor" for each image.

Traditional AI needs those labels to learn. Without labels, traditional AI is useless.

**MedJEPA solves this.**

It teaches a computer to understand medical images WITHOUT needing any labels.
How? By playing a game:

1. Take a medical image
2. Hide (mask) some parts of it
3. Ask the model: "What do you think is in the hidden parts?"
4. The model doesn't guess the exact pixels — it guesses the **meaning/concept**
5. After doing this millions of times, the model deeply understands medical images

Once it understands images, you can then use a TINY number of labeled images (like 50 instead of 50,000)
to teach it specific tasks like "detect pneumonia" or "find tumors."

## What Exactly Will We Build?

```
MedJEPA/
│
├── Data pipelines      → Code to download, clean, and prepare medical images
├── LeJEPA model        → The AI model for 2D images (X-rays, skin photos, eye photos)
├── V-JEPA extension    → The AI model for 3D data (CT scans, MRI) and video (surgical videos)
├── Training scripts    → Code to actually train the model
├── Evaluation scripts  → Code to test how good the model is
├── Pre-trained models  → Saved model files others can download and use
└── Documentation       → Tutorials and guides
```

## Key Terms You'll See Everywhere

| Term                           | What It Means                                                                                                                                    |
| ------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------ |
| **Self-supervised learning**   | Learning from data WITHOUT labels. The data itself creates the learning signal.                                                                  |
| **Pre-training**               | The first phase where the model learns general understanding from lots of unlabeled data.                                                        |
| **Fine-tuning**                | The second phase where you teach the pre-trained model a specific task using a small labeled dataset.                                            |
| **Encoder**                    | A neural network that takes an image and converts it into a list of numbers (called "embeddings" or "representations") that capture the meaning. |
| **Embeddings/Representations** | A list of numbers that represents the meaning of an image. Similar images have similar numbers.                                                  |
| **JEPA**                       | Joint-Embedding Predictive Architecture — predicts MEANING of hidden parts, not pixels.                                                          |
| **LeJEPA**                     | A clean, math-proven version of JEPA for images. Uses SIGReg. No tricks needed.                                                                  |
| **V-JEPA**                     | JEPA for video and 3D data.                                                                                                                      |
| **SIGReg**                     | A rule that prevents the model from cheating (giving the same boring answer for everything).                                                     |
| **Masking**                    | Hiding parts of an image so the model can practice predicting what's there.                                                                      |
| **Vision Transformer (ViT)**   | A type of neural network architecture (think: the "brain structure" of the model). Processes images as small patches.                            |
| **Linear probing**             | A test: freeze the model, add one simple layer on top, see how well it classifies. Shows how good the learned representations are.               |
| **Few-shot learning**          | Learning a new task from very few examples (like 5 or 10 labeled images).                                                                        |
| **DICOM**                      | The standard file format for medical images (like .jpg but for hospitals).                                                                       |
| **NIfTI**                      | Another medical image format, common for brain scans and 3D volumes.                                                                             |

---

# Phase 1: Setting Up Your Machines

## Cell 1.1: Setting Up HP-Lite (Your Main Development Machine)

> **Do this on: HP-Lite (HP 15s with integrated graphics)**

### Step 1: Install Python

Go to [python.org](https://www.python.org/downloads/) and download Python 3.10 or 3.11.

During installation:

- CHECK the box that says **"Add Python to PATH"** (very important!)
- Click "Install Now"

Verify it worked — open Command Prompt (search "cmd" in Start menu) and type:

```bash
python --version
```

You should see something like `Python 3.10.x` or `Python 3.11.x`.

### Step 2: Install Git

Go to [git-scm.com](https://git-scm.com/download/win) and download Git for Windows.
Install with default settings.

Verify:

```bash
git --version
```

### Step 3: Install VS Code

Go to [code.visualstudio.com](https://code.visualstudio.com/) and install it.
Open it, then install these extensions (click the square icon on the left sidebar):

- **Python** (by Microsoft)
- **Jupyter** (by Microsoft)
- **GitLens** (by GitKraken)

### Step 4: Create the Project Folder and Virtual Environment

Open a terminal in VS Code (Terminal > New Terminal) and run these ONE BY ONE:

```bash
# Go to where you want the project
cd F:\Projects\MedJEPA

# Create a virtual environment (an isolated Python space for this project)
python -m venv venv

# Activate the virtual environment
# On Windows:
venv\Scripts\activate

# You should now see (venv) at the start of your terminal line
```

### Step 5: Install Basic Libraries (CPU-only version for HP-Lite)

```bash
# Upgrade pip first
pip install --upgrade pip

# Install PyTorch (CPU version — no GPU on this machine)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Install essential libraries
pip install numpy pandas matplotlib seaborn scikit-learn tqdm

# Install Jupyter for notebooks
pip install jupyter jupyterlab ipykernel

# Install medical imaging libraries
pip install pydicom        # For reading DICOM files (medical image format)
pip install nibabel        # For reading NIfTI files (brain scan format)
pip install SimpleITK      # Advanced medical image processing
pip install monai          # NVIDIA's medical imaging AI toolkit
pip install albumentations # Image augmentation library
pip install opencv-python  # Computer vision basics
pip install Pillow         # Image handling

# Install experiment tracking
pip install wandb          # Track your experiments online (free)
pip install tensorboard    # Visualize training progress

# Install code quality tools
pip install black flake8 pytest

# Save all installed packages to a file (so other machines can install the same thing)
pip freeze > requirements.txt
```

### Step 6: Verify Everything Works

Create a file called `test_setup.py` and paste this:

```python
import torch
import numpy as np
import pydicom
import nibabel
import monai
import cv2
from PIL import Image

print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")  # Will say False on HP-Lite, that's OK
print(f"NumPy version: {np.__version__}")
print(f"MONAI version: {monai.__version__}")
print(f"OpenCV version: {cv2.__version__}")

# Quick test: create a random tensor
x = torch.randn(2, 3, 224, 224)  # Simulating a batch of 2 images, 3 color channels, 224x224 pixels
print(f"\nTest tensor shape: {x.shape}")
print("All imports successful! HP-Lite is ready for development.")
```

Run it:

```bash
python test_setup.py
```

---

## Cell 1.2: Setting Up HP-GPU (Your NVIDIA Machine)

> **Do this on: HP-GPU (HP with NVIDIA graphics)**

Repeat Steps 1-3 from Cell 1.1 above (Python, Git, VS Code).

### Step 4: Check Your NVIDIA GPU

Open Command Prompt and run:

```bash
nvidia-smi
```

This will show your GPU name, memory, and driver version.
**Write down your GPU name and memory** — you'll need this later.

### Step 5: Install CUDA Toolkit

Go to [developer.nvidia.com/cuda-downloads](https://developer.nvidia.com/cuda-downloads)
and download CUDA 11.8 or 12.1 (match what PyTorch supports).

### Step 6: Create Environment and Install (GPU version)

```bash
cd <your-project-path>\MedJEPA
python -m venv venv
venv\Scripts\activate

pip install --upgrade pip

# Install PyTorch WITH CUDA support (change cu118 to match your CUDA version)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install the same libraries as HP-Lite
pip install numpy pandas matplotlib seaborn scikit-learn tqdm
pip install jupyter jupyterlab ipykernel
pip install pydicom nibabel SimpleITK monai albumentations opencv-python Pillow
pip install wandb tensorboard
pip install black flake8 pytest
```

### Step 7: Verify GPU is Detected

```python
import torch
print(f"CUDA available: {torch.cuda.is_available()}")      # Should say True
print(f"GPU name: {torch.cuda.get_device_name(0)}")         # Should show your GPU name
print(f"GPU memory: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB")
```

---

## Cell 1.3: Setting Up Mac M4

> **Do this on: Mac-M4**

### Step 1: Install Homebrew (Mac's package manager)

Open Terminal (Cmd + Space, type "Terminal") and run:

```bash
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
```

### Step 2: Install Python and Git

```bash
brew install python@3.11 git
```

### Step 3: Install VS Code

Download from [code.visualstudio.com](https://code.visualstudio.com/) — get the Apple Silicon version.

### Step 4: Create Environment and Install

```bash
cd ~/Projects/MedJEPA    # or wherever your project is
python3 -m venv venv
source venv/bin/activate

pip install --upgrade pip

# Install PyTorch (Apple Silicon MPS version — comes by default now)
pip install torch torchvision torchaudio

# Install everything else (same as HP-Lite)
pip install numpy pandas matplotlib seaborn scikit-learn tqdm
pip install jupyter jupyterlab ipykernel
pip install pydicom nibabel SimpleITK monai albumentations opencv-python Pillow
pip install wandb tensorboard
pip install black flake8 pytest
```

### Step 5: Verify MPS (Apple GPU) is Detected

```python
import torch
print(f"MPS available: {torch.backends.mps.is_available()}")  # Should say True
print(f"MPS built: {torch.backends.mps.is_built()}")          # Should say True

# Test: move a tensor to Apple GPU
x = torch.randn(2, 3, 224, 224).to("mps")
print(f"Tensor device: {x.device}")  # Should say "mps:0"
print("Mac M4 is ready!")
```

---

## Cell 1.4: Setting Up Google Colab (Free Cloud GPU)

> **Do this on: Any machine with a web browser**

1. Go to [colab.research.google.com](https://colab.research.google.com)
2. Sign in with your Google account
3. Click **Runtime** > **Change runtime type** > Select **GPU** (T4 is free)
4. Create a new notebook

In the first cell of any Colab notebook, always run:

```python
# Check what GPU Colab gave you
!nvidia-smi

import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"GPU: {torch.cuda.get_device_name(0)}")

# Install medical imaging libraries (not pre-installed in Colab)
!pip install -q pydicom nibabel SimpleITK monai albumentations wandb
```

### How to Sync Code Between Machines

The trick: use **GitHub** as your central hub.

```
HP-Lite (write code) → push to GitHub → pull on HP-GPU/Mac/Colab
```

We'll set this up in Phase 3.

---

# Phase 2: Reading the Research Papers

> **Do this on: HP-Lite (just reading, no compute needed)**

Before writing any code, you MUST understand the theory. Don't skip this.

## Cell 2.1: Paper Reading Order

Read in this EXACT order (each builds on the previous):

### Paper 1: I-JEPA (The Foundation)

- **Title**: "Self-Supervised Learning from Images with a Joint-Embedding Predictive Architecture"
- **Authors**: Mahmoud Assran et al., CVPR 2023
- **Link**: https://arxiv.org/abs/2301.08243
- **What to focus on**:
  - Figure 1 — the overall architecture diagram
  - How masking works (context blocks and target blocks)
  - The predictor network
  - How it differs from MAE (Masked Autoencoder)
- **Time**: 2-3 days to read and understand

### Paper 2: LeJEPA (The Core of Our Project)

- **Title**: "LeJEPA: Provable and Scalable Self-Supervised Learning Without the Heuristics"
- **Authors**: Randall Balestriero and Yann LeCun, arXiv 2024
- **Link**: Search on arxiv.org for "LeJEPA Balestriero LeCun"
- **What to focus on**:
  - What SIGReg is and why it matters
  - How it eliminates momentum encoders and stop-gradients
  - The single trade-off hyperparameter
  - The loss function
- **Time**: 3-4 days (this is the most important paper)

### Paper 3: V-JEPA (Video/3D Extension)

- **Title**: "Revisiting Feature Prediction for Learning Visual Representations from Video"
- **Authors**: Adrien Bardes et al., arXiv 2024
- **Link**: https://arxiv.org/abs/2401.12178
- **What to focus on**:
  - How masking extends to time/video
  - Spatiotemporal encoding
  - How it handles temporal information
- **Time**: 2-3 days

## Cell 2.2: Background Knowledge to Build Up

If ANY of the papers feel confusing, study these topics first (YouTube is great for these):

```
Week 1: Python & PyTorch basics
  - PyTorch official tutorials (pytorch.org/tutorials)
  - YouTube: "PyTorch in 60 minutes" by Patrick Loeber

Week 2: How Neural Networks learn
  - YouTube: 3Blue1Brown "Neural Networks" series (4 videos)
  - YouTube: "What is backpropagation?" by 3Blue1Brown

Week 3: Convolutional Neural Networks (CNNs)
  - YouTube: "CNNs explained" by deeplizard
  - Why they matter: most image models use CNN ideas

Week 4: Vision Transformers (ViT)
  - YouTube: "Vision Transformer explained" by Yannic Kilcher
  - Paper: "An Image is Worth 16x16 Words" (Dosovitskiy et al.)
  - KEY CONCEPTS: patches, positional encoding, attention mechanism

Week 5: Self-Supervised Learning overview
  - YouTube: "Self-Supervised Learning" by Yann LeCun (his talks)
  - Understand contrastive learning (SimCLR) vs predictive (JEPA)
```

## Cell 2.3: Notes Template

Create this file to take notes while reading:

```
File: notes/paper_notes.md

For each paper, answer:
1. What problem does this solve?
2. How did people solve it before? (previous methods)
3. What's new/different about this approach?
4. What are the main components? (draw diagrams!)
5. What is the loss function? (the math that tells the model if it's doing well)
6. What results did they get?
7. What parts are relevant to medical imaging?
```

---

# Phase 3: Project Skeleton & GitHub Setup

> **Do this on: HP-Lite**

## Cell 3.1: Create the GitHub Repository

> **Note**: You already have `F:\Projects\MedJEPA` from Phase 1 (with your venv and installed packages).
> We will NOT clone — we'll initialize git inside the existing folder and connect it to GitHub.

1. Go to [github.com](https://github.com) and create an account (if you don't have one)
2. Click **"New Repository"**
   - Name: `MedJEPA`
   - Description: "Self-Supervised Medical Image Representation Learning with JEPA"
   - Public (required for open-source project)
   - **DO NOT** check "Add README" (we already have files locally)
   - **DO NOT** add `.gitignore` (we'll create it ourselves)
   - **DO NOT** add a License yet (we'll add it later)
   - This creates an **empty** repository on GitHub
3. Initialize git in your existing project folder:

```bash
# If using Git Bash (MINGW64):
cd /f/Projects/MedJEPA

# If using PowerShell or CMD:
# cd F:\Projects\MedJEPA

# Initialize a git repository in the existing folder
git init

# Connect it to your GitHub repository
git remote add origin https://github.com/YOUR-USERNAME/MedJEPA.git

# Set main as the default branch
git branch -M main
```

## Cell 3.2: Create the Project Folder Structure

Run these commands to create all the folders:

```bash
# On Windows (HP-Lite):
mkdir configs
mkdir data
mkdir data\raw
mkdir data\processed
mkdir data\scripts
mkdir medjepa
mkdir medjepa\models
mkdir medjepa\data
mkdir medjepa\training
mkdir medjepa\evaluation
mkdir medjepa\utils
mkdir scripts
mkdir notebooks
mkdir tests
mkdir checkpoints
mkdir results
mkdir docs
mkdir notes
```

Your project should now look like this:

```
MedJEPA/
├── configs/                  # Configuration files (settings for experiments)
│   ├── base_config.yaml      # Default settings
│   ├── chest_xray.yaml       # Settings specific to chest X-ray experiments
│   ├── histopathology.yaml   # Settings for histopathology experiments
│   └── brain_mri.yaml        # Settings for brain MRI experiments
│
├── data/                     # All data-related stuff
│   ├── raw/                  # Original downloaded datasets (DON'T MODIFY THESE)
│   ├── processed/            # Cleaned, preprocessed data ready for training
│   └── scripts/              # Scripts to download and prepare datasets
│
├── medjepa/                  # THE MAIN CODE (this is a Python package)
│   ├── __init__.py           # Makes this folder a Python package
│   ├── models/               # Neural network architectures
│   │   ├── __init__.py
│   │   ├── encoder.py        # The encoder (converts images → embeddings)
│   │   ├── predictor.py      # The predictor (predicts hidden parts)
│   │   ├── lejepa.py         # LeJEPA model for 2D images
│   │   └── vjepa.py          # V-JEPA model for 3D/video
│   ├── data/                 # Data loading and processing code
│   │   ├── __init__.py
│   │   ├── datasets.py       # Dataset classes for each medical imaging type
│   │   ├── preprocessing.py  # Image cleaning and normalization
│   │   ├── masking.py        # Masking strategies (hiding parts of images)
│   │   └── dicom_utils.py    # DICOM file reading utilities
│   ├── training/             # Training loop code
│   │   ├── __init__.py
│   │   ├── trainer.py        # Main training loop
│   │   ├── losses.py         # Loss functions (including SIGReg)
│   │   └── scheduler.py      # Learning rate scheduling
│   ├── evaluation/           # Testing and evaluation code
│   │   ├── __init__.py
│   │   ├── linear_probe.py   # Linear probing evaluation
│   │   ├── few_shot.py       # Few-shot evaluation
│   │   ├── segmentation.py   # Segmentation evaluation
│   │   └── metrics.py        # Accuracy, AUC, Dice score, etc.
│   └── utils/                # Helper functions
│       ├── __init__.py
│       ├── device.py         # Auto-detect GPU/MPS/CPU
│       ├── logging.py        # Logging utilities
│       └── visualization.py  # Plotting and attention maps
│
├── scripts/                  # Standalone scripts to run things
│   ├── pretrain.py           # Run pre-training
│   ├── evaluate.py           # Run evaluation
│   ├── download_data.py      # Download datasets
│   └── visualize.py          # Generate visualizations
│
├── notebooks/                # Jupyter notebooks for exploration
│   ├── 01_explore_data.ipynb
│   ├── 02_test_masking.ipynb
│   ├── 03_test_model.ipynb
│   └── 04_results_analysis.ipynb
│
├── tests/                    # Unit tests
│   ├── test_encoder.py
│   ├── test_masking.py
│   └── test_dataloader.py
│
├── checkpoints/              # Saved model files (NOT pushed to GitHub — too large)
├── results/                  # Evaluation results, plots, tables
├── docs/                     # Documentation and tutorials
├── notes/                    # Your personal notes
│
├── .gitignore                # Files Git should ignore
├── README.md                 # Project description
├── requirements.txt          # Python dependencies
├── setup.py                  # Package installation file
└── LICENSE                   # Open source license
```

## Cell 3.3: Create Essential Files

### Create `__init__.py` files (makes folders into Python packages):

These files can be empty initially. Create an `__init__.py` in:

- `medjepa/`
- `medjepa/models/`
- `medjepa/data/`
- `medjepa/training/`
- `medjepa/evaluation/`
- `medjepa/utils/`

### Create the device helper (auto-detects your hardware):

File: `medjepa/utils/device.py`

```python
"""
Auto-detect the best available device across all your machines.
- HP-Lite: will use CPU
- HP-GPU: will use CUDA (NVIDIA GPU)
- Mac-M4: will use MPS (Apple GPU)
- Colab: will use CUDA
"""

import torch

def get_device():
    """Automatically pick the best device available."""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using NVIDIA GPU: {torch.cuda.get_device_name(0)}")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using Apple MPS (Metal Performance Shaders)")
    else:
        device = torch.device("cpu")
        print("Using CPU (no GPU detected)")
    return device

def get_device_info():
    """Print detailed info about available compute."""
    print("=" * 50)
    print("DEVICE INFORMATION")
    print("=" * 50)
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        mem = torch.cuda.get_device_properties(0).total_mem
        print(f"  Memory: {mem / 1e9:.1f} GB")
    mps_available = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    print(f"MPS available: {mps_available}")
    print(f"Selected device: {get_device()}")
    print("=" * 50)
```

### Create `.gitignore` additions:

Add these lines to your `.gitignore`:

```
# Data (too large for GitHub)
data/raw/
data/processed/

# Model checkpoints (too large for GitHub)
checkpoints/

# Python
__pycache__/
*.pyc
venv/
.egg-info/

# Jupyter
.ipynb_checkpoints/

# OS files
.DS_Store
Thumbs.db

# IDE
.vscode/
.idea/

# Logs
wandb/
runs/
*.log
```

## Cell 3.4: Push to GitHub and Set Up Multi-Machine Workflow

```bash
# On HP-Lite: push your initial structure (first time only)
git add .
git commit -m "Initial structure"
git push -u origin main
```

### Setting Up Other Machines (HP-GPU / Mac-M4)

On your OTHER machines, NOW you clone (because the folder doesn't exist there yet):

```bash
# On HP-GPU or Mac-M4 (first time only):
git clone https://github.com/YOUR-USERNAME/MedJEPA.git
cd MedJEPA
# Then create venv and install packages as shown in Phase 1
# (each machine gets its own venv — venv is NOT pushed to GitHub)
```

### Daily Workflow Across Machines:

```bash
# BEFORE starting work on any machine:
git pull origin main

# AFTER finishing work:
git add .
git commit -m "Describe what you changed"
git push origin main
```

---

# Phase 4: Medical Data — Getting & Understanding It

> **Do this on: HP-Lite (downloading), later move datasets to other machines as needed**

## Cell 4.1: Understanding Medical Image Formats

### What is DICOM?

DICOM = Digital Imaging and Communications in Medicine.
It's the standard file format hospitals use for medical images.

Think of it like this:

- A `.jpg` file = just the image
- A `.dcm` (DICOM) file = the image + patient info + scan settings + hospital info + timestamps

The extra information is called **metadata**. When you open a DICOM file, you get BOTH the image pixels AND all this information.

**Privacy Warning**: DICOM files contain patient names, IDs, dates of birth, etc.
You MUST anonymize (remove personal info from) DICOM files before using them.

### What is NIfTI?

NIfTI = Neuroimaging Informatics Technology Initiative.
Used mainly for brain scans and 3D medical images.

- File extension: `.nii` or `.nii.gz` (compressed)
- Contains 3D volumetric data (like a stack of 2D slices forming a 3D cube)
- Less metadata than DICOM, usually already anonymized

## Cell 4.2: Essential Datasets (6 — Covers All Required Modalities & Tasks)

These 6 datasets are the **minimum necessary** to complete all evaluation tasks
described in the [UCSC OSPO project description](https://ucsc-ospo.github.io/project/osre26/nelbl/medjepa/).
They cover every modality (2D, 3D) and every evaluation task (classification,
segmentation, few-shot) the project requires.

### 2D Datasets (Pre-training + Classification Evaluation)

| #   | Dataset          | Modality       | What It Contains                            | Size   | Eval Task                            | How to Get It                                                               | Difficulty |
| --- | ---------------- | -------------- | ------------------------------------------- | ------ | ------------------------------------ | --------------------------------------------------------------------------- | ---------- |
| 1   | **HAM10000**     | Dermatology    | 10,015 skin lesion photos                   | ~3 GB  | 7-class skin lesion classification   | [Kaggle](https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000) | Easy       |
| 2   | **PCam**         | Histopathology | 327,680 histopathology patches              | ~8 GB  | Patch-level cancer classification    | [Kaggle](https://www.kaggle.com/c/histopathologic-cancer-detection)         | Easy       |
| 3   | **APTOS 2019**   | Retinal        | 3,662 retinal fundus images                 | ~10 GB | 5-class diabetic retinopathy grading | [Kaggle](https://www.kaggle.com/c/aptos2019-blindness-detection)            | Easy       |
| 4   | **ChestX-ray14** | Chest X-ray    | 112,120 chest X-rays with 14 disease labels | ~45 GB | 14-class multi-label classification  | [NIH Box](https://nihcc.app.box.com/v/ChestXray-NIHCC)                      | Medium     |

### 3D Datasets (V-JEPA Extension + Segmentation Evaluation)

| #   | Dataset                            | Modality    | What It Contains           | Size   | Eval Task                           | How to Get It                                                        | Difficulty |
| --- | ---------------------------------- | ----------- | -------------------------- | ------ | ----------------------------------- | -------------------------------------------------------------------- | ---------- |
| 5   | **BraTS**                          | Brain MRI   | Brain tumor MRI scans (3D) | ~20 GB | Tumor classification + segmentation | [Synapse](https://www.synapse.org/#!Synapse:syn27046444/wiki/616571) | Medium     |
| 6   | **Medical Segmentation Decathlon** | Multi-organ | 10 segmentation tasks (3D) | ~30 GB | Multi-organ segmentation            | [medicaldecathlon.com](http://medicaldecathlon.com/)                 | Easy       |

### Why These 6 and Not More?

| Skipped Dataset      | Why Not Needed                                           |
| -------------------- | -------------------------------------------------------- |
| MIMIC-CXR / CheXpert | ChestX-ray14 already covers chest X-rays                 |
| Camelyon16/17        | ~700 GB — PCam covers histopathology at 1/90th the size  |
| EyePACS              | ~80 GB — APTOS covers retinal images at 1/8th the size   |
| ISIC                 | HAM10000 is derived from ISIC (same data, easier to get) |
| LIDC-IDRI            | Med Seg Decathlon includes lung tasks                    |
| ACDC                 | Add later if time permits (cardiac MRI)                  |

> **Total storage needed**: ~116 GB across all 6 datasets.

### Recommended Download Order:

```
Phase 1: HAM10000 + APTOS + PCam     (~21 GB — all from Kaggle, build your pipeline here)
Phase 2: ChestX-ray14                (~45 GB — THE benchmark, start downloading early)
Phase 3: BraTS + Med Seg Decathlon   (~50 GB — 3D data for V-JEPA extension)
```

## Cell 4.3: Downloading Your First Dataset (HAM10000)

> **Do this on: HP-Lite**

### Option A: Download from Kaggle

1. Go to [kaggle.com](https://www.kaggle.com) and create account
2. Go to Account Settings → API → Create New Token
3. This downloads `kaggle.json` — put it in `C:\Users\YOUR_NAME\.kaggle\`

```bash
# Install Kaggle CLI
pip install kaggle

# Download HAM10000
kaggle datasets download -d kmader/skin-cancer-mnist-ham10000 -p data/raw/ham10000

# Unzip it
cd data/raw/ham10000
# On Windows:
tar -xf skin-cancer-mnist-ham10000.zip
```

### Option B: Manual Download

1. Go to the Kaggle link
2. Click "Download" button
3. Unzip to `data/raw/ham10000/`

## Cell 4.4: Explore Your First Dataset

Create a Jupyter notebook: `notebooks/01_explore_data.ipynb`

```python
# Cell 1: Imports
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from pathlib import Path
```

```python
# Cell 2: Look at what files we have
data_dir = Path("../data/raw/ham10000")

# List all files (show first 20)
all_files = list(data_dir.rglob("*"))
print(f"Total files: {len(all_files)}")
for f in all_files[:20]:
    print(f"  {f.name} ({f.stat().st_size / 1024:.0f} KB)")
```

```python
# Cell 3: Load and look at the metadata CSV
metadata = pd.read_csv(data_dir / "HAM10000_metadata.csv")
print(f"Shape: {metadata.shape}")
print(f"\nColumns: {list(metadata.columns)}")
print(f"\nFirst 5 rows:")
metadata.head()
```

```python
# Cell 4: What types of skin conditions are in this dataset?
print("Disease Distribution:")
print(metadata['dx'].value_counts())

# Plot it
metadata['dx'].value_counts().plot(kind='bar', figsize=(10, 5))
plt.title("HAM10000 Disease Distribution")
plt.ylabel("Count")
plt.tight_layout()
plt.show()
```

```python
# Cell 5: Look at some actual images
fig, axes = plt.subplots(2, 5, figsize=(20, 8))
axes = axes.flatten()

for i, (_, row) in enumerate(metadata.sample(10, random_state=42).iterrows()):
    # Find the image file
    image_id = row['image_id']
    # HAM10000 images might be in different folders — search for them
    img_files = list(data_dir.rglob(f"{image_id}.jpg"))
    if img_files:
        img = Image.open(img_files[0])
        axes[i].imshow(img)
        axes[i].set_title(f"{row['dx']}\n{img.size}", fontsize=10)
    axes[i].axis('off')

plt.suptitle("Sample HAM10000 Images", fontsize=16)
plt.tight_layout()
plt.show()
```

```python
# Cell 6: Check image properties
from collections import Counter

sizes = []
for img_path in list(data_dir.rglob("*.jpg"))[:100]:  # Check first 100
    with Image.open(img_path) as img:
        sizes.append(img.size)

size_counts = Counter(sizes)
print("Image sizes found:")
for size, count in size_counts.most_common():
    print(f"  {size}: {count} images")
```

---

# Phase 5: Data Preprocessing Pipeline

> **Do this on: HP-Lite (coding) → test on HP-GPU or Mac-M4 for speed**

## Cell 5.1: Understanding Why We Preprocess

Medical images come in all shapes and sizes:

- Different resolutions (some X-rays are 1024x1024, others are 2048x2048)
- Different intensity ranges (some pixels go from 0-255, others from 0-65535)
- Different formats (DICOM, PNG, JPEG, NIfTI)
- Different orientations

We need to make them all UNIFORM so the model can process them consistently.

## Cell 5.2: Build the 2D Image Preprocessor

File: `medjepa/data/preprocessing.py`

```python
"""
Preprocessing pipeline for medical images.
Handles different formats and normalizes everything to a consistent format.
"""

import numpy as np
from PIL import Image
from pathlib import Path
import pydicom
from typing import Tuple, Optional


class MedicalImagePreprocessor:
    """
    Takes any medical image and converts it to a clean, normalized format.

    What "normalization" means:
    - Resize all images to the same size (e.g., 224x224)
    - Scale pixel values to 0-1 range (instead of 0-255 or 0-65535)
    - Handle grayscale vs color images consistently
    """

    def __init__(self, target_size: Tuple[int, int] = (224, 224)):
        """
        Args:
            target_size: What size to make all images. (224, 224) is standard for ViT.
        """
        self.target_size = target_size

    def load_image(self, path: str) -> np.ndarray:
        """
        Load a medical image from any common format.

        Args:
            path: Path to the image file

        Returns:
            numpy array of the image pixels
        """
        path = Path(path)
        suffix = path.suffix.lower()

        if suffix == ".dcm":
            # DICOM format (hospital standard)
            return self._load_dicom(path)
        elif suffix in [".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"]:
            # Regular image formats
            return self._load_standard_image(path)
        elif suffix in [".nii", ".gz"]:
            # NIfTI format (brain scans)
            return self._load_nifti(path)
        else:
            raise ValueError(f"Unknown image format: {suffix}")

    def _load_dicom(self, path: Path) -> np.ndarray:
        """Load a DICOM file and extract the pixel data."""
        ds = pydicom.dcmread(str(path))
        pixel_array = ds.pixel_array.astype(np.float32)
        return pixel_array

    def _load_standard_image(self, path: Path) -> np.ndarray:
        """Load a standard image file (JPG, PNG, etc.)."""
        img = Image.open(path)
        return np.array(img, dtype=np.float32)

    def _load_nifti(self, path: Path) -> np.ndarray:
        """Load a NIfTI file (3D brain/body scans)."""
        import nibabel as nib
        nii = nib.load(str(path))
        return nii.get_fdata().astype(np.float32)

    def normalize_intensity(self, image: np.ndarray) -> np.ndarray:
        """
        Scale pixel values to 0-1 range.

        Why: Different scanners produce different value ranges.
        An X-ray from Hospital A might have values 0-4095,
        while Hospital B produces 0-65535.
        Normalizing to 0-1 makes them comparable.
        """
        # Handle edge case: image is all one value
        img_min = image.min()
        img_max = image.max()
        if img_max - img_min == 0:
            return np.zeros_like(image)
        return (image - img_min) / (img_max - img_min)

    def resize_image(self, image: np.ndarray) -> np.ndarray:
        """Resize image to target size."""
        pil_image = Image.fromarray(
            (image * 255).astype(np.uint8) if image.max() <= 1.0
            else image.astype(np.uint8)
        )
        pil_image = pil_image.resize(self.target_size, Image.LANCZOS)
        return np.array(pil_image, dtype=np.float32) / 255.0

    def ensure_3_channels(self, image: np.ndarray) -> np.ndarray:
        """
        Make sure image has 3 color channels (RGB).

        Why: X-rays are grayscale (1 channel), skin photos are color (3 channels).
        The model expects a consistent number of channels.
        We convert grayscale to 3 channels by repeating the single channel 3 times.
        """
        if image.ndim == 2:
            # Grayscale → repeat to make 3 channels
            image = np.stack([image, image, image], axis=-1)
        elif image.ndim == 3 and image.shape[-1] == 1:
            image = np.repeat(image, 3, axis=-1)
        elif image.ndim == 3 and image.shape[-1] == 4:
            # RGBA → drop alpha channel
            image = image[:, :, :3]
        return image

    def preprocess(self, path: str) -> np.ndarray:
        """
        Full preprocessing pipeline: load → normalize → resize → ensure 3 channels.

        Args:
            path: Path to any medical image file

        Returns:
            numpy array of shape (224, 224, 3) with values in [0, 1]
        """
        image = self.load_image(path)
        image = self.normalize_intensity(image)
        image = self.resize_image(image)
        image = self.ensure_3_channels(image)
        return image
```

## Cell 5.3: Build the 3D Volume Preprocessor

File: `medjepa/data/preprocessing.py` (add to the same file)

```python
class VolumetricPreprocessor:
    """
    Preprocessor for 3D medical data (CT scans, MRI volumes).

    A CT scan is like a stack of 2D X-ray slices.
    Imagine slicing a loaf of bread — each slice is a 2D image,
    and the full loaf is a 3D volume.
    """

    def __init__(
        self,
        target_size: Tuple[int, int, int] = (128, 128, 64),
        # width=128, height=128, depth=64 slices
    ):
        self.target_size = target_size

    def load_nifti_volume(self, path: str) -> np.ndarray:
        """Load a full 3D volume from a NIfTI file."""
        import nibabel as nib
        nii = nib.load(path)
        volume = nii.get_fdata().astype(np.float32)
        return volume

    def normalize_volume(self, volume: np.ndarray) -> np.ndarray:
        """Normalize the entire 3D volume to 0-1 range."""
        v_min, v_max = volume.min(), volume.max()
        if v_max - v_min == 0:
            return np.zeros_like(volume)
        return (volume - v_min) / (v_max - v_min)

    def resize_volume(self, volume: np.ndarray) -> np.ndarray:
        """Resize 3D volume to target size using simple interpolation."""
        from scipy.ndimage import zoom

        current_shape = volume.shape
        zoom_factors = [
            t / c for t, c in zip(self.target_size, current_shape[:3])
        ]
        resized = zoom(volume, zoom_factors, order=1)  # Linear interpolation
        return resized

    def preprocess(self, path: str) -> np.ndarray:
        """Full 3D preprocessing pipeline."""
        volume = self.load_nifti_volume(path)
        volume = self.normalize_volume(volume)
        volume = self.resize_volume(volume)
        return volume
```

## Cell 5.4: Build the Dataset Class

File: `medjepa/data/datasets.py`

```python
"""
PyTorch Dataset classes for loading medical images.

A "Dataset" in PyTorch is an object that knows:
1. How many images there are
2. How to load image number N

PyTorch then uses this to efficiently feed batches of images to the model.
"""

import torch
from torch.utils.data import Dataset
from pathlib import Path
import pandas as pd
import numpy as np
from typing import Optional, List, Callable
from medjepa.data.preprocessing import MedicalImagePreprocessor


class MedicalImageDataset(Dataset):
    """
    A general dataset class for 2D medical images.
    Works for chest X-rays, skin photos, retinal images, etc.
    """

    def __init__(
        self,
        image_dir: str,
        metadata_csv: Optional[str] = None,
        image_column: str = "image_id",
        label_column: Optional[str] = None,
        file_extension: str = ".jpg",
        target_size: tuple = (224, 224),
        transform: Optional[Callable] = None,
    ):
        """
        Args:
            image_dir: Folder containing the images
            metadata_csv: Optional CSV with image IDs and labels
            image_column: Column name in CSV that has image file names
            label_column: Column name in CSV that has labels (None for self-supervised)
            file_extension: What type of image files to look for
            target_size: Size to resize images to
            transform: Optional extra transformations (augmentations)
        """
        self.image_dir = Path(image_dir)
        self.target_size = target_size
        self.transform = transform
        self.preprocessor = MedicalImagePreprocessor(target_size=target_size)

        # Find all images
        if metadata_csv:
            self.metadata = pd.read_csv(metadata_csv)
            self.image_files = [
                self._find_image(row[image_column], file_extension)
                for _, row in self.metadata.iterrows()
            ]
            if label_column and label_column in self.metadata.columns:
                self.labels = self.metadata[label_column].values
            else:
                self.labels = None
        else:
            # No CSV — just find all image files in the folder
            self.image_files = sorted(
                self.image_dir.rglob(f"*{file_extension}")
            )
            self.labels = None

    def _find_image(self, image_id: str, ext: str) -> Path:
        """Find an image file by its ID."""
        # Try with and without extension
        candidates = list(self.image_dir.rglob(f"{image_id}{ext}"))
        if not candidates:
            candidates = list(self.image_dir.rglob(f"{image_id}.*"))
        return candidates[0] if candidates else None

    def __len__(self):
        """How many images in this dataset."""
        return len(self.image_files)

    def __getitem__(self, idx):
        """
        Load image number 'idx'.
        This is called by PyTorch's DataLoader.
        """
        image_path = self.image_files[idx]
        if image_path is None or not image_path.exists():
            # Return a blank image if file not found
            image = np.zeros((*self.target_size, 3), dtype=np.float32)
        else:
            image = self.preprocessor.preprocess(str(image_path))

        # Convert to PyTorch tensor
        # PyTorch expects shape (channels, height, width), not (height, width, channels)
        image = torch.from_numpy(image).permute(2, 0, 1)  # HWC → CHW

        if self.transform:
            image = self.transform(image)

        if self.labels is not None:
            label = self.labels[idx]
            return image, label
        else:
            return image
```

## Cell 5.5: Build the DICOM Anonymization Tool

File: `medjepa/data/dicom_utils.py`

```python
"""
Utilities for working with DICOM medical images.
Includes anonymization (removing patient information for privacy).
"""

import pydicom
from pathlib import Path
import numpy as np
from typing import List


# These DICOM tags contain personal information that MUST be removed
TAGS_TO_ANONYMIZE = [
    "PatientName",
    "PatientID",
    "PatientBirthDate",
    "PatientSex",
    "PatientAge",
    "PatientAddress",
    "InstitutionName",
    "InstitutionAddress",
    "ReferringPhysicianName",
    "PerformingPhysicianName",
    "StudyDate",
    "SeriesDate",
    "AcquisitionDate",
    "ContentDate",
    "AccessionNumber",
]


def anonymize_dicom(input_path: str, output_path: str):
    """
    Remove all personal/identifying information from a DICOM file.

    IMPORTANT: This MUST be done before using any hospital DICOM data.
    Laws like HIPAA (USA) and GDPR (Europe) require this.

    Args:
        input_path: Path to original DICOM file
        output_path: Path where the anonymized file will be saved
    """
    ds = pydicom.dcmread(input_path)

    for tag_name in TAGS_TO_ANONYMIZE:
        if hasattr(ds, tag_name):
            setattr(ds, tag_name, "ANONYMIZED")

    ds.save_as(output_path)
    print(f"Anonymized: {input_path} → {output_path}")


def extract_pixel_data(dicom_path: str) -> np.ndarray:
    """
    Extract just the image pixels from a DICOM file.
    Useful when you want to save as PNG/NPY and discard the metadata entirely.
    """
    ds = pydicom.dcmread(dicom_path)
    return ds.pixel_array.astype(np.float32)


def get_dicom_info(dicom_path: str) -> dict:
    """
    Get useful (non-personal) information from a DICOM file.
    Things like image size, scan type, etc.
    """
    ds = pydicom.dcmread(dicom_path)
    info = {
        "rows": getattr(ds, "Rows", None),
        "columns": getattr(ds, "Columns", None),
        "modality": getattr(ds, "Modality", None),  # CT, MR, CR, etc.
        "bits_stored": getattr(ds, "BitsStored", None),
        "pixel_spacing": getattr(ds, "PixelSpacing", None),
        "slice_thickness": getattr(ds, "SliceThickness", None),
    }
    return info
```

## Cell 5.6: Test the Preprocessing Pipeline

Create notebook: `notebooks/02_test_preprocessing.ipynb`

```python
# Cell 1: Test on HAM10000 images
import sys
sys.path.append("..")

from medjepa.data.preprocessing import MedicalImagePreprocessor
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
```

```python
# Cell 2: Preprocess some images
preprocessor = MedicalImagePreprocessor(target_size=(224, 224))

# Find some HAM10000 images
image_dir = Path("../data/raw/ham10000")
images = list(image_dir.rglob("*.jpg"))[:5]

fig, axes = plt.subplots(2, 5, figsize=(20, 8))

for i, img_path in enumerate(images):
    # Show original
    from PIL import Image
    orig = Image.open(img_path)
    axes[0][i].imshow(orig)
    axes[0][i].set_title(f"Original\n{orig.size}")

    # Show preprocessed
    processed = preprocessor.preprocess(str(img_path))
    axes[1][i].imshow(processed)
    axes[1][i].set_title(f"Preprocessed\n{processed.shape}")

axes[0][0].set_ylabel("Original", fontsize=14)
axes[1][0].set_ylabel("Preprocessed", fontsize=14)
plt.suptitle("Preprocessing Test", fontsize=16)
plt.tight_layout()
plt.show()
```

```python
# Cell 3: Test the dataset class
from medjepa.data.datasets import MedicalImageDataset
from torch.utils.data import DataLoader

dataset = MedicalImageDataset(
    image_dir="../data/raw/ham10000",
    file_extension=".jpg",
    target_size=(224, 224),
)

print(f"Dataset size: {len(dataset)} images")

# Load one image
sample = dataset[0]
print(f"Image tensor shape: {sample.shape}")    # Should be (3, 224, 224)
print(f"Value range: {sample.min():.3f} to {sample.max():.3f}")  # Should be near 0-1

# Test DataLoader (loads images in batches)
loader = DataLoader(dataset, batch_size=16, shuffle=True, num_workers=0)
batch = next(iter(loader))
print(f"Batch shape: {batch.shape}")  # Should be (16, 3, 224, 224)
```

---

# Phase 6: Building the Masking System

> **Do this on: HP-Lite (coding) → visualize on any machine**

## Cell 6.1: What is Masking and Why?

Masking is the core of JEPA's self-supervised learning.

Imagine a chest X-ray. We:

1. Divide it into a grid of small patches (like cutting a photo into puzzle pieces)
2. HIDE some patches (the "target" — what the model must predict)
3. SHOW the remaining patches (the "context" — what the model can see)
4. Ask the model: "Based on the patches you CAN see, predict the MEANING of the hidden patches"

Different masking strategies for different data types:

- **2D images**: Hide random rectangles or patches
- **3D volumes**: Hide random cubes
- **Video/sequences**: Hide random time chunks

## Cell 6.2: Implement the Masking Module

File: `medjepa/data/masking.py`

```python
"""
Masking strategies for JEPA self-supervised learning.

The model sees some patches (context) and must predict the hidden patches (target).
This file implements different ways to choose which patches to hide.
"""

import torch
import numpy as np
from typing import Tuple, List, Optional


class PatchMasker2D:
    """
    Masking for 2D medical images.

    An image of size 224x224, divided into 16x16 patches, gives a 14x14 grid
    of patches (224/16 = 14). That's 196 total patches.

    We might hide 75% of them (147 patches) and show 25% (49 patches).
    """

    def __init__(
        self,
        image_size: int = 224,
        patch_size: int = 16,
        mask_ratio: float = 0.75,
        # How much of the image to hide. 0.75 = hide 75%.
        num_target_blocks: int = 4,
        # How many separate rectangular blocks to hide
    ):
        self.image_size = image_size
        self.patch_size = patch_size
        self.grid_size = image_size // patch_size  # e.g., 224/16 = 14
        self.num_patches = self.grid_size ** 2      # e.g., 14*14 = 196
        self.mask_ratio = mask_ratio
        self.num_target_blocks = num_target_blocks

    def generate_mask(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate context (visible) and target (hidden) patch indices.

        Returns:
            context_indices: Which patches the model can see
            target_indices: Which patches the model must predict
        """
        num_masked = int(self.num_patches * self.mask_ratio)
        num_visible = self.num_patches - num_masked

        # Randomly shuffle all patch indices
        all_indices = np.random.permutation(self.num_patches)

        # Split into visible (context) and hidden (target)
        context_indices = torch.tensor(sorted(all_indices[:num_visible]))
        target_indices = torch.tensor(sorted(all_indices[num_visible:]))

        return context_indices, target_indices

    def generate_block_mask(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate block-style masking (like I-JEPA).
        Instead of random patches, hides rectangular BLOCKS.

        This is more natural — in a chest X-ray, hiding a rectangular region
        forces the model to understand spatial relationships (e.g., if you
        hide the left lung, the model must understand anatomy to predict it).
        """
        mask = np.zeros((self.grid_size, self.grid_size), dtype=bool)
        total_to_mask = int(self.num_patches * self.mask_ratio)
        masked_so_far = 0

        for _ in range(self.num_target_blocks):
            remaining = total_to_mask - masked_so_far
            if remaining <= 0:
                break

            # Random block size
            block_h = np.random.randint(
                self.grid_size // 4, self.grid_size // 2 + 1
            )
            block_w = np.random.randint(
                self.grid_size // 4, self.grid_size // 2 + 1
            )

            # Random position
            top = np.random.randint(0, self.grid_size - block_h + 1)
            left = np.random.randint(0, self.grid_size - block_w + 1)

            # Apply mask
            mask[top:top + block_h, left:left + block_w] = True
            masked_so_far = mask.sum()

        # Convert 2D mask to 1D patch indices
        mask_flat = mask.flatten()
        target_indices = torch.tensor(np.where(mask_flat)[0])
        context_indices = torch.tensor(np.where(~mask_flat)[0])

        return context_indices, target_indices

    def visualize_mask(self, context_indices, target_indices):
        """
        Create a visual representation of the mask for debugging.
        Returns a grid where 0=visible, 1=hidden.
        """
        grid = np.zeros(self.num_patches)
        grid[target_indices.numpy()] = 1
        return grid.reshape(self.grid_size, self.grid_size)


class PatchMasker3D:
    """
    Masking for 3D medical volumes (CT, MRI).

    A 3D volume has width, height, AND depth (number of slices).
    We divide it into 3D cubes (like small dice) and hide some of them.
    """

    def __init__(
        self,
        volume_size: Tuple[int, int, int] = (128, 128, 64),
        patch_size: Tuple[int, int, int] = (16, 16, 8),
        mask_ratio: float = 0.75,
    ):
        self.volume_size = volume_size
        self.patch_size = patch_size
        self.grid_size = tuple(
            v // p for v, p in zip(volume_size, patch_size)
        )
        self.num_patches = (
            self.grid_size[0] * self.grid_size[1] * self.grid_size[2]
        )
        self.mask_ratio = mask_ratio

    def generate_mask(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generate random 3D mask."""
        num_masked = int(self.num_patches * self.mask_ratio)
        all_indices = np.random.permutation(self.num_patches)

        context_indices = torch.tensor(sorted(all_indices[num_masked:]))
        target_indices = torch.tensor(sorted(all_indices[:num_masked]))

        return context_indices, target_indices


class TemporalMasker:
    """
    Masking for medical video/sequences.
    Hides entire time frames or temporal chunks.

    Used for: cardiac MRI sequences, surgical videos, ultrasound clips.
    """

    def __init__(
        self,
        num_frames: int = 16,
        patch_size_spatial: int = 16,
        image_size: int = 224,
        mask_ratio_temporal: float = 0.5,  # Hide 50% of frames
        mask_ratio_spatial: float = 0.75,  # Also hide 75% of patches in remaining frames
    ):
        self.num_frames = num_frames
        self.grid_size = image_size // patch_size_spatial
        self.num_spatial_patches = self.grid_size ** 2
        self.mask_ratio_temporal = mask_ratio_temporal
        self.mask_ratio_spatial = mask_ratio_spatial

    def generate_mask(self) -> dict:
        """Generate spatiotemporal mask."""
        # Temporal: which frames to hide
        num_hidden_frames = int(self.num_frames * self.mask_ratio_temporal)
        frame_order = np.random.permutation(self.num_frames)
        hidden_frames = sorted(frame_order[:num_hidden_frames])
        visible_frames = sorted(frame_order[num_hidden_frames:])

        # Spatial: in visible frames, which patches to show
        num_visible_patches = int(
            self.num_spatial_patches * (1 - self.mask_ratio_spatial)
        )
        spatial_indices = np.random.permutation(self.num_spatial_patches)
        visible_patches = sorted(spatial_indices[:num_visible_patches])

        return {
            "visible_frames": torch.tensor(visible_frames),
            "hidden_frames": torch.tensor(hidden_frames),
            "visible_patches": torch.tensor(visible_patches),
        }
```

## Cell 6.3: Visualize the Masking

Create notebook: `notebooks/02_test_masking.ipynb`

```python
# Cell 1: Setup
import sys
sys.path.append("..")

from medjepa.data.masking import PatchMasker2D
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
```

```python
# Cell 2: Test random masking
masker = PatchMasker2D(image_size=224, patch_size=16, mask_ratio=0.75)

fig, axes = plt.subplots(1, 4, figsize=(20, 5))
for i in range(4):
    ctx, tgt = masker.generate_mask()
    grid = masker.visualize_mask(ctx, tgt)
    axes[i].imshow(grid, cmap="RdYlGn_r", vmin=0, vmax=1)
    axes[i].set_title(f"Mask {i+1}\nVisible: {len(ctx)}, Hidden: {len(tgt)}")
    axes[i].grid(True, linewidth=0.5)

plt.suptitle("Random Masking (Green = Visible, Red = Hidden)", fontsize=14)
plt.tight_layout()
plt.show()
```

```python
# Cell 3: Test block masking (like I-JEPA)
fig, axes = plt.subplots(1, 4, figsize=(20, 5))
for i in range(4):
    ctx, tgt = masker.generate_block_mask()
    grid = masker.visualize_mask(ctx, tgt)
    axes[i].imshow(grid, cmap="RdYlGn_r", vmin=0, vmax=1)
    axes[i].set_title(f"Block Mask {i+1}\nVisible: {len(ctx)}, Hidden: {len(tgt)}")
    axes[i].grid(True, linewidth=0.5)

plt.suptitle("Block Masking (Green = Visible, Red = Hidden)", fontsize=14)
plt.tight_layout()
plt.show()
```

```python
# Cell 4: Show masking on a real medical image
from medjepa.data.preprocessing import MedicalImagePreprocessor
from pathlib import Path

preprocessor = MedicalImagePreprocessor(target_size=(224, 224))

# Load a sample image (use any image you have)
sample_images = list(Path("../data/raw/ham10000").rglob("*.jpg"))
if sample_images:
    img = preprocessor.preprocess(str(sample_images[0]))

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Original
    axes[0].imshow(img)
    axes[0].set_title("Original Image")

    # Apply mask visually
    ctx, tgt = masker.generate_block_mask()
    mask_grid = masker.visualize_mask(ctx, tgt)

    # Create masked version
    masked_img = img.copy()
    patch_size = 16
    for patch_idx in tgt.numpy():
        row = (patch_idx // 14) * patch_size
        col = (patch_idx % 14) * patch_size
        masked_img[row:row+patch_size, col:col+patch_size] = 0.5  # Gray out

    axes[1].imshow(masked_img)
    axes[1].set_title("What the model SEES (context)")

    # Show target regions
    target_img = np.ones_like(img) * 0.3
    for patch_idx in tgt.numpy():
        row = (patch_idx // 14) * patch_size
        col = (patch_idx % 14) * patch_size
        target_img[row:row+patch_size, col:col+patch_size] = img[row:row+patch_size, col:col+patch_size]

    axes[2].imshow(target_img)
    axes[2].set_title("What the model must PREDICT (target)")

    plt.suptitle("JEPA Masking on a Medical Image", fontsize=14)
    plt.tight_layout()
    plt.show()
```

---

# Phase 7: Building the LeJEPA Model (2D Images)

> **Do this on: HP-Lite (coding) → test forward pass on HP-GPU or Mac-M4**

## Cell 7.1: The Architecture in Plain English

LeJEPA has 3 main parts:

```
                    CONTEXT PATCHES                TARGET PATCHES
                    (what model sees)              (what model predicts)
                          │                               │
                          ▼                               ▼
                    ┌──────────┐                   ┌──────────┐
                    │  ENCODER │                   │  ENCODER │  (same encoder!)
                    └────┬─────┘                   └────┬─────┘
                         │                               │
                    context                          target
                    embeddings                      embeddings
                         │                               │
                         ▼                               │
                    ┌───────────┐                        │
                    │ PREDICTOR │                        │
                    └─────┬─────┘                        │
                          │                              │
                    predicted target                     │
                    embeddings                           │
                          │                              │
                          ▼                              ▼
                    ┌─────────────────────────────────────┐
                    │   LOSS: predicted should match      │
                    │   actual target embeddings          │
                    │   + SIGReg regularization           │
                    └─────────────────────────────────────┘
```

Important: BOTH the context and target use the **SAME encoder**.
There's no momentum encoder, no teacher-student — just ONE encoder.
This is what makes LeJEPA "heuristics-free."

## Cell 7.2: Build the Encoder

File: `medjepa/models/encoder.py`

```python
"""
Encoder: Takes image patches and converts them into meaningful embeddings.

Uses Vision Transformer (ViT) architecture.

Think of it like this:
1. Cut the image into small patches (like 16x16 pixel squares)
2. Flatten each patch into a 1D vector
3. Add position information (so the model knows where each patch was)
4. Run through Transformer layers (these learn relationships between patches)
5. Output: one embedding vector per patch
"""

import torch
import torch.nn as nn
import math
from typing import Optional


class PatchEmbedding(nn.Module):
    """
    Step 1: Cut image into patches and project each patch to embedding dimension.

    Example: A 224x224 image with 16x16 patches becomes 14x14 = 196 patches.
    Each 16x16x3 patch (768 numbers) gets projected to, say, 768 numbers.
    """

    def __init__(
        self,
        image_size: int = 224,
        patch_size: int = 16,
        in_channels: int = 3,      # 3 for RGB
        embed_dim: int = 768,      # Size of each embedding vector
    ):
        super().__init__()
        self.patch_size = patch_size
        self.num_patches = (image_size // patch_size) ** 2
        self.embed_dim = embed_dim

        # This conv layer does the patch cutting AND projection in one step
        # It's a clever trick: a convolution with kernel_size=patch_size
        # and stride=patch_size is equivalent to cutting into non-overlapping
        # patches and projecting each one
        self.projection = nn.Conv2d(
            in_channels, embed_dim,
            kernel_size=patch_size, stride=patch_size
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Images, shape (batch_size, 3, 224, 224)
        Returns:
            Patch embeddings, shape (batch_size, num_patches, embed_dim)
        """
        # x shape: (B, 3, 224, 224) → (B, embed_dim, 14, 14)
        x = self.projection(x)
        # Flatten spatial dimensions: (B, embed_dim, 14, 14) → (B, embed_dim, 196)
        x = x.flatten(2)
        # Transpose: (B, embed_dim, 196) → (B, 196, embed_dim)
        x = x.transpose(1, 2)
        return x


class TransformerBlock(nn.Module):
    """
    One layer of the Transformer.

    Each block does:
    1. Self-attention: each patch looks at all other patches to understand context
    2. Feed-forward: process the information through a small neural network
    3. Residual connections and layer normalization for stable training
    """

    def __init__(
        self,
        embed_dim: int = 768,
        num_heads: int = 12,     # Number of attention "perspectives"
        mlp_ratio: float = 4.0,  # Feed-forward layer is 4x wider
        dropout: float = 0.0,
    ):
        super().__init__()

        # Layer normalization (keeps values in a nice range)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)

        # Multi-head self-attention
        self.attention = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=dropout, batch_first=True
        )

        # Feed-forward network (MLP)
        mlp_dim = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_dim),
            nn.GELU(),  # Activation function
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: shape (batch_size, num_patches, embed_dim)
        Returns:
            Same shape, but with refined representations
        """
        # Self-attention with residual connection
        normed = self.norm1(x)
        attended, _ = self.attention(normed, normed, normed)
        x = x + attended

        # Feed-forward with residual connection
        x = x + self.mlp(self.norm2(x))
        return x


class ViTEncoder(nn.Module):
    """
    The full Vision Transformer encoder.

    Puts together: Patch Embedding + Positional Encoding + Transformer Blocks
    """

    def __init__(
        self,
        image_size: int = 224,
        patch_size: int = 16,
        in_channels: int = 3,
        embed_dim: int = 768,
        depth: int = 12,         # Number of Transformer blocks
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        num_patches = (image_size // patch_size) ** 2

        # Step 1: Patch embedding
        self.patch_embed = PatchEmbedding(
            image_size, patch_size, in_channels, embed_dim
        )

        # Step 2: Positional encoding
        # Each patch gets a learnable position vector so the model
        # knows WHERE in the image each patch is
        self.pos_embed = nn.Parameter(
            torch.randn(1, num_patches, embed_dim) * 0.02
        )

        # Step 3: Stack of Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, mlp_ratio, dropout)
            for _ in range(depth)
        ])

        # Final normalization
        self.norm = nn.LayerNorm(embed_dim)

    def forward(
        self,
        x: torch.Tensor,
        patch_indices: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x: Images, shape (batch_size, 3, 224, 224)
            patch_indices: Optional — which patches to process
                           (used for masking: only encode visible patches)

        Returns:
            Embeddings, shape (batch_size, num_patches, embed_dim)
        """
        # Convert image to patch embeddings
        x = self.patch_embed(x)

        # Add positional encoding
        if patch_indices is not None:
            # Only add positions for selected patches
            pos = self.pos_embed[:, patch_indices, :]
            x = x[:, patch_indices, :] if x.shape[1] != len(patch_indices) else x
            x = x + pos
        else:
            x = x + self.pos_embed

        # Run through Transformer blocks
        for block in self.blocks:
            x = block(x)

        # Final normalization
        x = self.norm(x)
        return x
```

## Cell 7.3: Build the Predictor

File: `medjepa/models/predictor.py`

```python
"""
Predictor: Takes context embeddings and predicts target embeddings.

This is a smaller, simpler network than the encoder.
It takes the embeddings from visible patches and predicts
what the embeddings of the hidden patches should be.
"""

import torch
import torch.nn as nn
from typing import Optional


class JEPAPredictor(nn.Module):
    """
    Predicts the embeddings of hidden (target) patches
    given the embeddings of visible (context) patches.

    Architecture: A small Transformer that:
    1. Takes context embeddings as input
    2. Adds learnable "mask tokens" for target positions
    3. Uses attention to predict target embeddings from context
    """

    def __init__(
        self,
        embed_dim: int = 768,    # Must match encoder output dim
        predictor_dim: int = 384, # Predictor is usually smaller
        depth: int = 6,           # Fewer layers than encoder
        num_heads: int = 6,
        num_patches: int = 196,  # Total patches in the image
    ):
        super().__init__()

        # Project from encoder dimension to (smaller) predictor dimension
        self.input_proj = nn.Linear(embed_dim, predictor_dim)

        # Learnable mask tokens — these stand in for the hidden patches
        # The model will transform these into predictions
        self.mask_token = nn.Parameter(torch.randn(1, 1, predictor_dim) * 0.02)

        # Position embeddings for the predictor
        self.pos_embed = nn.Parameter(
            torch.randn(1, num_patches, predictor_dim) * 0.02
        )

        # Transformer blocks for prediction
        self.blocks = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=predictor_dim,
                nhead=num_heads,
                dim_feedforward=predictor_dim * 4,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            for _ in range(depth)
        ])

        self.norm = nn.LayerNorm(predictor_dim)

        # Project back to encoder dimension for loss computation
        self.output_proj = nn.Linear(predictor_dim, embed_dim)

    def forward(
        self,
        context_embeddings: torch.Tensor,
        context_indices: torch.Tensor,
        target_indices: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            context_embeddings: Encoder output for visible patches
                                Shape: (batch_size, num_context, embed_dim)
            context_indices: Which patches are visible (1D tensor of indices)
            target_indices: Which patches are hidden (1D tensor of indices)

        Returns:
            Predicted embeddings for target patches
            Shape: (batch_size, num_target, embed_dim)
        """
        batch_size = context_embeddings.shape[0]
        num_target = len(target_indices)

        # Project context to predictor dimension
        context = self.input_proj(context_embeddings)

        # Add positional encoding to context
        context = context + self.pos_embed[:, context_indices, :]

        # Create mask tokens for target positions
        targets = self.mask_token.expand(batch_size, num_target, -1)
        targets = targets + self.pos_embed[:, target_indices, :]

        # Concatenate context and target tokens
        # The predictor sees both, but only the target tokens need to be predicted
        full_sequence = torch.cat([context, targets], dim=1)

        # Run through Transformer blocks
        for block in self.blocks:
            full_sequence = block(full_sequence)

        # Extract only the target predictions (last num_target tokens)
        target_predictions = full_sequence[:, -num_target:, :]

        # Normalize and project back to encoder dimension
        target_predictions = self.norm(target_predictions)
        target_predictions = self.output_proj(target_predictions)

        return target_predictions
```

## Cell 7.4: Build the SIGReg Loss and Full LeJEPA Model

File: `medjepa/training/losses.py`

```python
"""
Loss functions for MedJEPA training.

The "loss" is a number that tells the model how wrong it is.
Lower loss = better predictions. The model tries to minimize this number.

SIGReg = Sketched Isotropic Gaussian Regularization
This is the KEY innovation of LeJEPA. It prevents "collapse" — when the model
cheats by making all embeddings identical (which would be useless).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SIGRegLoss(nn.Module):
    """
    Sketched Isotropic Gaussian Regularization loss.

    Has two parts:
    1. PREDICTION LOSS: The predicted target embeddings should match
       the actual target embeddings (MSE — Mean Squared Error).

    2. REGULARIZATION: The embeddings should be spread out (not all the same).
       Specifically, the distribution of embeddings should look like a
       "nice" Gaussian (bell curve) — spread evenly in all directions.

    The single trade-off hyperparameter (lambda_reg) balances these two goals.
    """

    def __init__(
        self,
        lambda_reg: float = 1.0,
        # The ONE hyperparameter: how much to emphasize regularization
        # vs prediction accuracy. Start with 1.0.
    ):
        super().__init__()
        self.lambda_reg = lambda_reg

    def prediction_loss(
        self,
        predicted: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        """
        How different are the predictions from the actual embeddings?
        Uses MSE (Mean Squared Error) — the average squared difference.

        Args:
            predicted: What the predictor thinks the target embeddings are
            target: What the target embeddings actually are (from the encoder)
        """
        # Normalize both to unit length (makes comparison direction-based, not magnitude-based)
        predicted = F.normalize(predicted, dim=-1)
        target = F.normalize(target, dim=-1)

        # MSE loss
        loss = F.mse_loss(predicted, target)
        return loss

    def regularization_loss(
        self,
        embeddings: torch.Tensor,
    ) -> torch.Tensor:
        """
        Prevent collapse: make sure embeddings are diverse and spread out.

        We want the covariance matrix of embeddings to look like an identity matrix.
        (Identity matrix = each dimension is independent and equally important.)

        If all embeddings were the same, the covariance would be all zeros.
        By pushing it toward identity, we force diversity.
        """
        batch_size, num_tokens, embed_dim = embeddings.shape

        # Reshape: combine batch and token dimensions
        flat = embeddings.reshape(-1, embed_dim)

        # Center the embeddings (subtract mean)
        flat = flat - flat.mean(dim=0, keepdim=True)

        # Compute covariance matrix
        # (what correlations exist between embedding dimensions?)
        n = flat.shape[0]
        cov = (flat.T @ flat) / (n - 1)

        # We want cov to be close to the identity matrix
        # Loss = how far is cov from identity?
        identity = torch.eye(embed_dim, device=embeddings.device)
        reg_loss = F.mse_loss(cov, identity)

        return reg_loss

    def forward(
        self,
        predicted_target: torch.Tensor,
        actual_target: torch.Tensor,
        all_embeddings: torch.Tensor,
    ) -> dict:
        """
        Compute the total LeJEPA loss.

        Args:
            predicted_target: Predictor's guess for target embeddings
            actual_target: Encoder's actual target embeddings
            all_embeddings: All embeddings (for regularization)

        Returns:
            Dictionary with total loss and components
        """
        pred_loss = self.prediction_loss(predicted_target, actual_target)
        reg_loss = self.regularization_loss(all_embeddings)

        total_loss = pred_loss + self.lambda_reg * reg_loss

        return {
            "total_loss": total_loss,
            "prediction_loss": pred_loss,
            "regularization_loss": reg_loss,
        }
```

## Cell 7.5: Assemble the Complete LeJEPA Model

File: `medjepa/models/lejepa.py`

```python
"""
LeJEPA: The Complete Model for 2D Medical Images.

This puts together the encoder, predictor, masking, and loss into one model.
"""

import torch
import torch.nn as nn
from medjepa.models.encoder import ViTEncoder
from medjepa.models.predictor import JEPAPredictor
from medjepa.data.masking import PatchMasker2D
from medjepa.training.losses import SIGRegLoss


class LeJEPA(nn.Module):
    """
    LeJEPA for medical image self-supervised learning.

    Training flow:
    1. Take a batch of medical images
    2. Generate masks (which patches to hide)
    3. Encode CONTEXT patches → get context embeddings
    4. Encode TARGET patches → get target embeddings (ground truth)
    5. PREDICT target embeddings from context embeddings
    6. Compute loss: prediction should match target + SIGReg
    7. Update the model to minimize the loss
    """

    def __init__(
        self,
        image_size: int = 224,
        patch_size: int = 16,
        in_channels: int = 3,
        embed_dim: int = 768,
        encoder_depth: int = 12,
        encoder_heads: int = 12,
        predictor_dim: int = 384,
        predictor_depth: int = 6,
        predictor_heads: int = 6,
        mask_ratio: float = 0.75,
        lambda_reg: float = 1.0,
    ):
        super().__init__()

        num_patches = (image_size // patch_size) ** 2

        # The ONE encoder (no momentum encoder, no teacher — just one!)
        self.encoder = ViTEncoder(
            image_size=image_size,
            patch_size=patch_size,
            in_channels=in_channels,
            embed_dim=embed_dim,
            depth=encoder_depth,
            num_heads=encoder_heads,
        )

        # The predictor (smaller than encoder)
        self.predictor = JEPAPredictor(
            embed_dim=embed_dim,
            predictor_dim=predictor_dim,
            depth=predictor_depth,
            num_heads=predictor_heads,
            num_patches=num_patches,
        )

        # Masking strategy
        self.masker = PatchMasker2D(
            image_size=image_size,
            patch_size=patch_size,
            mask_ratio=mask_ratio,
        )

        # Loss function
        self.loss_fn = SIGRegLoss(lambda_reg=lambda_reg)

        # Save config
        self.config = {
            "image_size": image_size,
            "patch_size": patch_size,
            "embed_dim": embed_dim,
            "encoder_depth": encoder_depth,
            "predictor_depth": predictor_depth,
            "mask_ratio": mask_ratio,
            "lambda_reg": lambda_reg,
        }

    def forward(self, images: torch.Tensor) -> dict:
        """
        One forward pass of training.

        Args:
            images: Batch of images, shape (batch_size, 3, 224, 224)

        Returns:
            Dictionary containing losses
        """
        batch_size = images.shape[0]

        # Step 1: Generate mask
        context_indices, target_indices = self.masker.generate_block_mask()
        context_indices = context_indices.to(images.device)
        target_indices = target_indices.to(images.device)

        # Step 2: Encode ALL patches (we need both context and target embeddings)
        all_embeddings = self.encoder(images)

        # Step 3: Extract context and target embeddings
        context_embeddings = all_embeddings[:, context_indices, :]
        target_embeddings = all_embeddings[:, target_indices, :]

        # Step 4: Predict target embeddings from context
        predicted = self.predictor(
            context_embeddings, context_indices, target_indices
        )

        # Step 5: Compute loss
        # IMPORTANT: target_embeddings are detached (no gradient flows through them)
        # This is NOT a heuristic — it's part of the mathematical formulation
        losses = self.loss_fn(
            predicted_target=predicted,
            actual_target=target_embeddings.detach(),
            all_embeddings=all_embeddings,
        )

        return losses

    def encode(self, images: torch.Tensor) -> torch.Tensor:
        """
        Encode images to get their representations.
        Used AFTER training for downstream tasks (classification, etc.)

        Args:
            images: shape (batch_size, 3, 224, 224)
        Returns:
            Embeddings: shape (batch_size, embed_dim)
        """
        with torch.no_grad():
            embeddings = self.encoder(images)
            # Average all patch embeddings → one embedding per image
            return embeddings.mean(dim=1)
```

## Cell 7.6: Test the Model (Forward Pass)

Create notebook: `notebooks/03_test_model.ipynb`

> **Do this on: HP-Lite (CPU, will be slow but works), or HP-GPU/Mac-M4 for speed**

```python
# Cell 1: Setup
import sys
sys.path.append("..")

import torch
from medjepa.models.lejepa import LeJEPA
from medjepa.utils.device import get_device_info, get_device
```

```python
# Cell 2: Check your machine
get_device_info()
device = get_device()
```

```python
# Cell 3: Create a small model for testing
# Using smaller dimensions so it runs fast on CPU
model = LeJEPA(
    image_size=224,
    patch_size=16,
    embed_dim=384,        # Smaller than default 768
    encoder_depth=6,      # Fewer layers than default 12
    encoder_heads=6,
    predictor_dim=192,
    predictor_depth=3,
    predictor_heads=3,
    mask_ratio=0.75,
    lambda_reg=1.0,
).to(device)

# Count parameters
total_params = sum(p.numel() for p in model.parameters())
print(f"Total parameters: {total_params:,}")
print(f"That's {total_params / 1e6:.1f} million parameters")
```

```python
# Cell 4: Test forward pass with fake data
# Create fake batch of 4 images
fake_images = torch.randn(4, 3, 224, 224).to(device)

# Run forward pass
model.train()
losses = model(fake_images)

print(f"Total loss: {losses['total_loss'].item():.4f}")
print(f"Prediction loss: {losses['prediction_loss'].item():.4f}")
print(f"Regularization loss: {losses['regularization_loss'].item():.4f}")
print("\nForward pass successful!")
```

```python
# Cell 5: Test encoding (what you'd use for downstream tasks)
model.eval()
embeddings = model.encode(fake_images)
print(f"Embedding shape: {embeddings.shape}")  # Should be (4, 384)
print(f"Each image is represented by a vector of {embeddings.shape[1]} numbers")
```

```python
# Cell 6: Test that gradients flow (model can learn)
model.train()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

losses = model(fake_images)
losses["total_loss"].backward()  # Compute gradients
optimizer.step()                  # Update weights
optimizer.zero_grad()             # Reset gradients

print("Backward pass successful! Model can learn.")
```

---

# Phase 8: Building the V-JEPA Extension (3D/Video)

> **Do this on: HP-Lite (coding) → test on HP-GPU or Mac-M4**
> Note: 3D models require more memory. HP-Lite testing will use tiny volumes.

## Cell 8.1: Build the 3D/Video Encoder

File: `medjepa/models/vjepa.py`

```python
"""
V-JEPA: Extension for medical video and 3D volumetric data.

This handles:
- CT scans (stack of 2D slices = 3D volume)
- MRI sequences (3D volume changing over time = 4D)
- Surgical videos (2D frames over time = 3D)
- Cardiac MRI (heart beating = temporal sequence)

Key difference from 2D LeJEPA:
- Patches are 3D cubes (or 2D + time)
- Masking happens in space AND time
- The model must understand BOTH spatial anatomy AND temporal changes
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


class PatchEmbedding3D(nn.Module):
    """
    Convert a 3D volume into a sequence of patch embeddings.

    Example:
    A volume of size (1, 128, 128, 64) with patch size (16, 16, 8)
    becomes a grid of 8 x 8 x 8 = 512 patches.
    Each patch is projected to embed_dim dimensions.
    """

    def __init__(
        self,
        volume_size: Tuple[int, int, int] = (128, 128, 64),
        patch_size: Tuple[int, int, int] = (16, 16, 8),
        in_channels: int = 1,  # Medical images are usually grayscale
        embed_dim: int = 768,
    ):
        super().__init__()
        self.patch_size = patch_size
        self.grid_size = tuple(v // p for v, p in zip(volume_size, patch_size))
        self.num_patches = self.grid_size[0] * self.grid_size[1] * self.grid_size[2]

        # 3D convolution for patch extraction
        self.projection = nn.Conv3d(
            in_channels, embed_dim,
            kernel_size=patch_size, stride=patch_size
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Volume, shape (batch, channels, D, H, W)
        Returns:
            Patch embeddings, shape (batch, num_patches, embed_dim)
        """
        x = self.projection(x)                    # (B, E, gD, gH, gW)
        x = x.flatten(2)                           # (B, E, num_patches)
        x = x.transpose(1, 2)                      # (B, num_patches, E)
        return x


class VJEPA(nn.Module):
    """
    V-JEPA model for 3D medical data.

    Same principle as LeJEPA but in 3D:
    - Hide some 3D cubes of the volume
    - Predict their embeddings from the visible cubes
    - Use SIGReg to prevent collapse
    """

    def __init__(
        self,
        volume_size: Tuple[int, int, int] = (128, 128, 64),
        patch_size: Tuple[int, int, int] = (16, 16, 8),
        in_channels: int = 1,
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        predictor_dim: int = 384,
        predictor_depth: int = 6,
        mask_ratio: float = 0.75,
        lambda_reg: float = 1.0,
    ):
        super().__init__()

        grid_size = tuple(v // p for v, p in zip(volume_size, patch_size))
        num_patches = grid_size[0] * grid_size[1] * grid_size[2]

        # 3D patch embedding
        self.patch_embed = PatchEmbedding3D(
            volume_size, patch_size, in_channels, embed_dim
        )

        # 3D positional encoding
        self.pos_embed = nn.Parameter(
            torch.randn(1, num_patches, embed_dim) * 0.02
        )

        # Transformer blocks (same as 2D, attention doesn't care about dimensionality)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        self.encoder_norm = nn.LayerNorm(embed_dim)

        # Predictor (same architecture as 2D)
        predictor_layer = nn.TransformerEncoderLayer(
            d_model=predictor_dim,
            nhead=num_heads // 2,
            dim_feedforward=predictor_dim * 4,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.predictor_input_proj = nn.Linear(embed_dim, predictor_dim)
        self.predictor = nn.TransformerEncoder(
            predictor_layer, num_layers=predictor_depth
        )
        self.predictor_norm = nn.LayerNorm(predictor_dim)
        self.predictor_output_proj = nn.Linear(predictor_dim, embed_dim)

        self.mask_token = nn.Parameter(
            torch.randn(1, 1, predictor_dim) * 0.02
        )
        self.predictor_pos_embed = nn.Parameter(
            torch.randn(1, num_patches, predictor_dim) * 0.02
        )

        self.mask_ratio = mask_ratio
        self.num_patches = num_patches

        # Loss
        from medjepa.training.losses import SIGRegLoss
        self.loss_fn = SIGRegLoss(lambda_reg=lambda_reg)

    def _generate_mask(self):
        """Generate random 3D mask."""
        import numpy as np
        num_masked = int(self.num_patches * self.mask_ratio)
        indices = np.random.permutation(self.num_patches)
        context = torch.tensor(sorted(indices[num_masked:]))
        target = torch.tensor(sorted(indices[:num_masked]))
        return context, target

    def forward(self, volumes: torch.Tensor) -> dict:
        """
        Args:
            volumes: shape (batch, channels, D, H, W)
        """
        # Patch embed
        x = self.patch_embed(volumes) + self.pos_embed

        # Generate mask
        ctx_idx, tgt_idx = self._generate_mask()
        ctx_idx = ctx_idx.to(volumes.device)
        tgt_idx = tgt_idx.to(volumes.device)

        # Encode all patches
        all_embeddings = self.encoder_norm(self.encoder(x))

        # Separate context and target
        context_emb = all_embeddings[:, ctx_idx, :]
        target_emb = all_embeddings[:, tgt_idx, :]

        # Predict targets
        batch_size = volumes.shape[0]
        pred_ctx = self.predictor_input_proj(context_emb)
        pred_ctx = pred_ctx + self.predictor_pos_embed[:, ctx_idx, :]

        mask_tokens = self.mask_token.expand(batch_size, len(tgt_idx), -1)
        mask_tokens = mask_tokens + self.predictor_pos_embed[:, tgt_idx, :]

        pred_input = torch.cat([pred_ctx, mask_tokens], dim=1)
        pred_output = self.predictor_norm(self.predictor(pred_input))
        predicted = self.predictor_output_proj(pred_output[:, -len(tgt_idx):, :])

        # Loss
        losses = self.loss_fn(predicted, target_emb.detach(), all_embeddings)
        return losses

    def encode(self, volumes: torch.Tensor) -> torch.Tensor:
        """Get volume representations for downstream tasks."""
        with torch.no_grad():
            x = self.patch_embed(volumes) + self.pos_embed
            x = self.encoder_norm(self.encoder(x))
            return x.mean(dim=1)
```

---

# Phase 9: Training Infrastructure

> **Do this on: HP-Lite (coding the trainer) → HP-GPU/Mac-M4 (small-scale tests)**

## Cell 9.1: Build the Configuration System

File: `configs/base_config.yaml`

```yaml
# ===================================================
# MedJEPA Base Configuration
# ===================================================
# This file contains ALL settings for an experiment.
# Override specific values in modality-specific configs.

# --- Model Architecture ---
model:
  type: "lejepa" # "lejepa" for 2D, "vjepa" for 3D
  image_size: 224
  patch_size: 16
  in_channels: 3
  embed_dim: 768
  encoder_depth: 12
  encoder_heads: 12
  predictor_dim: 384
  predictor_depth: 6
  predictor_heads: 6

# --- Masking ---
masking:
  mask_ratio: 0.75 # Hide 75% of patches
  mask_type: "block" # "block" or "random"
  num_target_blocks: 4

# --- Training ---
training:
  batch_size: 64 # How many images per step
  num_epochs: 100 # How many times to go through the full dataset
  learning_rate: 0.001
  weight_decay: 0.05 # Prevents overfitting
  warmup_epochs: 10 # Slowly increase learning rate at start
  lambda_reg: 1.0 # SIGReg regularization strength

# --- Data ---
data:
  dataset: "ham10000"
  data_dir: "data/raw/ham10000"
  target_size: 224
  num_workers: 4 # Parallel data loading threads

# --- Hardware ---
hardware:
  device: "auto" # "auto", "cuda", "mps", or "cpu"
  mixed_precision: true # Use float16 to save memory (GPU only)

# --- Logging ---
logging:
  log_every: 50 # Print progress every N steps
  save_every: 5 # Save model every N epochs
  checkpoint_dir: "checkpoints"
  use_wandb: false # Set true to log to wandb.ai
  project_name: "MedJEPA"
```

## Cell 9.2: Build the Training Loop

File: `medjepa/training/trainer.py`

```python
"""
The training loop: where the model actually LEARNS.

This is like a gym workout routine:
1. Show the model a batch of images (exercise)
2. The model makes predictions (attempt)
3. Compute how wrong it was (feedback)
4. Adjust the model weights to be less wrong next time (improve)
5. Repeat thousands of times
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path
import time
import json
from typing import Optional
from medjepa.utils.device import get_device


class MedJEPATrainer:
    """
    Handles the full training process.
    """

    def __init__(
        self,
        model: nn.Module,
        train_dataset,
        config: dict,
        val_dataset=None,
    ):
        """
        Args:
            model: The LeJEPA or VJEPA model
            train_dataset: Training data (PyTorch Dataset)
            config: Configuration dictionary
            val_dataset: Optional validation data
        """
        self.model = model
        self.config = config
        self.device = get_device()
        self.model = self.model.to(self.device)

        # DataLoader: feeds batches of images to the model
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=config.get("batch_size", 32),
            shuffle=True,       # Random order each epoch
            num_workers=config.get("num_workers", 0),
            # Use 0 workers on HP-Lite (Windows), 4 on Linux/Mac
            pin_memory=torch.cuda.is_available(),
            drop_last=True,     # Drop incomplete last batch
        )

        if val_dataset:
            self.val_loader = DataLoader(
                val_dataset,
                batch_size=config.get("batch_size", 32),
                shuffle=False,
                num_workers=config.get("num_workers", 0),
            )
        else:
            self.val_loader = None

        # Optimizer: the algorithm that updates model weights
        # AdamW is the standard choice for Transformers
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.get("learning_rate", 1e-3),
            weight_decay=config.get("weight_decay", 0.05),
            betas=(0.9, 0.999),
        )

        # Learning rate scheduler:
        # Cosine annealing = start with high lr, gradually decrease
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=config.get("num_epochs", 100),
            eta_min=1e-6,
        )

        # Mixed precision training (saves GPU memory)
        self.use_amp = config.get("mixed_precision", False) and torch.cuda.is_available()
        self.scaler = torch.amp.GradScaler("cuda") if self.use_amp else None

        # Tracking
        self.checkpoint_dir = Path(config.get("checkpoint_dir", "checkpoints"))
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.history = {"train_loss": [], "val_loss": [], "epochs": []}

    def train_one_epoch(self, epoch: int) -> float:
        """
        Train for one complete pass through the dataset.

        Returns:
            Average loss for this epoch
        """
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        start_time = time.time()

        for batch_idx, batch in enumerate(self.train_loader):
            # Handle both (images,) and (images, labels) formats
            if isinstance(batch, (list, tuple)):
                images = batch[0]
            else:
                images = batch

            images = images.to(self.device)

            # Forward pass (with optional mixed precision)
            if self.use_amp:
                with torch.amp.autocast("cuda"):
                    losses = self.model(images)
                    loss = losses["total_loss"]

                # Backward pass with scaling
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                losses = self.model(images)
                loss = losses["total_loss"]

                # Backward pass
                loss.backward()
                self.optimizer.step()

            self.optimizer.zero_grad()

            total_loss += loss.item()
            num_batches += 1

            # Print progress
            log_every = self.config.get("log_every", 50)
            if (batch_idx + 1) % log_every == 0:
                elapsed = time.time() - start_time
                avg_loss = total_loss / num_batches
                print(
                    f"  Epoch {epoch+1} | "
                    f"Batch {batch_idx+1}/{len(self.train_loader)} | "
                    f"Loss: {avg_loss:.4f} | "
                    f"Pred Loss: {losses['prediction_loss'].item():.4f} | "
                    f"Reg Loss: {losses['regularization_loss'].item():.4f} | "
                    f"Time: {elapsed:.1f}s"
                )

        # Step the learning rate scheduler
        self.scheduler.step()

        avg_loss = total_loss / max(num_batches, 1)
        return avg_loss

    def train(self) -> dict:
        """
        Full training loop: run for all epochs.
        """
        num_epochs = self.config.get("num_epochs", 100)
        save_every = self.config.get("save_every", 5)
        best_loss = float("inf")

        print("=" * 60)
        print(f"Starting MedJEPA Training")
        print(f"Device: {self.device}")
        print(f"Epochs: {num_epochs}")
        print(f"Batch size: {self.config.get('batch_size', 32)}")
        print(f"Training samples: {len(self.train_loader.dataset)}")
        print("=" * 60)

        for epoch in range(num_epochs):
            epoch_start = time.time()

            # Train
            train_loss = self.train_one_epoch(epoch)
            self.history["train_loss"].append(train_loss)
            self.history["epochs"].append(epoch)

            epoch_time = time.time() - epoch_start
            lr = self.optimizer.param_groups[0]["lr"]

            print(
                f"\nEpoch {epoch+1}/{num_epochs} completed | "
                f"Train Loss: {train_loss:.4f} | "
                f"LR: {lr:.6f} | "
                f"Time: {epoch_time:.1f}s\n"
            )

            # Save checkpoint
            if (epoch + 1) % save_every == 0:
                self.save_checkpoint(epoch, train_loss)

            # Save best model
            if train_loss < best_loss:
                best_loss = train_loss
                self.save_checkpoint(epoch, train_loss, is_best=True)

        print("=" * 60)
        print(f"Training complete! Best loss: {best_loss:.4f}")
        print("=" * 60)

        return self.history

    def save_checkpoint(
        self,
        epoch: int,
        loss: float,
        is_best: bool = False,
    ):
        """Save the model so we can resume later or use it for evaluation."""
        filename = "best_model.pt" if is_best else f"checkpoint_epoch_{epoch+1}.pt"
        path = self.checkpoint_dir / filename

        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "loss": loss,
            "config": self.config,
        }

        torch.save(checkpoint, path)
        print(f"  Saved checkpoint: {path}")

    def load_checkpoint(self, path: str):
        """Load a saved model to resume training or for evaluation."""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if "scheduler_state_dict" in checkpoint:
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        print(f"Loaded checkpoint from epoch {checkpoint['epoch']+1}")
        return checkpoint["epoch"]
```

## Cell 9.3: Create the Pre-Training Launch Script

File: `scripts/pretrain.py`

```python
"""
Main script to launch pre-training.

Usage:
  python scripts/pretrain.py --dataset ham10000 --batch_size 32 --epochs 50

Run this on HP-GPU, Mac-M4, or Cloud.
On HP-Lite, use --batch_size 4 --epochs 2 for testing only.
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
from medjepa.models.lejepa import LeJEPA
from medjepa.data.datasets import MedicalImageDataset
from medjepa.training.trainer import MedJEPATrainer


def parse_args():
    parser = argparse.ArgumentParser(description="MedJEPA Pre-training")

    # Data
    parser.add_argument("--dataset", type=str, default="ham10000")
    parser.add_argument("--data_dir", type=str, default="data/raw/ham10000")
    parser.add_argument("--file_extension", type=str, default=".jpg")

    # Model
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--patch_size", type=int, default=16)
    parser.add_argument("--embed_dim", type=int, default=768)
    parser.add_argument("--encoder_depth", type=int, default=12)
    parser.add_argument("--predictor_depth", type=int, default=6)
    parser.add_argument("--mask_ratio", type=float, default=0.75)

    # Training
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--lambda_reg", type=float, default=1.0)

    # Hardware
    parser.add_argument("--num_workers", type=int, default=0)
    # Use 0 on Windows, 4 on Mac/Linux

    # Logging
    parser.add_argument("--log_every", type=int, default=10)
    parser.add_argument("--save_every", type=int, default=5)
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints")

    return parser.parse_args()


def main():
    args = parse_args()

    print("\n" + "=" * 60)
    print("MedJEPA Pre-Training")
    print("=" * 60)

    # Create dataset
    print(f"\nLoading dataset from: {args.data_dir}")
    dataset = MedicalImageDataset(
        image_dir=args.data_dir,
        file_extension=args.file_extension,
        target_size=(args.image_size, args.image_size),
    )
    print(f"Found {len(dataset)} images")

    # Create model
    model = LeJEPA(
        image_size=args.image_size,
        patch_size=args.patch_size,
        embed_dim=args.embed_dim,
        encoder_depth=args.encoder_depth,
        predictor_depth=args.predictor_depth,
        mask_ratio=args.mask_ratio,
        lambda_reg=args.lambda_reg,
    )

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,} ({total_params/1e6:.1f}M)")

    # Training config
    config = {
        "batch_size": args.batch_size,
        "num_epochs": args.epochs,
        "learning_rate": args.lr,
        "weight_decay": 0.05,
        "num_workers": args.num_workers,
        "log_every": args.log_every,
        "save_every": args.save_every,
        "checkpoint_dir": args.checkpoint_dir,
        "mixed_precision": torch.cuda.is_available(),
    }

    # Create trainer and start!
    trainer = MedJEPATrainer(
        model=model,
        train_dataset=dataset,
        config=config,
    )

    history = trainer.train()

    # Save training history
    import json
    with open(Path(args.checkpoint_dir) / "training_history.json", "w") as f:
        json.dump(history, f, indent=2)

    print("\nDone! Model saved to:", args.checkpoint_dir)


if __name__ == "__main__":
    main()
```

## Cell 9.4: Machine-Specific Training Commands

### On HP-Lite (TESTING ONLY — tiny run to verify code works):

```bash
# Activate your venv first!
cd F:\Projects\MedJEPA
venv\Scripts\activate

python scripts/pretrain.py \
    --data_dir data/raw/ham10000 \
    --batch_size 4 \
    --epochs 2 \
    --embed_dim 384 \
    --encoder_depth 4 \
    --predictor_depth 2 \
    --num_workers 0 \
    --log_every 5
```

### On HP-GPU (Real small-scale experiments):

```bash
python scripts/pretrain.py \
    --data_dir data/raw/ham10000 \
    --batch_size 32 \
    --epochs 50 \
    --embed_dim 768 \
    --encoder_depth 12 \
    --num_workers 4 \
    --log_every 50
```

### On Mac-M4 (Medium experiments):

```bash
cd ~/Projects/MedJEPA
source venv/bin/activate

python scripts/pretrain.py \
    --data_dir data/raw/ham10000 \
    --batch_size 32 \
    --epochs 50 \
    --embed_dim 768 \
    --encoder_depth 12 \
    --num_workers 4 \
    --log_every 50
```

### On Google Colab (Full-scale training):

```python
# In a Colab notebook cell:
!git clone https://github.com/YOUR-USERNAME/MedJEPA.git
%cd MedJEPA
!pip install -q pydicom nibabel SimpleITK monai albumentations wandb

# Upload or mount your dataset (Google Drive recommended)
from google.colab import drive
drive.mount('/content/drive')

# Run training
!python scripts/pretrain.py \
    --data_dir /content/drive/MyDrive/datasets/ham10000 \
    --batch_size 128 \
    --epochs 100 \
    --embed_dim 768 \
    --encoder_depth 12 \
    --num_workers 4 \
    --log_every 100
```

---

# Phase 10: Pre-Training the Model

> **Do this on: HP-GPU or Mac-M4 (small datasets), Cloud (large datasets)**

## Cell 10.1: Pre-Training Strategy

```
Start small, then scale up:

Week 1-2: HAM10000 (10K images, ~3GB)
  → On: HP-GPU or Mac-M4
  → Goal: Verify the pipeline works end-to-end
  → Expected time: 2-4 hours for 50 epochs

Week 3-4: PCam (327K patches, ~8GB)
  → On: HP-GPU or Mac-M4
  → Goal: Test with larger dataset
  → Expected time: 1-2 days for 50 epochs

Week 5-8: ChestX-ray14 (112K images, ~45GB)
  → On: Cloud GPU (Colab or lab cluster)
  → Goal: Real pre-training on a standard benchmark
  → Expected time: 2-5 days for 100 epochs

Week 9+: Multi-dataset pre-training
  → On: Cloud GPU cluster (multi-GPU)
  → Goal: Train on ALL collected datasets
  → Expected time: 1-2 weeks
```

## Cell 10.2: Monitor Training

During training, watch these numbers:

```
1. Total Loss: Should DECREASE over epochs
   - If it goes UP: Learning rate might be too high
   - If it's STUCK: Model might be too small, or data issue

2. Prediction Loss: How well the model predicts hidden patches
   - Should decrease steadily

3. Regularization Loss: How "collapsed" the embeddings are
   - Should stay LOW (close to 0)
   - If it's very HIGH: embeddings are collapsing, increase lambda_reg

4. Learning Rate: Should follow a cosine curve
   - Starts high, gradually decreases to near zero
```

---

# Phase 11: Evaluation — Disease Classification

> **Do this on: HP-GPU or Mac-M4 (small evaluations), Cloud (full benchmarks)**

## Cell 11.1: What is Evaluation?

Pre-training teaches the model to UNDERSTAND medical images.
Evaluation tests HOW WELL it understands them by giving it specific tasks.

Three evaluation methods:

```
1. LINEAR PROBING (simplest):
   - Freeze the pre-trained model (don't change it)
   - Add ONE new layer on top (a simple classifier)
   - Train ONLY that new layer
   - If it works well → the pre-trained representations are good!

2. FEW-SHOT LEARNING:
   - Give the model only 5 or 10 labeled examples
   - See if it can classify new images correctly
   - This tests efficiency: can it learn from very few examples?

3. FULL FINE-TUNING:
   - Unfreeze the entire model
   - Train everything on the labeled dataset
   - This gives the best performance but takes longer
```

## Cell 11.2: Build the Linear Probing Evaluator

File: `medjepa/evaluation/linear_probe.py`

```python
"""
Linear Probing: The standard way to evaluate self-supervised models.

After pre-training, we:
1. Freeze the encoder (no more changes to it)
2. Add a simple linear classification layer on top
3. Train ONLY the classification layer on a small labeled dataset
4. Measure accuracy

Good results here = the encoder learned useful representations.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from typing import Optional
import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report
from medjepa.utils.device import get_device


class LinearProbe(nn.Module):
    """A single linear layer for classification."""

    def __init__(self, embed_dim: int, num_classes: int):
        super().__init__()
        self.linear = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        return self.linear(x)


class LinearProbeEvaluator:
    """
    Evaluates a pre-trained model using linear probing.
    """

    def __init__(
        self,
        pretrained_model,
        num_classes: int,
        embed_dim: int = 768,
    ):
        self.device = get_device()
        self.pretrained_model = pretrained_model.to(self.device)
        self.pretrained_model.eval()

        # Freeze the model — we don't want to change it
        for param in self.pretrained_model.parameters():
            param.requires_grad = False

        # Create the linear probe
        self.probe = LinearProbe(embed_dim, num_classes).to(self.device)

    def extract_features(self, dataloader: DataLoader) -> tuple:
        """
        Run all images through the frozen encoder to get embeddings.
        This only needs to be done ONCE.
        """
        all_features = []
        all_labels = []

        with torch.no_grad():
            for batch in dataloader:
                images, labels = batch
                images = images.to(self.device)

                # Get embeddings from the frozen encoder
                features = self.pretrained_model.encode(images)

                all_features.append(features.cpu())
                all_labels.append(labels)

        features = torch.cat(all_features, dim=0)
        labels = torch.cat(all_labels, dim=0)
        return features, labels

    def train_probe(
        self,
        train_features: torch.Tensor,
        train_labels: torch.Tensor,
        num_epochs: int = 100,
        lr: float = 0.01,
        batch_size: int = 256,
    ):
        """Train the linear classification layer."""
        from torch.utils.data import TensorDataset, DataLoader

        dataset = TensorDataset(train_features, train_labels)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        optimizer = torch.optim.SGD(self.probe.parameters(), lr=lr, momentum=0.9)
        criterion = nn.CrossEntropyLoss()

        self.probe.train()
        for epoch in range(num_epochs):
            total_loss = 0
            for features, labels in loader:
                features = features.to(self.device)
                labels = labels.to(self.device)

                logits = self.probe(features)
                loss = criterion(logits, labels)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

            if (epoch + 1) % 20 == 0:
                avg_loss = total_loss / len(loader)
                print(f"  Probe Epoch {epoch+1}/{num_epochs} | Loss: {avg_loss:.4f}")

    def evaluate(
        self,
        test_features: torch.Tensor,
        test_labels: torch.Tensor,
    ) -> dict:
        """Evaluate the linear probe on test data."""
        self.probe.eval()
        with torch.no_grad():
            test_features = test_features.to(self.device)
            logits = self.probe(test_features)
            predictions = logits.argmax(dim=1).cpu().numpy()
            probabilities = torch.softmax(logits, dim=1).cpu().numpy()

        true_labels = test_labels.numpy()

        results = {
            "accuracy": accuracy_score(true_labels, predictions),
            "report": classification_report(
                true_labels, predictions, output_dict=True
            ),
        }

        # AUC if binary or we can compute it
        try:
            if probabilities.shape[1] == 2:
                results["auc"] = roc_auc_score(true_labels, probabilities[:, 1])
            else:
                results["auc"] = roc_auc_score(
                    true_labels, probabilities, multi_class="ovr", average="macro"
                )
        except Exception:
            results["auc"] = None

        return results
```

## Cell 11.3: Build the Few-Shot Evaluator

File: `medjepa/evaluation/few_shot.py`

```python
"""
Few-Shot Learning Evaluation.

Tests: Can the model learn from very few examples?

Example: Given only 5 examples of "pneumonia" and 5 examples of "normal",
can the model correctly classify new chest X-rays?

This is the most clinically relevant evaluation because in real hospitals,
getting 5 labeled examples is feasible, but 50,000 is not.
"""

import torch
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from typing import Tuple
from medjepa.utils.device import get_device


class FewShotEvaluator:
    """
    Evaluate representations using few-shot classification.

    Method: k-Nearest Neighbors (kNN)
    - Encode all images to embeddings
    - For each test image, find the k closest training embeddings
    - Classify based on majority vote of those k neighbors
    """

    def __init__(self, pretrained_model, k: int = 5):
        self.device = get_device()
        self.model = pretrained_model.to(self.device)
        self.model.eval()
        self.k = k

    def evaluate_n_shot(
        self,
        support_images: torch.Tensor,  # The few labeled examples
        support_labels: torch.Tensor,
        query_images: torch.Tensor,     # Test images
        query_labels: torch.Tensor,
        n_shot: int = 5,                # How many examples per class
    ) -> dict:
        """
        Evaluate n-shot classification.

        Args:
            support_images: Few labeled examples, shape (n_classes * n_shot, C, H, W)
            support_labels: Their labels
            query_images: Test images to classify
            query_labels: True labels for test images
            n_shot: Number of examples per class
        """
        with torch.no_grad():
            # Encode support and query images
            support_features = self.model.encode(
                support_images.to(self.device)
            ).cpu().numpy()
            query_features = self.model.encode(
                query_images.to(self.device)
            ).cpu().numpy()

        # Use kNN classifier
        knn = KNeighborsClassifier(n_neighbors=min(self.k, len(support_labels)))
        knn.fit(support_features, support_labels.numpy())

        predictions = knn.predict(query_features)
        accuracy = accuracy_score(query_labels.numpy(), predictions)

        return {
            "n_shot": n_shot,
            "accuracy": accuracy,
            "num_support": len(support_labels),
            "num_query": len(query_labels),
        }

    def evaluate_data_efficiency(
        self,
        full_train_features: torch.Tensor,
        full_train_labels: torch.Tensor,
        test_features: torch.Tensor,
        test_labels: torch.Tensor,
        fractions: list = [0.01, 0.05, 0.1, 0.25, 0.5, 1.0],
    ) -> list:
        """
        Test with different amounts of labeled data.
        Answers: "How much does performance improve as we add more labels?"

        Args:
            fractions: What percentage of labels to use (0.01 = 1%, 0.5 = 50%)
        """
        results = []

        for frac in fractions:
            n = max(1, int(len(full_train_labels) * frac))
            indices = np.random.choice(len(full_train_labels), n, replace=False)

            subset_features = full_train_features[indices].numpy()
            subset_labels = full_train_labels[indices].numpy()

            knn = KNeighborsClassifier(
                n_neighbors=min(self.k, len(subset_labels))
            )
            knn.fit(subset_features, subset_labels)
            predictions = knn.predict(test_features.numpy())
            accuracy = accuracy_score(test_labels.numpy(), predictions)

            results.append({
                "fraction": frac,
                "num_labeled": n,
                "accuracy": accuracy,
            })
            print(f"  {frac*100:.0f}% data ({n} samples): Accuracy = {accuracy:.4f}")

        return results
```

---

# Phase 12: Evaluation — Detection & Segmentation

> **Do this on: Cloud GPU (segmentation is memory-intensive)**

## Cell 12.1: Segmentation Overview

Segmentation = drawing a precise outline around objects in images.

For medical imaging:

- Outline the tumor in a brain MRI
- Outline the lungs in a chest X-ray
- Outline polyps in a colonoscopy image

We'll test if MedJEPA's representations help with segmentation by using the frozen encoder as a "backbone" and adding a segmentation head on top.

File: `medjepa/evaluation/segmentation.py`

```python
"""
Segmentation evaluation for MedJEPA.

Uses the pre-trained encoder as a feature extractor,
then adds a simple segmentation decoder to produce pixel-level predictions.

Metric: Dice Score
- 1.0 = perfect overlap between predicted and ground truth masks
- 0.0 = no overlap at all
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from medjepa.utils.device import get_device


class SimpleSegmentationHead(nn.Module):
    """
    A simple decoder that converts patch embeddings into a segmentation mask.

    Takes encoder output (patch embeddings) and upsamples them to pixel-level
    predictions (same size as the original image).
    """

    def __init__(
        self,
        embed_dim: int = 768,
        num_classes: int = 2,    # Usually 2: background + foreground (lesion)
        image_size: int = 224,
        patch_size: int = 16,
    ):
        super().__init__()
        self.grid_size = image_size // patch_size
        self.image_size = image_size

        # Decode: embed_dim → image pixels
        self.decoder = nn.Sequential(
            nn.Linear(embed_dim, 256),
            nn.ReLU(),
            nn.Linear(256, patch_size * patch_size * num_classes),
        )
        self.num_classes = num_classes
        self.patch_size = patch_size

    def forward(self, patch_embeddings: torch.Tensor) -> torch.Tensor:
        """
        Args:
            patch_embeddings: (batch, num_patches, embed_dim)
        Returns:
            Segmentation mask: (batch, num_classes, H, W)
        """
        batch_size = patch_embeddings.shape[0]

        # Decode each patch
        decoded = self.decoder(patch_embeddings)
        # Shape: (batch, num_patches, patch_size * patch_size * num_classes)

        # Reshape to spatial grid
        decoded = decoded.reshape(
            batch_size, self.grid_size, self.grid_size,
            self.patch_size, self.patch_size, self.num_classes
        )

        # Rearrange to full image
        decoded = decoded.permute(0, 5, 1, 3, 2, 4)
        decoded = decoded.reshape(
            batch_size, self.num_classes, self.image_size, self.image_size
        )

        return decoded


def dice_score(pred: torch.Tensor, target: torch.Tensor, smooth: float = 1e-5) -> float:
    """
    Compute Dice Score — the standard segmentation metric.

    Dice = 2 * |A ∩ B| / (|A| + |B|)

    Think of it like: how much do the predicted and actual outlines overlap?
    """
    pred = (pred > 0.5).float()
    intersection = (pred * target).sum()
    union = pred.sum() + target.sum()
    dice = (2.0 * intersection + smooth) / (union + smooth)
    return dice.item()
```

---

# Phase 13: Evaluation — Clinical Predictions & Few-Shot

> **Do this on: Cloud GPU for large evaluations, HP-GPU/Mac-M4 for small tests**

## Cell 13.1: The Complete Evaluation Script

File: `scripts/evaluate.py`

```python
"""
Run all evaluations on a pre-trained MedJEPA model.

Usage:
  python scripts/evaluate.py --checkpoint checkpoints/best_model.pt --dataset ham10000

This will run:
1. Linear probing (accuracy on classification tasks)
2. Few-shot learning (accuracy with 5, 10, 20 examples)
3. Data efficiency analysis (1%, 5%, 10%, 25%, 50%, 100% labeled data)
"""

import argparse
import sys
import json
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
from torch.utils.data import DataLoader, random_split
from medjepa.models.lejepa import LeJEPA
from medjepa.data.datasets import MedicalImageDataset
from medjepa.evaluation.linear_probe import LinearProbeEvaluator
from medjepa.evaluation.few_shot import FewShotEvaluator
from medjepa.utils.device import get_device


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--metadata_csv", type=str, default=None)
    parser.add_argument("--label_column", type=str, default="dx")
    parser.add_argument("--num_classes", type=int, default=7)  # HAM10000 has 7 classes
    parser.add_argument("--batch_size", type=int, default=64)
    args = parser.parse_args()

    device = get_device()

    # Load model
    print("Loading pre-trained model...")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    config = checkpoint.get("config", {})

    model = LeJEPA(
        embed_dim=config.get("embed_dim", 768),
        encoder_depth=config.get("encoder_depth", 12),
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    print("Model loaded!")

    # Load labeled dataset
    print("\nLoading dataset...")
    dataset = MedicalImageDataset(
        image_dir=args.data_dir,
        metadata_csv=args.metadata_csv,
        label_column=args.label_column,
    )

    # Split into train (80%) and test (20%)
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, num_workers=0)

    # ==========================================
    # Evaluation 1: Linear Probing
    # ==========================================
    print("\n" + "=" * 50)
    print("EVALUATION 1: Linear Probing")
    print("=" * 50)

    evaluator = LinearProbeEvaluator(
        pretrained_model=model,
        num_classes=args.num_classes,
        embed_dim=config.get("embed_dim", 768),
    )

    print("Extracting features...")
    train_features, train_labels = evaluator.extract_features(train_loader)
    test_features, test_labels = evaluator.extract_features(test_loader)
    print(f"Train: {train_features.shape}, Test: {test_features.shape}")

    print("Training linear probe...")
    evaluator.train_probe(train_features, train_labels)

    print("Evaluating...")
    results_lp = evaluator.evaluate(test_features, test_labels)
    print(f"\nLinear Probe Accuracy: {results_lp['accuracy']:.4f}")
    if results_lp.get("auc"):
        print(f"Linear Probe AUC: {results_lp['auc']:.4f}")

    # ==========================================
    # Evaluation 2: Few-Shot Learning
    # ==========================================
    print("\n" + "=" * 50)
    print("EVALUATION 2: Few-Shot Learning")
    print("=" * 50)

    few_shot = FewShotEvaluator(pretrained_model=model)
    results_fs = few_shot.evaluate_data_efficiency(
        train_features, train_labels,
        test_features, test_labels,
        fractions=[0.01, 0.05, 0.1, 0.25, 0.5, 1.0],
    )

    # ==========================================
    # Save Results
    # ==========================================
    results = {
        "linear_probing": {
            "accuracy": results_lp["accuracy"],
            "auc": results_lp.get("auc"),
        },
        "few_shot": results_fs,
    }

    output_path = Path("results") / "evaluation_results.json"
    output_path.parent.mkdir(exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
```

---

# Phase 14: Interpretability & Visualizations

> **Do this on: Any machine (generates images, not compute-heavy)**

## Cell 14.1: Build Visualization Tools

File: `medjepa/utils/visualization.py`

```python
"""
Visualization tools for understanding what MedJEPA learned.

These visuals help answer:
- What parts of the image does the model pay attention to?
- Do similar diseases have similar embeddings?
- Can we see what features the model learned?
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from typing import Optional, List


def plot_training_history(history: dict, save_path: Optional[str] = None):
    """Plot loss curves from training."""
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))

    ax.plot(history["epochs"], history["train_loss"], label="Training Loss")
    if "val_loss" in history and history["val_loss"]:
        ax.plot(history["epochs"], history["val_loss"], label="Validation Loss")

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("MedJEPA Training Progress")
    ax.legend()
    ax.grid(True, alpha=0.3)

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


def plot_embedding_space(
    embeddings: np.ndarray,
    labels: np.ndarray,
    class_names: Optional[List[str]] = None,
    title: str = "Embedding Space (t-SNE)",
    save_path: Optional[str] = None,
):
    """
    Visualize the learned embedding space using t-SNE.

    t-SNE takes high-dimensional embeddings (768 numbers per image)
    and projects them to 2D so we can see them.

    What to look for:
    - Images of the same disease should cluster together
    - Different diseases should be separated
    """
    # Reduce to 2D
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    embeddings_2d = tsne.fit_transform(embeddings)

    # Plot
    fig, ax = plt.subplots(1, 1, figsize=(12, 10))

    unique_labels = sorted(set(labels))
    colors = plt.cm.tab10(np.linspace(0, 1, len(unique_labels)))

    for i, label in enumerate(unique_labels):
        mask = labels == label
        name = class_names[i] if class_names else f"Class {label}"
        ax.scatter(
            embeddings_2d[mask, 0],
            embeddings_2d[mask, 1],
            c=[colors[i]],
            label=name,
            alpha=0.6,
            s=10,
        )

    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    ax.set_title(title)
    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


def plot_attention_map(
    image: np.ndarray,
    attention_weights: np.ndarray,
    title: str = "Attention Map",
    save_path: Optional[str] = None,
):
    """
    Overlay attention weights on a medical image.

    Shows WHERE the model is looking when analyzing the image.
    Bright areas = model pays more attention there.

    Clinicians can use this to verify the model is looking at
    the right areas (e.g., the tumor, not the text annotation).
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Original image
    axes[0].imshow(image)
    axes[0].set_title("Original Image")
    axes[0].axis("off")

    # Attention map
    axes[1].imshow(attention_weights, cmap="hot")
    axes[1].set_title("Attention Map")
    axes[1].axis("off")

    # Overlay
    axes[2].imshow(image)
    axes[2].imshow(attention_weights, cmap="hot", alpha=0.5)
    axes[2].set_title("Overlay")
    axes[2].axis("off")

    plt.suptitle(title, fontsize=14)

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


def plot_data_efficiency(
    results: list,
    baseline_accuracy: float = None,
    title: str = "Data Efficiency Curve",
    save_path: Optional[str] = None,
):
    """
    Plot how accuracy changes with amount of labeled data.

    This is the MONEY PLOT for self-supervised learning:
    Shows that MedJEPA needs far fewer labels than supervised methods.
    """
    fractions = [r["fraction"] * 100 for r in results]
    accuracies = [r["accuracy"] * 100 for r in results]

    fig, ax = plt.subplots(1, 1, figsize=(10, 6))

    ax.plot(fractions, accuracies, "bo-", linewidth=2, markersize=8, label="MedJEPA")

    if baseline_accuracy:
        ax.axhline(
            y=baseline_accuracy * 100, color="r", linestyle="--",
            label=f"Supervised baseline ({baseline_accuracy*100:.1f}%)"
        )

    ax.set_xlabel("% of Labeled Training Data")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
```

---

# Phase 15: Final Packaging & Open-Source Release

> **Do this on: HP-Lite (writing docs and packaging)**

## Cell 15.1: Create setup.py

File: `setup.py`

```python
from setuptools import setup, find_packages

setup(
    name="medjepa",
    version="0.1.0",
    description="Self-Supervised Medical Image Representation Learning with JEPA",
    author="Your Name",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[
        "torch>=2.0",
        "torchvision>=0.15",
        "numpy",
        "pandas",
        "matplotlib",
        "scikit-learn",
        "pydicom",
        "nibabel",
        "monai",
        "albumentations",
        "opencv-python",
        "Pillow",
        "tqdm",
        "pyyaml",
    ],
)
```

## Cell 15.2: Write the README

File: `README.md` (your project's front page)

```markdown
# MedJEPA: Self-Supervised Medical Image Representation Learning with JEPA

Learn powerful medical image representations WITHOUT labeled data.

## What is MedJEPA?

MedJEPA applies Joint-Embedding Predictive Architecture (JEPA) to medical imaging,
enabling AI models to learn from the vast amounts of unlabeled medical images in
hospital archives.

## Quick Start

​`bash
pip install -e .
python scripts/pretrain.py --data_dir data/raw/ham10000 --epochs 50
python scripts/evaluate.py --checkpoint checkpoints/best_model.pt
​`

## Supported Modalities

- 2D: Chest X-rays, histopathology, retinal images, dermatology
- 3D: CT scans, MRI volumes
- Video: Cardiac MRI sequences, surgical videos

## Results

| Dataset                     | Linear Probe | 10-shot | Full Fine-Tune |
| --------------------------- | ------------ | ------- | -------------- |
| HAM10000                    | XX.X%        | XX.X%   | XX.X%          |
| ChestX-ray14                | XX.X%        | XX.X%   | XX.X%          |
| (fill in after experiments) |

## Citation

If you use this work, please cite:
​`bibtex
@misc{medjepa2026,
  title={MedJEPA: Self-Supervised Medical Image Representation Learning with JEPA},
  year={2026},
}
​`
```

## Cell 15.3: Final Checklist

```
Before submitting the project, make sure you have:

□ Code
  □ All code runs without errors
  □ Code is well-commented
  □ Tests pass (python -m pytest tests/)
  □ Code follows consistent style

□ Models
  □ Pre-trained checkpoints uploaded (HuggingFace Hub or GitHub Releases)
  □ Model configs documented
  □ Instructions to load and use pre-trained models

□ Data
  □ Data download scripts work
  □ Preprocessing pipeline documented
  □ No private patient data in the repository!

□ Results
  □ Benchmarks on 10+ datasets
  □ Few-shot evaluation results
  □ Cross-institutional validation
  □ Comparison tables against baselines
  □ Interpretability visualizations

□ Documentation
  □ README with quick start guide
  □ Tutorial notebooks
  □ API documentation
  □ Installation instructions for all platforms

□ Repository
  □ Clean git history
  □ Proper .gitignore
  □ LICENSE file
  □ requirements.txt
  □ All notebooks have clean outputs
```

---

# Appendix A: Complete Timeline

```
Weeks 1-2:   Setup + Read Papers                    [HP-Lite]
Weeks 3-4:   Build Data Pipeline + Download Data     [HP-Lite]
Weeks 5-6:   Build Masking + Encoder + Predictor     [HP-Lite → test on HP-GPU]
Weeks 7-8:   Build LeJEPA Model + Training Loop      [HP-Lite → HP-GPU]
Week 9:      Small-scale Pre-training (HAM10000)     [HP-GPU / Mac-M4]
Week 10:     Build Evaluation Scripts                 [HP-Lite → HP-GPU]
Weeks 11-12: Pre-train on ChestX-ray14               [Cloud GPU]
Week 13:     Build V-JEPA Extension                   [HP-Lite → Cloud]
Weeks 14-15: Full Evaluation Suite                    [Cloud GPU]
Week 16:     Interpretability + Viz                   [Any machine]
Weeks 17-18: More Datasets + Cross-validation         [Cloud GPU]
Weeks 19-20: Documentation + Polish + Release         [HP-Lite]
```

# Appendix B: Troubleshooting Common Issues

```
PROBLEM: "CUDA out of memory"
SOLUTION: Reduce batch_size, or use smaller embed_dim, or use gradient accumulation

PROBLEM: "MPS backend not available" on Mac
SOLUTION: Update PyTorch (pip install --upgrade torch)

PROBLEM: Loss is NaN (not a number)
SOLUTION: Reduce learning rate, check for division by zero in preprocessing

PROBLEM: Loss is not decreasing
SOLUTION: Check data loading (are images loaded correctly?), try different learning rate

PROBLEM: "No module named medjepa"
SOLUTION: Run from project root, or install with pip install -e .

PROBLEM: DataLoader workers crash on Windows
SOLUTION: Set num_workers=0 on Windows (a known limitation)

PROBLEM: Import errors for pydicom/nibabel
SOLUTION: Make sure you activated the virtual environment (venv\Scripts\activate)

PROBLEM: Git push rejected
SOLUTION: git pull origin main first, resolve conflicts, then push
```

# Appendix C: Useful Commands Cheat Sheet

```bash
# --- Virtual Environment ---
# Windows:
venv\Scripts\activate
# Mac/Linux:
source venv/bin/activate

# --- Git ---
git add .
git commit -m "Your message"
git push origin main
git pull origin main
git status

# --- Running Scripts ---
python scripts/pretrain.py --help          # See all options
python -m pytest tests/                     # Run tests
python -c "import torch; print(torch.cuda.is_available())"  # Check GPU

# --- Jupyter ---
jupyter notebook                            # Open notebooks in browser
jupyter lab                                 # Better notebook interface

# --- Monitor GPU ---
# NVIDIA:
nvidia-smi                                  # GPU usage and memory
watch -n 1 nvidia-smi                       # Auto-refresh every sec (Linux/Mac)

# --- Check Disk Space ---
# Windows:
dir data /s                                 # Size of data folder
# Mac/Linux:
du -sh data/                                # Size of data folder
```

---

**You've got this! Start with Phase 0 (understanding) and Phase 1 (setup) on your HP-Lite, and work through each phase systematically. Don't skip the paper reading — understanding WHY things work is more important than making them work.**
