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
        mem = torch.cuda.get_device_properties(0).total_memory
        print(f"  Memory: {mem / 1e9:.1f} GB")
    mps_available = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    print(f"MPS available: {mps_available}")
    print(f"Selected device: {get_device()}")
    print("=" * 50)


if __name__ == "__main__":
    get_device_info()
