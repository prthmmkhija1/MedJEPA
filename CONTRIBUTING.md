# Contributing to MedJEPA

Thank you for your interest in contributing to MedJEPA! This project is part of the
[UCSC OSPO 2026](https://ucsc-ospo.github.io/project/osre26/nelbl/medjepa/) Open Source
Research Experience under the NeuroHealth / NELBL Lab.

## Getting Started

1. **Fork** the repository and clone your fork
2. Create a virtual environment and install in development mode:
   ```bash
   python -m venv venv
   source venv/bin/activate   # or venv\Scripts\activate on Windows
   pip install -e ".[dev]"
   ```
3. Run tests to make sure everything works:
   ```bash
   python -m pytest tests/ -v
   ```

## Development Workflow

1. **Create a branch** for your change: `git checkout -b feature/my-change`
2. **Make changes** — keep commits focused and well-described
3. **Run tests**: `python -m pytest tests/ -v`
4. **Format code**: `black medjepa/ scripts/ tests/`
5. **Lint**: `flake8 medjepa/ scripts/ --max-line-length 120`
6. **Open a Pull Request** with a clear description of what and why

## Code Style

- Follow PEP 8 (enforced by `black` and `flake8`)
- Max line length: 120 characters
- Add docstrings to public functions
- Type hints for function signatures are encouraged
- Keep comments focused on _why_, not _what_

## What to Contribute

### High-Impact Areas

- **New datasets**: Add support for additional medical imaging datasets
  (MIMIC-CXR, CheXpert, EyePACS, Camelyon16/17, ISIC)
- **Domain-invariant training**: Improve cross-institutional generalization
- **Segmentation improvements**: Better decoder architectures (UNet-style head)
- **3D V-JEPA enhancements**: Better volumetric masking strategies
- **Benchmarks**: Run on more datasets and compare with other SSL methods

### Good First Issues

- Improve documentation or fix typos
- Add more unit tests for edge cases
- Add type hints to untyped functions
- Improve error messages for common setup issues

## Testing

All contributions should include tests where applicable:

```bash
# Run all tests
python -m pytest tests/ -v

# Run a specific test
python -m pytest tests/test_core.py::TestLeJEPA -v

# Quick smoke test (CPU, tiny model)
python scripts/pretrain.py \
    --data_dir data/raw/ham10000 \
    --batch_size 4 --epochs 2 \
    --embed_dim 192 --encoder_depth 2 --predictor_depth 1
```

## Medical Data Privacy

When working with medical data:

- **NEVER** commit patient data or DICOM files with identifiable information
- Always use `medjepa.data.dicom_utils.anonymize_dicom()` before processing
- Keep all raw data in `data/raw/` which is `.gitignore`-d
- Follow HIPAA/GDPR guidelines for any clinical data

## Reporting Issues

When reporting a bug, include:

- Python and PyTorch versions (`python --version`, `python -c "import torch; print(torch.__version__)"`)
- GPU info if relevant (`nvidia-smi`)
- Full error traceback
- Minimal code to reproduce the issue

## License

By contributing, you agree that your contributions will be licensed under the
[MIT License](LICENSE).
