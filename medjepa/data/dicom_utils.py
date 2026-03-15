"""
Utilities for working with DICOM medical images.
Includes anonymization (removing patient information for privacy).
"""

from pathlib import Path
import numpy as np
from typing import List

try:
    import pydicom
except ImportError:
    pydicom = None


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


def _require_pydicom():
    if pydicom is None:
        raise ImportError(
            "pydicom is required for DICOM operations. "
            "Install it with: pip install pydicom"
        )


def anonymize_dicom(input_path: str, output_path: str):
    """
    Remove all personal/identifying information from a DICOM file.

    IMPORTANT: This MUST be done before using any hospital DICOM data.
    Laws like HIPAA (USA) and GDPR (Europe) require this.

    Args:
        input_path: Path to original DICOM file
        output_path: Path where the anonymized file will be saved
    """
    _require_pydicom()
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
    _require_pydicom()
    ds = pydicom.dcmread(dicom_path)
    return ds.pixel_array.astype(np.float32)


def get_dicom_info(dicom_path: str) -> dict:
    """
    Get useful (non-personal) information from a DICOM file.
    Things like image size, scan type, etc.
    """
    _require_pydicom()
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
