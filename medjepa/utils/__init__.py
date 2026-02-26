"""Utility functions: device detection, logging, visualization."""

from medjepa.utils.device import get_device, get_device_info
from medjepa.utils.visualization import (
    plot_training_history,
    plot_embedding_space,
    plot_attention_map,
    plot_data_efficiency,
    extract_attention_weights,
    plot_reconstruction_comparison,
    plot_evaluation_summary,
)

__all__ = [
    "get_device",
    "get_device_info",
    "plot_training_history",
    "plot_embedding_space",
    "plot_attention_map",
    "plot_data_efficiency",
    "extract_attention_weights",
    "plot_reconstruction_comparison",
    "plot_evaluation_summary",
]
