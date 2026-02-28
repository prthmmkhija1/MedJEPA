"""Evaluation: linear probing, few-shot, segmentation, metrics."""

from medjepa.evaluation.linear_probe import LinearProbe, LinearProbeEvaluator
from medjepa.evaluation.few_shot import FewShotEvaluator
from medjepa.evaluation.segmentation import (
    SimpleSegmentationHead,
    SegmentationEvaluator,
    dice_score,
)

__all__ = [
    "LinearProbe",
    "LinearProbeEvaluator",
    "FewShotEvaluator",
    "SimpleSegmentationHead",
    "SegmentationEvaluator",
    "dice_score",
]
