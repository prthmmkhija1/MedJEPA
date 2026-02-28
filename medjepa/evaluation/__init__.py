"""Evaluation: linear probing, few-shot, segmentation, fine-tuning, baselines."""

from medjepa.evaluation.linear_probe import LinearProbe, LinearProbeEvaluator
from medjepa.evaluation.few_shot import FewShotEvaluator
from medjepa.evaluation.segmentation import (
    SimpleSegmentationHead,
    SegmentationEvaluator,
    dice_score,
)
from medjepa.evaluation.fine_tune import (
    FineTuneEvaluator,
    ImageNetBaselineEvaluator,
)

__all__ = [
    "LinearProbe",
    "LinearProbeEvaluator",
    "FewShotEvaluator",
    "SimpleSegmentationHead",
    "SegmentationEvaluator",
    "dice_score",
    "FineTuneEvaluator",
    "ImageNetBaselineEvaluator",
]
