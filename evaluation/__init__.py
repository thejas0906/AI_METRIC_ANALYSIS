# evaluation/__init__.py
"""
evaluation package
===================
Evaluation infrastructure for the Hallucination Correction Framework.

Exports:
- EvaluationMetrics    → All 8 research metrics (HDA, P, R, F1, CSR, CPR, UMR, FRA)
- MetricsResult        → Data class: all computed metric values
- CorrectionOutcome    → Data class: per-claim correction data for metrics
- DatasetLoader        → Load FEVER / TruthfulQA / custom datasets
- DatasetSample        → Data class: single evaluation sample
"""

from evaluation.metrics import (
    EvaluationMetrics,
    MetricsResult,
    CorrectionOutcome,
)
from evaluation.dataset_loader import (
    DatasetLoader,
    DatasetSample,
)

__all__ = [
    "EvaluationMetrics",
    "MetricsResult",
    "CorrectionOutcome",
    "DatasetLoader",
    "DatasetSample",
]
