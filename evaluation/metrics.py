"""
metrics.py
===========
Evaluation Metrics Module
--------------------------
Implements all research evaluation metrics for the
Selective Evidence-Guided Hallucination Correction Framework.

Metrics Implemented:
--------------------

1. Hallucination Detection Accuracy (HDA)
   Fraction of claims correctly classified as hallucinated or not.

2. Precision
   Of all claims flagged as hallucinated, what fraction truly are?

3. Recall
   Of all truly hallucinated claims, what fraction did we detect?

4. F1 Score
   Harmonic mean of Precision and Recall.

5. Correction Success Rate (CSR)
   CSR = (Correctly Corrected Hallucinations) / (Total Hallucinations)
   Measures the effectiveness of the correction step.

6. Claim Preservation Rate (CPR)
   CPR = (Preserved Supported Claims) / (Total Supported Claims)
   A high CPR means we didn't unnecessarily modify correct content.

7. Unnecessary Modification Rate (UMR)
   UMR = (Modified Supported Claims) / (Total Supported Claims)
   Low UMR is desirable; UMR = 1 - CPR.

8. Final Response Accuracy (FRA)
   FRA = (Verified Correct Claims After Correction) / (Total Claims)
   End-to-end metric measuring the quality of the final response.

Input Format:
-------------
All metrics expect:
- A list of ground-truth labels (True = hallucinated, False = supported)
- A list of predicted labels from the framework
- Additional per-claim correction outcome data for CSR/CPR/UMR/FRA

Usage:
    metrics = EvaluationMetrics()
    results = metrics.compute(
        ground_truth=[True, False, True, False],
        predictions=[True, False, False, False],
        correction_outcomes=[...]
    )
    metrics.print_report(results)
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from loguru import logger

import numpy as np


# ------------------------------------------------------------------
# Data Structures
# ------------------------------------------------------------------

@dataclass
class CorrectionOutcome:
    """
    Per-claim data needed to compute CSR, CPR, UMR, and FRA.

    Attributes:
        claim:                 The atomic claim string.
        ground_truth_label:    True if claim is hallucinated.
        predicted_label:       True if framework predicted hallucination.
        was_corrected:         True if a correction was applied.
        post_correction_verified: True if corrected claim passed NLI verification.
        was_preserved:         True if claim was preserved (not modified).
        original_was_supported: True if claim was truly supported.
    """
    claim:                     str
    ground_truth_label:        bool   # True = hallucinated
    predicted_label:           bool   # True = framework flagged as hallucinated
    was_corrected:             bool   = False
    post_correction_verified:  bool   = False
    was_preserved:             bool   = False
    original_was_supported:    bool   = False


@dataclass
class MetricsResult:
    """
    Container for all computed evaluation metrics.

    Attributes:
        accuracy:                  Hallucination Detection Accuracy (HDA)
        precision:                 Detection Precision
        recall:                    Detection Recall
        f1_score:                  F1 Score
        csr:                       Correction Success Rate
        cpr:                       Claim Preservation Rate
        umr:                       Unnecessary Modification Rate
        fra:                       Final Response Accuracy
        true_positives:            # hallucinations correctly detected
        true_negatives:            # supported claims correctly identified
        false_positives:           # supported claims flagged as hallucination
        false_negatives:           # hallucinations missed
        total_hallucinated:        Total ground-truth hallucinated claims
        total_supported:           Total ground-truth supported claims
        total_corrected:           Claims that were corrected
        total_successfully_corrected: Corrections that passed verification
    """
    accuracy:                     float
    precision:                    float
    recall:                       float
    f1_score:                     float
    csr:                          float
    cpr:                          float
    umr:                          float
    fra:                          float
    true_positives:               int   = 0
    true_negatives:               int   = 0
    false_positives:              int   = 0
    false_negatives:              int   = 0
    total_hallucinated:           int   = 0
    total_supported:              int   = 0
    total_corrected:              int   = 0
    total_successfully_corrected: int   = 0


# ------------------------------------------------------------------
# Evaluation Metrics Calculator
# ------------------------------------------------------------------

class EvaluationMetrics:
    """
    Computes all evaluation metrics for the hallucination correction framework.

    Usage:
        metrics = EvaluationMetrics()
        result = metrics.compute(ground_truth, predictions, correction_outcomes)
        metrics.print_report(result)
        df = metrics.to_dataframe(result)
    """

    def compute(
        self,
        ground_truth: List[bool],
        predictions: List[bool],
        correction_outcomes: Optional[List[CorrectionOutcome]] = None,
    ) -> MetricsResult:
        """
        Compute all evaluation metrics.

        Args:
            ground_truth:        True = claim is hallucinated.
            predictions:         True = framework detected as hallucinated.
            correction_outcomes: Detailed per-claim correction data.
                                 Required for CSR, CPR, UMR, FRA.
                                 If None, these metrics will be 0.0.

        Returns:
            MetricsResult with all computed metrics.
        """
        assert len(ground_truth) == len(predictions), (
            "ground_truth and predictions must have the same length."
        )

        # -- Confusion Matrix --------------------------------------
        tp = sum(g and p for g, p in zip(ground_truth, predictions))
        tn = sum((not g) and (not p) for g, p in zip(ground_truth, predictions))
        fp = sum((not g) and p for g, p in zip(ground_truth, predictions))
        fn = sum(g and (not p) for g, p in zip(ground_truth, predictions))

        # -- Core Classification Metrics ---------------------------
        accuracy  = (tp + tn) / len(ground_truth) if ground_truth else 0.0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1        = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0 else 0.0
        )

        total_hallucinated = sum(ground_truth)
        total_supported    = len(ground_truth) - total_hallucinated

        # -- Correction-Specific Metrics ---------------------------
        csr = 0.0
        cpr = 0.0
        umr = 0.0
        fra = 0.0
        total_corrected = 0
        total_successfully_corrected = 0

        if correction_outcomes:
            csr, total_corrected, total_successfully_corrected = (
                self._compute_csr(correction_outcomes)
            )
            cpr, umr = self._compute_cpr_umr(correction_outcomes)
            fra      = self._compute_fra(correction_outcomes)

        logger.info(
            f"Metrics: Accuracy={accuracy:.3f}, P={precision:.3f}, "
            f"R={recall:.3f}, F1={f1:.3f}, CSR={csr:.3f}, "
            f"CPR={cpr:.3f}, UMR={umr:.3f}, FRA={fra:.3f}"
        )

        return MetricsResult(
            accuracy=accuracy,
            precision=precision,
            recall=recall,
            f1_score=f1,
            csr=csr,
            cpr=cpr,
            umr=umr,
            fra=fra,
            true_positives=tp,
            true_negatives=tn,
            false_positives=fp,
            false_negatives=fn,
            total_hallucinated=total_hallucinated,
            total_supported=total_supported,
            total_corrected=total_corrected,
            total_successfully_corrected=total_successfully_corrected,
        )

    # --------------------------------------------------------------
    # Individual Metric Computers
    # --------------------------------------------------------------

    @staticmethod
    def _compute_csr(
        outcomes: List[CorrectionOutcome],
    ) -> tuple:
        """
        Correction Success Rate (CSR):

            CSR = Corrected Hallucinated Claims / Total Hallucinated Claims

        A claim "correction" counts as successful if:
        - It was ground-truth hallucinated
        - The framework corrected it
        - The corrected version passed independent NLI verification

        Args:
            outcomes: List of CorrectionOutcome objects.

        Returns:
            Tuple of (CSR, total_corrected, total_successfully_corrected)
        """
        total_hallucinated = sum(1 for o in outcomes if o.ground_truth_label)
        if total_hallucinated == 0:
            return 0.0, 0, 0

        total_corrected = sum(1 for o in outcomes if o.was_corrected)

        # Successfully corrected = hallucinated AND corrected AND verified
        successfully_corrected = sum(
            1 for o in outcomes
            if o.ground_truth_label
            and o.was_corrected
            and o.post_correction_verified
        )

        csr = successfully_corrected / total_hallucinated
        return csr, total_corrected, successfully_corrected

    @staticmethod
    def _compute_cpr_umr(
        outcomes: List[CorrectionOutcome],
    ) -> tuple:
        """
        Claim Preservation Rate (CPR) and Unnecessary Modification Rate (UMR):

            CPR = Preserved Supported Claims / Total Supported Claims
            UMR = Modified Supported Claims  / Total Supported Claims
            Note: CPR + UMR = 1.0

        A "modified supported claim" is one that was ground-truth SUPPORTED
        but was changed by the framework (false positive correction).

        Args:
            outcomes: List of CorrectionOutcome objects.

        Returns:
            Tuple of (CPR, UMR)
        """
        supported_outcomes = [o for o in outcomes if not o.ground_truth_label]
        total_supported = len(supported_outcomes)

        if total_supported == 0:
            return 1.0, 0.0   # No supported claims → perfect preservation

        # A supported claim is "unnecessarily modified" if the framework
        # applied a correction to it (false positive)
        unnecessarily_modified = sum(
            1 for o in supported_outcomes if o.was_corrected
        )

        preserved = total_supported - unnecessarily_modified
        cpr = preserved / total_supported
        umr = unnecessarily_modified / total_supported

        return cpr, umr

    @staticmethod
    def _compute_fra(outcomes: List[CorrectionOutcome]) -> float:
        """
        Final Response Accuracy (FRA):

            FRA = Verified Correct Claims in Final Response / Total Claims

        A claim is "verified correct" in the final response if:
        - It was supported AND preserved (ground-truth correct, not modified)
        - OR it was hallucinated AND successfully corrected + verified

        Args:
            outcomes: List of CorrectionOutcome objects.

        Returns:
            FRA value in [0, 1].
        """
        total = len(outcomes)
        if total == 0:
            return 0.0

        verified_correct = 0
        for o in outcomes:
            if not o.ground_truth_label and o.was_preserved:
                # Correctly supported and preserved
                verified_correct += 1
            elif o.ground_truth_label and o.was_corrected and o.post_correction_verified:
                # Hallucination that was successfully corrected
                verified_correct += 1

        return verified_correct / total

    # --------------------------------------------------------------
    # Reporting
    # --------------------------------------------------------------

    @staticmethod
    def print_report(result: MetricsResult) -> None:
        """
        Print a formatted evaluation report to stdout.

        Args:
            result: MetricsResult from compute().
        """
        print("\n" + "=" * 65)
        print("  HALLUCINATION CORRECTION FRAMEWORK — EVALUATION REPORT")
        print("=" * 65)

        print("\n  -- Detection Performance ------------------------------")
        print(f"  Hallucination Detection Accuracy (HDA) : {result.accuracy:.4f}")
        print(f"  Precision                              : {result.precision:.4f}")
        print(f"  Recall                                 : {result.recall:.4f}")
        print(f"  F1 Score                               : {result.f1_score:.4f}")

        print("\n  -- Confusion Matrix -----------------------------------")
        print(f"  True Positives  (TP) : {result.true_positives}")
        print(f"  True Negatives  (TN) : {result.true_negatives}")
        print(f"  False Positives (FP) : {result.false_positives}")
        print(f"  False Negatives (FN) : {result.false_negatives}")

        print("\n  -- Correction Quality ---------------------------------")
        print(f"  Correction Success Rate  (CSR) : {result.csr:.4f}")
        print(f"  Claim Preservation Rate  (CPR) : {result.cpr:.4f}")
        print(f"  Unnecessary Modification (UMR) : {result.umr:.4f}")
        print(f"  Final Response Accuracy  (FRA) : {result.fra:.4f}")

        print("\n  -- Summary Statistics ---------------------------------")
        print(f"  Total Claims Evaluated : {result.true_positives + result.true_negatives + result.false_positives + result.false_negatives}")
        print(f"  Total Hallucinated     : {result.total_hallucinated}")
        print(f"  Total Supported        : {result.total_supported}")
        print(f"  Successfully Corrected : {result.total_successfully_corrected}")
        print("=" * 65 + "\n")

    def to_dataframe(self, result: MetricsResult):
        """
        Convert MetricsResult to a pandas DataFrame for easy export.

        Args:
            result: MetricsResult from compute().

        Returns:
            pandas.DataFrame with metric name → value columns.
        """
        try:
            import pandas as pd
        except ImportError:
            raise ImportError("pandas is required for to_dataframe(). Install it: pip install pandas")

        data = {
            "Metric": [
                "Hallucination Detection Accuracy (HDA)",
                "Precision",
                "Recall",
                "F1 Score",
                "Correction Success Rate (CSR)",
                "Claim Preservation Rate (CPR)",
                "Unnecessary Modification Rate (UMR)",
                "Final Response Accuracy (FRA)",
            ],
            "Value": [
                result.accuracy,
                result.precision,
                result.recall,
                result.f1_score,
                result.csr,
                result.cpr,
                result.umr,
                result.fra,
            ],
            "Description": [
                "Fraction of claims correctly classified",
                "Precision of hallucination detection",
                "Recall of hallucination detection",
                "Harmonic mean of Precision & Recall",
                "Corrected Hallucinations / Total Hallucinations",
                "Preserved Supported Claims / Total Supported",
                "Modified Supported Claims / Total Supported",
                "Verified Correct Claims / Total Claims",
            ],
        }
        return pd.DataFrame(data)

    def compute_from_pipeline_output(
        self,
        pipeline_results: List[Dict[str, Any]],
    ) -> MetricsResult:
        """
        Convenience method: compute metrics directly from pipeline output dicts.

        Each dict in pipeline_results should have:
            - 'ground_truth': bool (True = hallucinated)
            - 'predicted':    bool (True = detected as hallucinated)
            - 'was_corrected': bool
            - 'post_correction_verified': bool
            - 'was_preserved': bool
            - 'claim': str

        Args:
            pipeline_results: List of per-claim result dicts.

        Returns:
            MetricsResult
        """
        ground_truth = [r["ground_truth"] for r in pipeline_results]
        predictions  = [r["predicted"]    for r in pipeline_results]

        correction_outcomes = [
            CorrectionOutcome(
                claim=r.get("claim", ""),
                ground_truth_label=r["ground_truth"],
                predicted_label=r["predicted"],
                was_corrected=r.get("was_corrected", False),
                post_correction_verified=r.get("post_correction_verified", False),
                was_preserved=r.get("was_preserved", False),
                original_was_supported=not r["ground_truth"],
            )
            for r in pipeline_results
        ]

        return self.compute(ground_truth, predictions, correction_outcomes)
