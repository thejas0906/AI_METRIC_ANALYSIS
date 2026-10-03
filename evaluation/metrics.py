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
    Per-claim data needed to compute CSR, CPR, UMR, FRA, and secondary rates.

    Attributes:
        claim:                    The atomic claim string.
        ground_truth_label:       True if claim is hallucinated / contradicted.
        predicted_label:          True if framework predicted hallucination / contradicted.
        was_corrected:            True if a correction was applied.
        post_correction_verified: True if corrected claim passed independent NLI verification.
        was_preserved:            True if claim was preserved (not modified).
        original_was_supported:   True if claim was truly supported.
        was_unverifiable:         True if framework classified as UNVERIFIABLE.
        gold_label:               Optional string: "SUPPORTED", "CONTRADICTED", "UNVERIFIABLE".
        gold_correction:          Optional external gold correction text.
        is_ground_truth_correct:  Optional bool if external gold correction was matched.
    """
    claim:                     str
    ground_truth_label:        bool   # True = hallucinated / contradicted
    predicted_label:           bool   # True = framework flagged as contradicted
    was_corrected:             bool   = False
    post_correction_verified:  bool   = False
    was_preserved:             bool   = False
    original_was_supported:    bool   = False
    was_unverifiable:          bool   = False
    gold_label:                str    = ""
    gold_correction:           Optional[str] = None
    is_ground_truth_correct:   Optional[bool] = None


@dataclass
class MetricsResult:
    """
    Container for all computed evaluation metrics.

    Distinguishes:
      A. Detection correctness (HDA, precision, recall, F1)
      B. Correction correctness (CSR, incorrect_correction_rate)
      C. Preservation correctness (CPR)
      D. Unnecessary modification (UMR, edit_rate)
      E. End-to-end response quality (FRA)

    Note on Correction Success Rate (CSR):
      When external gold corrections are not provided, CSR is strictly
      a *verifier-based evaluation* metric (independent fresh NLI verification),
      not external ground-truth correction accuracy.
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
    unverifiable_rate:            float = 0.0
    incorrect_correction_rate:    float = 0.0
    edit_rate:                    float = 0.0
    is_verifier_based_evaluation: bool  = True
    ground_truth_csr:             Optional[float] = None

    @property
    def detection_precision(self) -> float:
        return self.precision

    @property
    def detection_recall(self) -> float:
        return self.recall

    @property
    def detection_f1(self) -> float:
        return self.f1_score

    @property
    def modification_rate(self) -> float:
        return self.edit_rate


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
        unverifiable_rate = 0.0
        incorrect_correction_rate = 0.0
        edit_rate = 0.0
        ground_truth_csr = None

        if correction_outcomes:
            csr, total_corrected, total_successfully_corrected, incorrect_correction_rate, ground_truth_csr = (
                self._compute_csr(correction_outcomes)
            )
            cpr, umr = self._compute_cpr_umr(correction_outcomes)
            fra      = self._compute_fra(correction_outcomes)

            unverifiable_count = sum(1 for o in correction_outcomes if getattr(o, "was_unverifiable", False))
            unverifiable_rate = unverifiable_count / len(correction_outcomes) if correction_outcomes else 0.0

            total_modified = sum(1 for o in correction_outcomes if o.was_corrected)
            edit_rate = total_modified / len(correction_outcomes) if correction_outcomes else 0.0

        logger.info(
            f"Metrics: Accuracy={accuracy:.3f}, P={precision:.3f}, "
            f"R={recall:.3f}, F1={f1:.3f}, CSR={csr:.3f}, "
            f"CPR={cpr:.3f}, UMR={umr:.3f}, FRA={fra:.3f}, "
            f"UnverifiableRate={unverifiable_rate:.3f}, "
            f"IncorrectCorrectionRate={incorrect_correction_rate:.3f}"
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
            unverifiable_rate=unverifiable_rate,
            incorrect_correction_rate=incorrect_correction_rate,
            edit_rate=edit_rate,
            is_verifier_based_evaluation=(ground_truth_csr is None),
            ground_truth_csr=ground_truth_csr,
        )

    # --------------------------------------------------------------
    # Individual Metric Computers
    # --------------------------------------------------------------

    @staticmethod
    def _compute_csr(
        outcomes: List[CorrectionOutcome],
    ) -> tuple:
        """
        Correction Success Rate (CSR) and related correction quality metrics:

            CSR = Corrected Hallucinated Claims / Total Hallucinated Claims

        A claim "correction" counts as successfully verified if:
        - It was ground-truth hallucinated / contradicted
        - The framework applied a correction
        - The corrected version passed independent fresh-evidence NLI verification

        If external gold correction strings are provided, ground_truth_csr evaluates
        exact or semantic match against gold correction.

        Returns:
            Tuple of (CSR, total_corrected, successfully_corrected,
                      incorrect_correction_rate, ground_truth_csr)
        """
        total_hallucinated = sum(1 for o in outcomes if o.ground_truth_label)
        if total_hallucinated == 0:
            return 0.0, 0, 0, 0.0, None

        total_corrected = sum(1 for o in outcomes if o.was_corrected)

        # Successfully corrected = hallucinated AND corrected AND verified
        successfully_corrected = sum(
            1 for o in outcomes
            if o.ground_truth_label
            and o.was_corrected
            and o.post_correction_verified
        )

        csr = successfully_corrected / total_hallucinated

        # Incorrect correction rate: attempted corrections that failed or were unverified
        failed_corrections = sum(
            1 for o in outcomes
            if o.was_corrected and not o.post_correction_verified
        )
        incorrect_correction_rate = (failed_corrections / total_corrected) if total_corrected > 0 else 0.0

        # Ground-truth CSR if external gold corrections were supplied
        has_gt = any(getattr(o, "gold_correction", None) is not None for o in outcomes if o.ground_truth_label)
        if has_gt:
            gt_correct = sum(
                1 for o in outcomes
                if o.ground_truth_label and getattr(o, "is_ground_truth_correct", False)
            )
            ground_truth_csr = gt_correct / total_hallucinated
        else:
            ground_truth_csr = None

        return csr, total_corrected, successfully_corrected, incorrect_correction_rate, ground_truth_csr

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

        print("\n  -- Detection & Verification Rates ---------------------")
        print(f"  Detection Precision                    : {result.detection_precision:.4f}")
        print(f"  Detection Recall                       : {result.detection_recall:.4f}")
        print(f"  Detection F1                           : {result.detection_f1:.4f}")
        print(f"  Unverifiable Rate                      : {result.unverifiable_rate:.4f}")

        print("\n  -- Correction Quality & Preservation ------------------")
        csr_label = "Verifier-Based CSR" if result.is_verifier_based_evaluation else "Ground-Truth CSR"
        print(f"  Correction Success Rate ({csr_label}) : {result.csr:.4f}")
        if result.ground_truth_csr is not None:
            print(f"  Ground-Truth CSR                       : {result.ground_truth_csr:.4f}")
        print(f"  Incorrect Correction Rate              : {result.incorrect_correction_rate:.4f}")
        print(f"  Claim Preservation Rate  (CPR)         : {result.cpr:.4f}")
        print(f"  Unnecessary Modification (UMR)         : {result.umr:.4f}")
        print(f"  Modification / Edit Rate               : {result.edit_rate:.4f}")
        print(f"  Final Response Accuracy  (FRA)         : {result.fra:.4f}")

        print("\n  -- Summary Statistics ---------------------------------")
        print(f"  Total Claims Evaluated : {result.true_positives + result.true_negatives + result.false_positives + result.false_negatives}")
        print(f"  Total Hallucinated     : {result.total_hallucinated}")
        print(f"  Total Supported        : {result.total_supported}")
        print(f"  Total Modified         : {result.total_corrected}")
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
