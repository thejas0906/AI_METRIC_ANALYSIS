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
    Container for all computed evaluation metrics with strict separation
    between Ground-Truth Metrics and Internal-Verifier Metrics.

    Distinguishes:
      1. Ground Truth Metrics (evaluation_type: 'ground_truth'):
         Evaluates actual correctness against external benchmark labels/corrections.
         - Accuracy (HDA), Precision, Recall, F1 Score
         - CSR_GT  (Ground Truth Correction Success Rate)
         - CPR_GT  (Ground Truth Claim Preservation Rate)
         - UMR_GT  (Ground Truth Unnecessary Modification Rate)
         - FRA_GT  (Ground Truth Final Response Accuracy)

      2. Internal Verifier Metrics (evaluation_type: 'internal_verifier'):
         Evaluates internal pipeline consistency using Phase-7 BART-NLI.
         Subject to self-verification bias; does NOT establish ground truth.
         - CSR_Verifier (Claims Accepted By Verifier / Total Corrected Claims)
         - Acceptance Rate (Claims Accepted / Total Correction Attempts)
         - Verification Pass Rate (Phase-7 Fresh Retrieval Pass Rate)
         - FRA_Verifier (Final Accuracy evaluated by internal verifier)
    """
    # Core Classification (Ground Truth)
    accuracy:                     float
    precision:                    float
    recall:                       float
    f1_score:                     float

    # Ground Truth Metrics (evaluation_type: ground_truth)
    csr_gt:                       Optional[float] = None
    cpr_gt:                       float = 0.0
    umr_gt:                       float = 0.0
    fra_gt:                       Optional[float] = None
    rps:                          float = 1.0  # Response Preservation Score

    # Internal Verifier Metrics (evaluation_type: internal_verifier)
    csr_verifier:                 float = 0.0
    acceptance_rate:              float = 0.0
    verification_pass_rate:       float = 0.0
    fra_verifier:                 float = 0.0

    # Aliases / Backward Compatibility
    csr:                          float = 0.0
    cpr:                          float = 0.0
    umr:                          float = 0.0
    fra:                          float = 0.0

    # Counts & Breakdown
    true_positives:               int   = 0
    true_negatives:               int   = 0
    false_positives:              int   = 0
    false_negatives:              int   = 0
    total_hallucinated:           int   = 0
    total_supported:              int   = 0
    total_corrected:              int   = 0
    total_successfully_corrected: int   = 0
    total_accepted_by_verifier:   int   = 0
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

    def to_dict(self) -> Dict[str, Any]:
        """Return structured metric dictionary with circularity warning metadata."""
        return {
            "ground_truth_metrics": {
                "evaluation_type": "ground_truth",
                "accuracy": self.accuracy,
                "precision": self.precision,
                "recall": self.recall,
                "f1_score": self.f1_score,
                "csr_gt": self.csr_gt,
                "cpr_gt": self.cpr_gt,
                "umr_gt": self.umr_gt,
                "fra_gt": self.fra_gt,
                "rps": self.rps,
                "total_hallucinated": self.total_hallucinated,
                "total_supported": self.total_supported,
            },
            "verifier_metrics": {
                "evaluation_type": "internal_verifier",
                "warning": (
                    "Internal verifier metrics evaluate pipeline self-consistency using "
                    "Phase-7 BART-NLI, not external ground-truth correctness."
                ),
                "csr_verifier": self.csr_verifier,
                "acceptance_rate": self.acceptance_rate,
                "verification_pass_rate": self.verification_pass_rate,
                "fra_verifier": self.fra_verifier,
                "total_corrected": self.total_corrected,
                "total_accepted_by_verifier": self.total_accepted_by_verifier,
            },
            # Flat attributes for backward compatibility
            "accuracy": self.accuracy,
            "precision": self.precision,
            "recall": self.recall,
            "f1_score": self.f1_score,
            "csr": self.csr,
            "cpr": self.cpr,
            "umr": self.umr,
            "fra": self.fra,
            "csr_gt": self.csr_gt,
            "cpr_gt": self.cpr_gt,
            "umr_gt": self.umr_gt,
            "fra_gt": self.fra_gt,
            "rps": self.rps,
            "csr_verifier": self.csr_verifier,
            "acceptance_rate": self.acceptance_rate,
            "verification_pass_rate": self.verification_pass_rate,
            "fra_verifier": self.fra_verifier,
        }


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
        response_reports: Optional[List[Dict[str, Any]]] = None,
    ) -> MetricsResult:
        """
        Compute all evaluation metrics.

        Args:
            ground_truth:        True = claim is hallucinated.
            predictions:         True = framework detected as hallucinated.
            correction_outcomes: Detailed per-claim correction data.
                                 Required for CSR, CPR, UMR, FRA.
                                 If None, these metrics will be 0.0.
            response_reports:    Optional list of response-level outcome summaries.
                                 Used to compute mean Response Preservation Score (RPS).

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

        # -- Metric Defaults ---------------------------------------
        csr_gt = None
        cpr_gt = 0.0
        umr_gt = 0.0
        fra_gt = None
        csr_verifier = 0.0
        acceptance_rate = 0.0
        verification_pass_rate = 0.0
        fra_verifier = 0.0
        csr = 0.0
        cpr = 0.0
        umr = 0.0
        fra = 0.0
        total_corrected = 0
        total_successfully_corrected = 0
        total_accepted_by_verifier = 0
        unverifiable_rate = 0.0
        incorrect_correction_rate = 0.0
        edit_rate = 0.0
        ground_truth_csr = None

        if correction_outcomes:
            total_corrected = sum(1 for o in correction_outcomes if o.was_corrected)
            total_accepted_by_verifier = sum(
                1 for o in correction_outcomes
                if o.was_corrected and o.post_correction_verified
            )

            # 1. Ground Truth Metrics (evaluation_type: ground_truth)
            # CSR_GT = Correctly Corrected Hallucinations / Total Ground-Truth Hallucinations
            has_gt = any(
                getattr(o, "gold_correction", None) is not None
                for o in correction_outcomes
                if o.ground_truth_label
            )
            if has_gt and total_hallucinated > 0:
                gt_correct = sum(
                    1 for o in correction_outcomes
                    if o.ground_truth_label and o.was_corrected and getattr(o, "is_ground_truth_correct", False)
                )
                csr_gt = gt_correct / total_hallucinated
                ground_truth_csr = csr_gt
                total_successfully_corrected = gt_correct
            else:
                gt_correct = 0
                csr_gt = None
                ground_truth_csr = None
                total_successfully_corrected = sum(
                    1 for o in correction_outcomes
                    if o.ground_truth_label and o.was_corrected and o.post_correction_verified
                )

            # CPR_GT & UMR_GT (Ground Truth supported claims)
            cpr_gt, umr_gt = self._compute_cpr_umr(correction_outcomes)
            cpr = cpr_gt
            umr = umr_gt

            # FRA_GT: (Preserved Supported + GT Correct Corrections) / Total Claims
            if has_gt and len(correction_outcomes) > 0:
                preserved_supported = sum(
                    1 for o in correction_outcomes
                    if not o.ground_truth_label and o.was_preserved
                )
                fra_gt = (preserved_supported + gt_correct) / len(correction_outcomes)
            else:
                fra_gt = None

            # 2. Internal Verifier Metrics (evaluation_type: internal_verifier)
            # CSR_Verifier = Claims Accepted By Phase-7 Verification / Total Corrected Claims
            csr_verifier = (
                total_accepted_by_verifier / total_corrected
                if total_corrected > 0 else 0.0
            )
            acceptance_rate = csr_verifier
            verification_pass_rate = (
                total_accepted_by_verifier / total_corrected
                if total_corrected > 0 else 0.0
            )
            fra_verifier = self._compute_fra(correction_outcomes)

            # 3. Backward Compatibility Fallbacks
            if csr_gt is not None:
                csr = csr_gt
                fra = fra_gt if fra_gt is not None else fra_verifier
            else:
                csr = (
                    total_successfully_corrected / total_hallucinated
                    if total_hallucinated > 0 else 0.0
                )
                fra = fra_verifier

            # Secondary rates
            failed_corrections = sum(
                1 for o in correction_outcomes
                if o.was_corrected and not o.post_correction_verified
            )
            incorrect_correction_rate = (
                failed_corrections / total_corrected
                if total_corrected > 0 else 0.0
            )

            unverifiable_count = sum(
                1 for o in correction_outcomes
                if getattr(o, "was_unverifiable", False)
            )
            unverifiable_rate = (
                unverifiable_count / len(correction_outcomes)
                if correction_outcomes else 0.0
            )

            edit_rate = (
                total_corrected / len(correction_outcomes)
                if correction_outcomes else 0.0
            )

        rps_val = 1.0
        if response_reports:
            rps_list = [r["rps"] for r in response_reports if "rps" in r and r["rps"] is not None]
            if rps_list:
                rps_val = float(np.mean(rps_list))

        logger.info(
            f"Metrics: Accuracy={accuracy:.3f}, P={precision:.3f}, "
            f"R={recall:.3f}, F1={f1:.3f}, "
            f"CSR_GT={csr_gt if csr_gt is not None else 'N/A'}, "
            f"CSR_Verifier={csr_verifier:.3f}, "
            f"CPR_GT={cpr_gt:.3f}, UMR_GT={umr_gt:.3f}, RPS={rps_val:.4f}, "
            f"FRA_GT={fra_gt if fra_gt is not None else 'N/A'}, "
            f"FRA_Verifier={fra_verifier:.3f}"
        )

        return MetricsResult(
            accuracy=accuracy,
            precision=precision,
            recall=recall,
            f1_score=f1,
            csr_gt=csr_gt,
            cpr_gt=cpr_gt,
            umr_gt=umr_gt,
            fra_gt=fra_gt,
            rps=rps_val,
            csr_verifier=csr_verifier,
            acceptance_rate=acceptance_rate,
            verification_pass_rate=verification_pass_rate,
            fra_verifier=fra_verifier,
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
            total_accepted_by_verifier=total_accepted_by_verifier,
            unverifiable_rate=unverifiable_rate,
            incorrect_correction_rate=incorrect_correction_rate,
            edit_rate=edit_rate,
            is_verifier_based_evaluation=(csr_gt is None),
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
        Print a formatted evaluation report to stdout with clear separation
        between Ground-Truth Metrics and Internal-Verifier Metrics.

        Args:
            result: MetricsResult from compute().
        """
        print("\n" + "=" * 75)
        print("  HALLUCINATION CORRECTION FRAMEWORK — EVALUATION REPORT")
        print("=" * 75)

        print("\n  -- Detection Performance [evaluation_type: ground_truth] ----------------")
        print(f"  Hallucination Detection Accuracy (HDA) : {result.accuracy:.4f}")
        print(f"  Precision                              : {result.precision:.4f}")
        print(f"  Recall                                 : {result.recall:.4f}")
        print(f"  F1 Score                               : {result.f1_score:.4f}")

        print("\n  -- Confusion Matrix -----------------------------------------------------")
        print(f"  True Positives  (TP) : {result.true_positives}")
        print(f"  True Negatives  (TN) : {result.true_negatives}")
        print(f"  False Positives (FP) : {result.false_positives}")
        print(f"  False Negatives (FN) : {result.false_negatives}")

        print("\n  -- Ground Truth Metrics [evaluation_type: ground_truth] -----------------")
        csr_gt_str = f"{result.csr_gt:.4f}" if result.csr_gt is not None else "N/A (No Gold Target Corrections)"
        fra_gt_str = f"{result.fra_gt:.4f}" if result.fra_gt is not None else "N/A (No Gold Target Corrections)"
        print(f"  CSR_GT  (Ground Truth Correction Success)  : {csr_gt_str}")
        print(f"  CPR_GT  (Claim Preservation Rate)          : {result.cpr_gt:.4f}")
        print(f"  UMR_GT  (Unnecessary Modification Rate)    : {result.umr_gt:.4f}")
        print(f"  RPS     (Response Preservation Score)      : {result.rps:.4f}")
        print(f"  FRA_GT  (Final Response Accuracy - GT)     : {fra_gt_str}")

        print("\n  -- Verifier Metrics [evaluation_type: internal_verifier] ----------------")
        print("  [CIRCULARITY NOTICE: Evaluates internal pipeline consistency via")
        print("   Phase-7 BART-NLI verifier, not independent external ground truth.]")
        print(f"  CSR_Verifier (Claims Accepted / Modified)  : {result.csr_verifier:.4f}")
        print(f"  Acceptance Rate (Claims Accepted / Total)  : {result.acceptance_rate:.4f}")
        print(f"  Verification Pass Rate (Phase-7 Pass Rate) : {result.verification_pass_rate:.4f}")
        print(f"  FRA_Verifier (Final Accuracy by Verifier)  : {result.fra_verifier:.4f}")

        print("\n  -- Summary Statistics ---------------------------------------------------")
        total_eval = result.true_positives + result.true_negatives + result.false_positives + result.false_negatives
        print(f"  Total Claims Evaluated : {total_eval}")
        print(f"  Total Hallucinated     : {result.total_hallucinated}")
        print(f"  Total Supported        : {result.total_supported}")
        print(f"  Total Modified         : {result.total_corrected}")
        print(f"  Accepted by Verifier   : {result.total_accepted_by_verifier}")
        if result.csr_gt is not None:
            print(f"  GT Successfully Corrected: {result.total_successfully_corrected}")
        print("=" * 75 + "\n")

    @staticmethod
    def print_response_level_report(response_reports: List[Dict[str, Any]]) -> None:
        """
        Print a formatted table of response-level evaluation outcomes.

        For each evaluated response shows:
        - Response ID
        - Total Claims
        - Supported Claims
        - Contradicted Claims
        - Unverifiable Claims
        - Corrected Claims
        - Preserved Claims
        - Failed Corrections
        Plus aggregate dataset-level statistics.
        """
        if not response_reports:
            return

        print("\n" + "=" * 98)
        print("  RESPONSE-LEVEL SELECTIVE CORRECTION & PRESERVATION REPORT")
        print("=" * 98)
        header = (
            f"  {'Response ID':<12} | {'Total':<5} | {'Supp':<5} | {'Contra':<6} | "
            f"{'Corr':<5} | {'Pres':<5} | {'CPR':<6} | {'UMR':<6} | {'RPS':<6} | Status"
        )
        print(header)
        print("  " + "-" * 94)

        agg = {
            "total_responses": len(response_reports),
            "total_claims": 0,
            "supported": 0,
            "contradicted": 0,
            "unverifiable": 0,
            "corrected": 0,
            "preserved": 0,
            "failed": 0,
            "responses_with_corrections": 0,
            "perfect_preservation_responses": 0,
        }

        all_cprs = []
        all_umrs = []
        all_rps = []

        for r in response_reports:
            resp_id = str(r.get("response_id", "N/A"))
            tot = r.get("total_claims", 0)
            sup = r.get("supported_claims", 0)
            con = r.get("contradicted_claims", 0)
            unv = r.get("unverifiable_claims", 0)
            cor = r.get("corrected_claims", 0)
            pre = r.get("preserved_claims", 0)
            fai = r.get("failed_corrections", 0)

            cpr_val = r.get("cpr")
            if cpr_val is None:
                cpr_val = (pre / sup) if sup > 0 else 1.0
            umr_val = r.get("umr")
            if umr_val is None:
                umr_val = (1.0 - cpr_val) if sup > 0 else 0.0
            rps_val = r.get("rps", 1.0)

            all_cprs.append(cpr_val)
            all_umrs.append(umr_val)
            all_rps.append(rps_val)

            agg["total_claims"] += tot
            agg["supported"] += sup
            agg["contradicted"] += con
            agg["unverifiable"] += unv
            agg["corrected"] += cor
            agg["preserved"] += pre
            agg["failed"] += fai

            if cor > 0:
                agg["responses_with_corrections"] += 1
            if sup > 0 and pre == sup:
                agg["perfect_preservation_responses"] += 1

            status_note = "SELECTIVE_EDIT" if (cor > 0 and pre > 0) else ("PRESERVED" if cor == 0 else "CORRECTED")
            print(
                f"  {resp_id:<12} | {tot:<5} | {sup:<5} | {con:<6} | "
                f"{cor:<5} | {pre:<5} | {cpr_val:.2f}   | {umr_val:.2f}   | {rps_val:.4f} | {status_note}"
            )

        mean_cpr = float(np.mean(all_cprs)) if all_cprs else 1.0
        mean_umr = float(np.mean(all_umrs)) if all_umrs else 0.0
        mean_rps = float(np.mean(all_rps)) if all_rps else 1.0

        print("  " + "-" * 94)
        print(
            f"  {'TOTAL':<12} | {agg['total_claims']:<5} | {agg['supported']:<5} | {agg['contradicted']:<6} | "
            f"{agg['corrected']:<5} | {agg['preserved']:<5} | {mean_cpr:.2f}   | {mean_umr:.2f}   | {mean_rps:.4f} | "
            f"{agg['total_responses']} responses"
        )
        print("=" * 98)

        cpr_pct = mean_cpr * 100
        csr_pct = (agg["corrected"] / agg["contradicted"] * 100) if agg["contradicted"] > 0 else 0.0
        print("\n  -- Response-Level Aggregate Insights ---------------------------------")
        print(f"  Total Responses Evaluated            : {agg['total_responses']}")
        print(f"  Responses with Selective Edits       : {agg['responses_with_corrections']}")
        print(f"  Responses with 100% CPR (Preserved)  : {agg['perfect_preservation_responses']}")
        print(f"  Mean Claim Preservation Rate (CPR)   : {cpr_pct:.1f}%")
        print(f"  Mean Unnecessary Modification (UMR)  : {mean_umr * 100:.1f}%")
        print(f"  Mean Response Preservation Score(RPS): {mean_rps:.4f}")
        print(f"  Aggregate Correction Success Rate    : {csr_pct:.1f}%")
        print("=" * 98 + "\n")

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

        csr_gt_val = result.csr_gt if result.csr_gt is not None else "N/A"
        fra_gt_val = result.fra_gt if result.fra_gt is not None else "N/A"

        data = {
            "Metric": [
                # Ground Truth Detection
                "Hallucination Detection Accuracy (HDA)",
                "Precision",
                "Recall",
                "F1 Score",
                # Ground Truth Correction & Preservation
                "CSR_GT (Ground Truth CSR)",
                "CPR_GT (Claim Preservation Rate)",
                "UMR_GT (Unnecessary Modification Rate)",
                "FRA_GT (Ground Truth Final Accuracy)",
                # Internal Verifier Metrics
                "CSR_Verifier (Claims Accepted / Modified)",
                "Acceptance Rate",
                "Verification Pass Rate",
                "FRA_Verifier (Final Accuracy by Verifier)",
            ],
            "Value": [
                result.accuracy,
                result.precision,
                result.recall,
                result.f1_score,
                csr_gt_val,
                result.cpr_gt,
                result.umr_gt,
                fra_gt_val,
                result.csr_verifier,
                result.acceptance_rate,
                result.verification_pass_rate,
                result.fra_verifier,
            ],
            "Category": [
                "Ground Truth",
                "Ground Truth",
                "Ground Truth",
                "Ground Truth",
                "Ground Truth",
                "Ground Truth",
                "Ground Truth",
                "Ground Truth",
                "Internal Verifier",
                "Internal Verifier",
                "Internal Verifier",
                "Internal Verifier",
            ],
            "Evaluation Type": [
                "ground_truth",
                "ground_truth",
                "ground_truth",
                "ground_truth",
                "ground_truth",
                "ground_truth",
                "ground_truth",
                "ground_truth",
                "internal_verifier",
                "internal_verifier",
                "internal_verifier",
                "internal_verifier",
            ],
            "Description": [
                "Fraction of claims correctly classified (TP+TN)/Total",
                "Precision of hallucination detection TP/(TP+FP)",
                "Recall of hallucination detection TP/(TP+FN)",
                "Harmonic mean of Precision & Recall",
                "Correctly corrected hallucinations / total GT hallucinations (matches gold targets)",
                "Preserved supported claims / total GT supported claims",
                "Modified supported claims / total GT supported claims (1 - CPR)",
                "Preserved supported + GT corrected / total claims",
                "Claims accepted by Phase-7 NLI / total claims modified (Self-Verification)",
                "Fraction of modified claims accepted by Phase-7 NLI",
                "Pass rate of Phase-7 fresh retrieval independent verification",
                "Final response accuracy evaluated by internal verifier",
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
            - 'gold_correction': Optional[str]
            - 'is_ground_truth_correct': Optional[bool]

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
                gold_label=r.get("gold_label", ""),
                gold_correction=r.get("gold_correction", None),
                is_ground_truth_correct=r.get("is_ground_truth_correct", None),
            )
            for r in pipeline_results
        ]

        return self.compute(ground_truth, predictions, correction_outcomes)
