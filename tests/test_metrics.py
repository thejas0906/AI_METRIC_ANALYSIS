"""
tests/test_metrics.py
======================
Unit Tests for the Evaluation Metrics Module
---------------------------------------------
Tests all 8 evaluation metrics against hand-computed expected values
to ensure correctness of the metrics implementation.

Run with:
    pytest tests/test_metrics.py -v
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from evaluation.metrics import EvaluationMetrics, CorrectionOutcome, MetricsResult


# ──────────────────────────────────────────────────────────────────
# Fixtures
# ──────────────────────────────────────────────────────────────────

@pytest.fixture
def perfect_detector():
    """Ground truth and predictions that are perfectly aligned."""
    ground_truth = [True, False, True, False, True]
    predictions  = [True, False, True, False, True]
    return ground_truth, predictions


@pytest.fixture
def imperfect_detector():
    """A realistic detector with FP and FN errors."""
    ground_truth = [True,  False, True,  False, True,  False]
    predictions  = [True,  False, False, True,  True,  False]
    # TP=2, TN=2, FP=1, FN=1
    return ground_truth, predictions


@pytest.fixture
def sample_correction_outcomes():
    """Sample CorrectionOutcome objects for CSR/CPR/UMR/FRA tests."""
    return [
        # Claim 1: GT=hallucinated, predicted=hallucinated, corrected+verified
        CorrectionOutcome("claim1", True,  True,  True,  True,  False, False),
        # Claim 2: GT=supported, predicted=supported, preserved
        CorrectionOutcome("claim2", False, False, False, False, True,  True),
        # Claim 3: GT=hallucinated, predicted=hallucinated, corrected but not verified
        CorrectionOutcome("claim3", True,  True,  True,  False, False, False),
        # Claim 4: GT=supported, predicted=hallucinated (FP) — was modified
        CorrectionOutcome("claim4", False, True,  True,  False, False, True),
        # Claim 5: GT=supported, predicted=supported, preserved
        CorrectionOutcome("claim5", False, False, False, False, True,  True),
    ]


# ──────────────────────────────────────────────────────────────────
# Tests: Core Classification Metrics
# ──────────────────────────────────────────────────────────────────

class TestCoreMetrics:
    """Tests for Accuracy, Precision, Recall, F1."""

    def test_perfect_accuracy(self, perfect_detector):
        gt, pred = perfect_detector
        calc = EvaluationMetrics()
        result = calc.compute(gt, pred)
        assert result.accuracy == pytest.approx(1.0)

    def test_perfect_precision_recall_f1(self, perfect_detector):
        gt, pred = perfect_detector
        calc = EvaluationMetrics()
        result = calc.compute(gt, pred)
        assert result.precision == pytest.approx(1.0)
        assert result.recall    == pytest.approx(1.0)
        assert result.f1_score  == pytest.approx(1.0)

    def test_imperfect_accuracy(self, imperfect_detector):
        gt, pred = imperfect_detector
        calc = EvaluationMetrics()
        result = calc.compute(gt, pred)
        # TP=2, TN=2, FP=1, FN=1 → Accuracy = 4/6
        assert result.accuracy == pytest.approx(4 / 6, abs=1e-4)

    def test_imperfect_precision(self, imperfect_detector):
        gt, pred = imperfect_detector
        calc = EvaluationMetrics()
        result = calc.compute(gt, pred)
        # Precision = TP / (TP + FP) = 2 / 3
        assert result.precision == pytest.approx(2 / 3, abs=1e-4)

    def test_imperfect_recall(self, imperfect_detector):
        gt, pred = imperfect_detector
        calc = EvaluationMetrics()
        result = calc.compute(gt, pred)
        # Recall = TP / (TP + FN) = 2 / 3
        assert result.recall == pytest.approx(2 / 3, abs=1e-4)

    def test_f1_harmonic_mean(self, imperfect_detector):
        gt, pred = imperfect_detector
        calc = EvaluationMetrics()
        result = calc.compute(gt, pred)
        # F1 = 2 * P * R / (P + R) = 2 * (2/3) * (2/3) / (4/3) = 2/3
        assert result.f1_score == pytest.approx(2 / 3, abs=1e-4)

    def test_all_hallucinated_no_fp(self):
        """When all claims are hallucinated and all detected correctly."""
        gt   = [True, True, True]
        pred = [True, True, True]
        calc = EvaluationMetrics()
        result = calc.compute(gt, pred)
        assert result.accuracy  == pytest.approx(1.0)
        assert result.precision == pytest.approx(1.0)
        assert result.recall    == pytest.approx(1.0)

    def test_zero_division_safe_precision(self):
        """Precision = 0 when no positive predictions."""
        gt   = [True, True]
        pred = [False, False]
        calc = EvaluationMetrics()
        result = calc.compute(gt, pred)
        # TP=0, FP=0 → precision = 0 (not division error)
        assert result.precision == pytest.approx(0.0)
        assert result.f1_score  == pytest.approx(0.0)

    def test_confusion_matrix_counts(self, imperfect_detector):
        gt, pred = imperfect_detector
        calc = EvaluationMetrics()
        result = calc.compute(gt, pred)
        assert result.true_positives  == 2
        assert result.true_negatives  == 2
        assert result.false_positives == 1
        assert result.false_negatives == 1


# ──────────────────────────────────────────────────────────────────
# Tests: Correction-Specific Metrics
# ──────────────────────────────────────────────────────────────────

class TestCorrectionMetrics:
    """Tests for CSR, CPR, UMR, FRA."""

    def test_csr_basic(self, sample_correction_outcomes):
        """
        CSR = successfully_corrected_hallucinations / total_hallucinations
        Total hallucinations = 2 (claims 1 & 3)
        Successfully corrected = 1 (claim 1 only — claim 3 not verified)
        CSR = 1/2 = 0.5
        """
        calc = EvaluationMetrics()
        gt   = [o.ground_truth_label for o in sample_correction_outcomes]
        pred = [o.predicted_label    for o in sample_correction_outcomes]
        result = calc.compute(gt, pred, sample_correction_outcomes)
        assert result.csr == pytest.approx(0.5, abs=1e-4)

    def test_cpr_basic(self, sample_correction_outcomes):
        """
        CPR = preserved_supported / total_supported
        Total supported = 3 (claims 2, 4, 5) — ground_truth=False
        Preserved = 2 (claims 2 & 5; claim 4 was modified — FP)
        CPR = 2/3
        """
        calc = EvaluationMetrics()
        gt   = [o.ground_truth_label for o in sample_correction_outcomes]
        pred = [o.predicted_label    for o in sample_correction_outcomes]
        result = calc.compute(gt, pred, sample_correction_outcomes)
        assert result.cpr == pytest.approx(2 / 3, abs=1e-4)

    def test_umr_basic(self, sample_correction_outcomes):
        """
        UMR = modified_supported / total_supported = 1/3
        CPR + UMR must equal 1.0
        """
        calc = EvaluationMetrics()
        gt   = [o.ground_truth_label for o in sample_correction_outcomes]
        pred = [o.predicted_label    for o in sample_correction_outcomes]
        result = calc.compute(gt, pred, sample_correction_outcomes)
        assert result.umr == pytest.approx(1 / 3, abs=1e-4)
        assert result.cpr + result.umr == pytest.approx(1.0, abs=1e-4)

    def test_fra_basic(self, sample_correction_outcomes):
        """
        FRA = verified_correct_in_final / total
        Verified correct:
          - Claim 2: supported + preserved ✓
          - Claim 5: supported + preserved ✓
          - Claim 1: hallucinated + corrected + verified ✓
        Total = 5
        FRA = 3/5 = 0.6
        """
        calc = EvaluationMetrics()
        gt   = [o.ground_truth_label for o in sample_correction_outcomes]
        pred = [o.predicted_label    for o in sample_correction_outcomes]
        result = calc.compute(gt, pred, sample_correction_outcomes)
        assert result.fra == pytest.approx(0.6, abs=1e-4)

    def test_csr_no_hallucinations(self):
        """CSR = 0 when there are no hallucinated claims."""
        outcomes = [
            CorrectionOutcome("c1", False, False, False, False, True,  True),
            CorrectionOutcome("c2", False, False, False, False, True,  True),
        ]
        calc = EvaluationMetrics()
        gt   = [o.ground_truth_label for o in outcomes]
        pred = [o.predicted_label    for o in outcomes]
        result = calc.compute(gt, pred, outcomes)
        assert result.csr == pytest.approx(0.0)

    def test_perfect_cpr(self):
        """CPR = 1.0 when all supported claims are preserved."""
        outcomes = [
            CorrectionOutcome("c1", True,  True,  True,  True,  False, False),
            CorrectionOutcome("c2", False, False, False, False, True,  True),
            CorrectionOutcome("c3", False, False, False, False, True,  True),
        ]
        calc = EvaluationMetrics()
        gt   = [o.ground_truth_label for o in outcomes]
        pred = [o.predicted_label    for o in outcomes]
        result = calc.compute(gt, pred, outcomes)
        assert result.cpr == pytest.approx(1.0)
        assert result.umr == pytest.approx(0.0)


# ──────────────────────────────────────────────────────────────────
# Tests: Edge Cases
# ──────────────────────────────────────────────────────────────────

class TestEdgeCases:
    """Tests for edge cases and robustness."""

    def test_empty_inputs(self):
        """Empty inputs should return zeros without errors."""
        calc = EvaluationMetrics()
        result = calc.compute([], [])
        assert result.accuracy == pytest.approx(0.0)

    def test_single_hallucinated_claim(self):
        """Single TP case."""
        calc = EvaluationMetrics()
        result = calc.compute([True], [True])
        assert result.accuracy  == pytest.approx(1.0)
        assert result.precision == pytest.approx(1.0)
        assert result.recall    == pytest.approx(1.0)
        assert result.f1_score  == pytest.approx(1.0)

    def test_single_supported_claim(self):
        """Single TN case."""
        calc = EvaluationMetrics()
        result = calc.compute([False], [False])
        assert result.accuracy == pytest.approx(1.0)

    def test_all_false_positives(self):
        """All supported claims flagged as hallucinated → UMR = 1."""
        outcomes = [
            CorrectionOutcome("c1", False, True, True, False, False, True),
            CorrectionOutcome("c2", False, True, True, False, False, True),
        ]
        calc = EvaluationMetrics()
        gt   = [o.ground_truth_label for o in outcomes]
        pred = [o.predicted_label    for o in outcomes]
        result = calc.compute(gt, pred, outcomes)
        assert result.umr == pytest.approx(1.0)
        assert result.cpr == pytest.approx(0.0)

    def test_mismatched_lengths_raises(self):
        """Mismatched ground_truth and predictions should raise AssertionError."""
        calc = EvaluationMetrics()
        with pytest.raises(AssertionError):
            calc.compute([True, False], [True])
