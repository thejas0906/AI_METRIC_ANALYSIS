"""
tests/test_integration.py
==========================
End-to-end integration tests for the Hallucination Correction Pipeline.

These tests verify the complete pipeline flow using MOCKED evidence
retrieval and NLI inference so they run fast without network access
or heavy model loading.

For full end-to-end tests with real models, see test_nli_verifier.py
(marked @pytest.mark.slow).
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
import unittest.mock as mock
from dataclasses import dataclass, field
from typing import List

from config import FrameworkConfig
from retrieval.claim_extractor import ClaimExtractor, ClaimSpan
from retrieval.models import EvidenceItem, RetrievalResult
from verification.evidence_quality import (
    EvidenceQualityAssessor, QualityAssessedRetrieval, ScoredEvidence
)
from verification.nli_verifier import VerificationLabel, ClaimVerificationResult, NLIResult
from correction.claim_corrector import CorrectedClaim, CorrectionStatus
from correction.response_reconstructor import ResponseReconstructor


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def make_se(passage: str, eqs: float = 0.8) -> ScoredEvidence:
    ev = EvidenceItem(
        passage=passage, source_url="https://en.wikipedia.org/wiki/Test",
        source_type="wikipedia", page_title="Test",
        relevance_score=1.0, reliability_weight=0.8
    )
    return ScoredEvidence(
        original=ev, reliability_weight=0.8,
        normalized_relevance=eqs, evidence_quality_score=eqs
    )


def make_cvr(claim: str, label: VerificationLabel, css: float) -> ClaimVerificationResult:
    return ClaimVerificationResult(claim=claim, label=label, css=css)


# ------------------------------------------------------------------
# TEST 1: Supported claim -- no modification
# ------------------------------------------------------------------

class TestSupportedClaimPreservation:

    def test_supported_claim_is_preserved(self):
        """
        A claim that is SUPPORTED must not be modified.
        The corrected_claim text must equal the original_claim.
        """
        from correction.claim_corrector import ClaimCorrector
        config = FrameworkConfig()
        corrector = ClaimCorrector(config)

        vr = make_cvr(
            "Albert Einstein was born in 1879.",
            VerificationLabel.SUPPORTED,
            css=0.85,
        )
        qa = QualityAssessedRetrieval(
            claim="Albert Einstein was born in 1879.",
            scored_evidence=[make_se("Einstein born 1879.", eqs=0.8)],
            best_eqs=0.8,
        )

        cc = corrector.correct(vr, qa)

        assert cc.status == CorrectionStatus.PRESERVED
        assert cc.corrected_claim == cc.original_claim
        assert cc.corrected_claim == "Albert Einstein was born in 1879."


# ------------------------------------------------------------------
# TEST 2: Span preservation -- unchanged portions must be identical
# ------------------------------------------------------------------

class TestSpanPreservation:

    def test_unchanged_spans_are_identical(self):
        """
        When only one claim is corrected in a multi-claim response,
        the unchanged portions must be character-identical to the original.
        """
        original = (
            "Einstein was born in 1879. "
            "He worked at Harvard. "
            "He won the Nobel Prize in Physics."
        )
        config = FrameworkConfig()
        extractor = ClaimExtractor(config)
        spans = extractor.extract_spans(original)

        # Simulate: only the second claim is corrected
        corrected_claims = []
        for i, span in enumerate(spans):
            if i == 1:  # "He worked at Harvard."
                cc = CorrectedClaim(
                    original_claim=span.text,
                    corrected_claim="He worked at the Swiss Patent Office.",
                    status=CorrectionStatus.CORRECTED,
                    verification_result=make_cvr(
                        span.text, VerificationLabel.CONTRADICTED, 0.1
                    ),
                )
            else:
                cc = CorrectedClaim(
                    original_claim=span.text,
                    corrected_claim=span.text,
                    status=CorrectionStatus.PRESERVED,
                    verification_result=make_cvr(
                        span.text, VerificationLabel.SUPPORTED, 0.85
                    ),
                )
            corrected_claims.append(cc)

        reconstructor = ResponseReconstructor()
        response = reconstructor.reconstruct(
            corrected_claims=corrected_claims,
            claim_spans=spans,
            original_response=original,
        )

        # The first claim span in the original must appear unchanged
        first_span = spans[0]
        original_first_text = original[first_span.start_char:first_span.end_char]
        assert original_first_text in response.final_text, (
            f"First claim span not found unchanged in final text.\n"
            f"Original span: {original_first_text!r}\n"
            f"Final text: {response.final_text!r}"
        )

    def test_only_corrected_claim_changes(self):
        """
        Verify that the number of corrected claims equals 1 when only one
        claim was actually wrong.
        """
        original = (
            "Einstein was born in 1879. "
            "He worked at Harvard. "
            "He won the Nobel Prize in Physics."
        )
        config = FrameworkConfig()
        extractor = ClaimExtractor(config)
        spans = extractor.extract_spans(original)

        corrected_claims = []
        for i, span in enumerate(spans):
            if i == 1:
                cc = CorrectedClaim(
                    original_claim=span.text,
                    corrected_claim="He worked at the Swiss Patent Office.",
                    status=CorrectionStatus.CORRECTED,
                    verification_result=make_cvr(
                        span.text, VerificationLabel.CONTRADICTED, 0.1
                    ),
                )
            else:
                cc = CorrectedClaim(
                    original_claim=span.text,
                    corrected_claim=span.text,
                    status=CorrectionStatus.PRESERVED,
                    verification_result=make_cvr(
                        span.text, VerificationLabel.SUPPORTED, 0.85
                    ),
                )
            corrected_claims.append(cc)

        reconstructor = ResponseReconstructor()
        response = reconstructor.reconstruct(
            corrected_claims=corrected_claims,
            claim_spans=spans,
            original_response=original,
        )

        assert len(response.corrected_claims) == 1
        assert len(response.preserved_claims) == len(spans) - 1


# ------------------------------------------------------------------
# TEST 3: Unverifiable claims -- no fabricated correction
# ------------------------------------------------------------------

class TestUnverifiableClaim:

    def test_unverifiable_claim_is_not_corrected(self):
        """
        A claim with UNVERIFIABLE evidence must not be corrected
        (it should be flagged, not corrected).
        """
        from correction.claim_corrector import ClaimCorrector
        config = FrameworkConfig()
        corrector = ClaimCorrector(config)

        vr = make_cvr(
            "The universe is approximately 13.8 billion years old.",
            VerificationLabel.UNVERIFIABLE,
            css=0.1,
        )
        qa = QualityAssessedRetrieval(
            claim="The universe is approximately 13.8 billion years old.",
            scored_evidence=[make_se("Unrelated passage about cooking.", eqs=0.05)],
            best_eqs=0.05,
        )

        cc = corrector.correct(vr, qa)

        # Must be FLAGGED, not CORRECTED
        assert cc.status in (CorrectionStatus.FLAGGED, CorrectionStatus.PRESERVED), (
            f"Unverifiable claim should not be CORRECTED, got: {cc.status}"
        )
        # The claim text must be preserved (no fabrication)
        assert cc.corrected_claim == cc.original_claim


# ------------------------------------------------------------------
# TEST 4: Mixed response (supported + contradicted + supported)
# ------------------------------------------------------------------

class TestMixedResponse:

    def test_only_contradicted_claim_is_modified(self):
        """
        In a response with mixed claim types, only CONTRADICTED claims
        must be modified.
        """
        original = (
            "Einstein was born in 1879. "
            "He was awarded the Nobel Prize for the photoelectric effect in 1905. "
            "He was born in Germany."
        )
        config = FrameworkConfig()
        extractor = ClaimExtractor(config)
        spans = extractor.extract_spans(original)

        # Simulate: second claim is contradicted (year is wrong)
        corrected_claims = []
        for i, span in enumerate(spans):
            if i == 1:
                cc = CorrectedClaim(
                    original_claim=span.text,
                    corrected_claim="He was awarded the Nobel Prize for the photoelectric effect in 1921.",
                    status=CorrectionStatus.CORRECTED,
                    verification_result=make_cvr(span.text, VerificationLabel.CONTRADICTED, 0.05),
                )
            else:
                cc = CorrectedClaim(
                    original_claim=span.text,
                    corrected_claim=span.text,
                    status=CorrectionStatus.PRESERVED,
                    verification_result=make_cvr(span.text, VerificationLabel.SUPPORTED, 0.85),
                )
            corrected_claims.append(cc)

        reconstructor = ResponseReconstructor()
        response = reconstructor.reconstruct(
            corrected_claims=corrected_claims,
            claim_spans=spans,
            original_response=original,
        )

        # Only one claim corrected
        assert len(response.corrected_claims) == 1
        # Corrected claim contains new year
        assert "1921" in response.final_text
        # Original preserved text still present
        assert "1879" in response.final_text


# ------------------------------------------------------------------
# TEST 5: Claim span extraction
# ------------------------------------------------------------------

class TestClaimSpanExtraction:

    def test_spans_have_valid_offsets(self):
        """Extracted claim spans must have valid start/end offsets."""
        text = "Einstein was born in 1879. He won the Nobel Prize in Physics."
        config = FrameworkConfig()
        extractor = ClaimExtractor(config)
        spans = extractor.extract_spans(text)

        assert len(spans) >= 1
        for span in spans:
            assert 0 <= span.start_char < len(text), f"start_char {span.start_char} out of range"
            assert span.start_char < span.end_char <= len(text), (
                f"end_char {span.end_char} out of range (start={span.start_char})"
            )
            # The span text in the original document should correspond to the original sentence
            original_span_text = text[span.start_char:span.end_char]
            assert len(original_span_text) > 0

    def test_backward_compat_extract_returns_strings(self):
        """extract() must still return List[str] for backward compatibility."""
        text = "Einstein was born in 1879. He won the Nobel Prize."
        config = FrameworkConfig()
        extractor = ClaimExtractor(config)
        claims = extractor.extract(text)
        assert isinstance(claims, list)
        for c in claims:
            assert isinstance(c, str)

    def test_claim_ids_are_sequential(self):
        """Claim IDs must be sequential starting from 0."""
        text = "Einstein was born in 1879. He won the Nobel Prize in Physics."
        config = FrameworkConfig()
        extractor = ClaimExtractor(config)
        spans = extractor.extract_spans(text)
        ids = [s.claim_id for s in spans]
        assert ids == list(range(len(spans)))


# ------------------------------------------------------------------
# TEST 6: Metrics
# ------------------------------------------------------------------

class TestMetrics:

    def test_cpr_umr_complement(self):
        """CPR + UMR must equal 1.0 when all claims are either supported or modified."""
        from evaluation.metrics import EvaluationMetrics, CorrectionOutcome
        metrics = EvaluationMetrics()

        outcomes = [
            CorrectionOutcome("c1", False, False, False, False, True,  True),
            CorrectionOutcome("c2", False, True,  True,  False, False, True),  # FP
            CorrectionOutcome("c3", True,  True,  True,  True,  False, False),
        ]
        gt = [o.ground_truth_label for o in outcomes]
        pr = [o.predicted_label    for o in outcomes]
        result = metrics.compute(gt, pr, outcomes)

        assert abs(result.cpr + result.umr - 1.0) < 1e-9

    def test_perfect_detection(self):
        """Perfect detection: precision=recall=f1=1.0."""
        from evaluation.metrics import EvaluationMetrics, CorrectionOutcome
        metrics = EvaluationMetrics()

        gt = [True, False, True, False]
        pr = [True, False, True, False]
        outcomes = [
            CorrectionOutcome(f"c{i}", g, p, g and p, g and p, not g, not g)
            for i, (g, p) in enumerate(zip(gt, pr))
        ]
        result = metrics.compute(gt, pr, outcomes)
        assert result.precision == pytest.approx(1.0)
        assert result.recall    == pytest.approx(1.0)
        assert result.f1_score  == pytest.approx(1.0)

    def test_zero_hallucinated_csr(self):
        """CSR should be 0.0 when there are no hallucinated claims."""
        from evaluation.metrics import EvaluationMetrics, CorrectionOutcome
        metrics = EvaluationMetrics()

        gt = [False, False]
        pr = [False, False]
        outcomes = [
            CorrectionOutcome("c1", False, False, False, False, True, True),
            CorrectionOutcome("c2", False, False, False, False, True, True),
        ]
        result = metrics.compute(gt, pr, outcomes)
        assert result.csr == pytest.approx(0.0)
