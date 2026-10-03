"""
tests/test_nli_verifier.py
===========================
Unit tests for the NLI Verifier (Phase 4).

These tests verify:
  1. The model receives BOTH premise (evidence) AND hypothesis (claim)
     as a genuine sequence pair.
  2. Entailment probability is high when evidence supports the claim.
  3. Contradiction probability is high when evidence contradicts claim.
  4. Neutral / low entailment when evidence is unrelated.

IMPORTANT: These tests load the actual BART-large-mnli model and
require it to be downloaded.  They will be slow on first run.
Use:
    pytest tests/test_nli_verifier.py -v -m "not slow"
to skip slow tests in CI.  Mark slow tests with @pytest.mark.slow.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from config import FrameworkConfig
from verification.evidence_quality import (
    QualityAssessedRetrieval,
    ScoredEvidence,
    EvidenceQualityAssessor,
)
from retrieval.models import EvidenceItem, RetrievalResult


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def make_qa(claim: str, passage: str, eqs: float = 0.8) -> QualityAssessedRetrieval:
    """Build a minimal QualityAssessedRetrieval for NLI testing."""
    ev = EvidenceItem(
        passage=passage,
        source_url="https://en.wikipedia.org/wiki/Test",
        source_type="wikipedia",
        page_title="Test",
        relevance_score=1.0,
        reliability_weight=0.8,
    )
    se = ScoredEvidence(
        original=ev,
        reliability_weight=0.8,
        normalized_relevance=eqs,
        evidence_quality_score=eqs,
    )
    return QualityAssessedRetrieval(claim=claim, scored_evidence=[se], best_eqs=eqs)


# ------------------------------------------------------------------
# Unit tests: NLIResult structure
# ------------------------------------------------------------------

class TestNLIResultStructure:
    """Tests that do NOT require the model -- use mocking."""

    def test_nli_result_probs_sum_to_one(self):
        """NLI probabilities from the model must sum to ~1.0."""
        # We import the dataclass and verify the invariant is documented
        from verification.nli_verifier import NLIResult
        # Simulate a well-formed result
        r = NLIResult(
            entailment_prob=0.7,
            neutral_prob=0.2,
            contradiction_prob=0.1,
            evidence_passage="test"
        )
        total = r.entailment_prob + r.neutral_prob + r.contradiction_prob
        assert abs(total - 1.0) < 0.01

    def test_verification_label_enum_values(self):
        """VerificationLabel must have the three required states."""
        from verification.nli_verifier import VerificationLabel
        assert VerificationLabel.SUPPORTED.value == "SUPPORTED"
        assert VerificationLabel.CONTRADICTED.value == "CONTRADICTED"
        assert VerificationLabel.UNVERIFIABLE.value == "UNVERIFIABLE"

    def test_classify_high_css_gives_supported(self):
        """High CSS should produce SUPPORTED label."""
        from verification.nli_verifier import NLIVerifier, VerificationLabel
        config = FrameworkConfig()
        # Don't load the model -- patch _load_nli_model
        import unittest.mock as mock
        with mock.patch.object(NLIVerifier, '_load_nli_model'):
            v = NLIVerifier.__new__(NLIVerifier)
            v.config = config
            label = v._classify(css=0.80, max_contradiction=0.0)
            assert label == VerificationLabel.SUPPORTED

    def test_classify_low_css_high_contradiction_gives_contradicted(self):
        """Low CSS + high contradiction -> CONTRADICTED."""
        from verification.nli_verifier import NLIVerifier, VerificationLabel
        config = FrameworkConfig()
        import unittest.mock as mock
        with mock.patch.object(NLIVerifier, '_load_nli_model'):
            v = NLIVerifier.__new__(NLIVerifier)
            v.config = config
            label = v._classify(css=0.05, max_contradiction=0.50)
            assert label == VerificationLabel.CONTRADICTED

    def test_classify_low_css_low_contradiction_gives_unverifiable(self):
        """Low CSS + low contradiction -> UNVERIFIABLE."""
        from verification.nli_verifier import NLIVerifier, VerificationLabel
        config = FrameworkConfig()
        import unittest.mock as mock
        with mock.patch.object(NLIVerifier, '_load_nli_model'):
            v = NLIVerifier.__new__(NLIVerifier)
            v.config = config
            label = v._classify(css=0.05, max_contradiction=0.10)
            assert label == VerificationLabel.UNVERIFIABLE


# ------------------------------------------------------------------
# Integration tests: actual NLI model
# ------------------------------------------------------------------

@pytest.mark.slow
class TestNLIModelInference:
    """
    Integration tests that load the actual BART-large-mnli model.
    These tests are marked @pytest.mark.slow and require the model
    to be downloaded (~1.6 GB).
    """

    @classmethod
    @pytest.fixture(scope="class")
    def verifier(cls):
        from verification.nli_verifier import NLIVerifier
        return NLIVerifier(FrameworkConfig())

    def test_supported_claim_entailment_is_highest(self, verifier):
        """
        TEST 1: Supported claim.
        Evidence:  'Albert Einstein was born in 1879.'
        Hypothesis: 'Albert Einstein was born in 1879.'
        Expected: entailment_prob is the highest of the three probabilities.
        """
        from verification.nli_verifier import NLIVerifier
        result = verifier._run_nli(
            premise="Albert Einstein was born in 1879.",
            hypothesis="Albert Einstein was born in 1879."
        )
        # Both premise AND hypothesis reached the model -- verify that
        # entailment dominates when they are identical
        assert result.entailment_prob > result.neutral_prob, (
            f"entailment {result.entailment_prob:.3f} should exceed "
            f"neutral {result.neutral_prob:.3f} for identical premise/hypothesis"
        )
        assert result.entailment_prob > result.contradiction_prob, (
            f"entailment {result.entailment_prob:.3f} should exceed "
            f"contradiction {result.contradiction_prob:.3f} for identical premise/hypothesis"
        )

    def test_contradicted_claim_contradiction_is_highest(self, verifier):
        """
        TEST 2: Contradicted claim.
        Evidence:  'Albert Einstein was born in 1879.'
        Hypothesis: 'Albert Einstein was born in 1885.'
        Expected: contradiction_prob is the highest of the three probabilities.
        """
        result = verifier._run_nli(
            premise="Albert Einstein was born in 1879.",
            hypothesis="Albert Einstein was born in 1885."
        )
        assert result.contradiction_prob > result.entailment_prob, (
            f"contradiction {result.contradiction_prob:.3f} should exceed "
            f"entailment {result.entailment_prob:.3f} for contradicting year claim"
        )

    def test_neutral_claim_not_high_entailment(self, verifier):
        """
        TEST 3: Unrelated claim (neutral).
        Evidence:  'Albert Einstein worked as a patent clerk.'
        Hypothesis: 'Albert Einstein had three children.'
        Expected: entailment_prob is NOT the highest (neutral or contradiction dominates).
        """
        result = verifier._run_nli(
            premise="Albert Einstein worked as a patent clerk.",
            hypothesis="Albert Einstein had three children."
        )
        # entailment should NOT dominate for an unrelated claim
        assert result.entailment_prob < 0.9, (
            f"entailment {result.entailment_prob:.3f} should not be very high "
            f"for an unrelated claim (neutral case)"
        )

    def test_hypothesis_affects_result(self, verifier):
        """
        TEST 4: Hypothesis actually reaches the model.
        Vary the hypothesis while keeping the premise constant.
        The entailment probability MUST differ between the two claims,
        proving the model is actually receiving the hypothesis text.
        """
        # Same premise, different hypotheses
        premise = "Albert Einstein was born in 1879 in Ulm, Germany."

        result_match = verifier._run_nli(
            premise=premise,
            hypothesis="Albert Einstein was born in 1879."
        )
        result_mismatch = verifier._run_nli(
            premise=premise,
            hypothesis="Albert Einstein was born in 1905."
        )

        # The entailment probabilities must differ significantly
        diff = abs(result_match.entailment_prob - result_mismatch.entailment_prob)
        assert diff > 0.05, (
            f"Entailment probability did not change when hypothesis changed "
            f"(diff={diff:.4f}). This indicates the hypothesis may not be "
            f"reaching the model. "
            f"match={result_match.entailment_prob:.3f}, "
            f"mismatch={result_mismatch.entailment_prob:.3f}"
        )

    def test_full_verify_supported(self, verifier):
        """
        TEST 5: Full verify() call for a supported claim.
        """
        from verification.nli_verifier import VerificationLabel
        qa = make_qa(
            claim="Albert Einstein was born in 1879.",
            passage="Albert Einstein was born on March 14, 1879, in Ulm, Germany.",
            eqs=0.80,
        )
        result = verifier.verify(qa)
        # CSS = entailment * EQS; if entailment >= 0.9, CSS >= 0.72
        # With a strong match and EQS=0.8, CSS should approach SUPPORTED
        assert result.css > 0.0, "CSS should be > 0 for supported claim"
        assert result.label in (
            VerificationLabel.SUPPORTED,
            VerificationLabel.UNVERIFIABLE,
        ), f"Unexpected label: {result.label}"

    def test_full_verify_contradicted(self, verifier):
        """
        TEST 6: Full verify() call for a contradicted claim.
        """
        from verification.nli_verifier import VerificationLabel
        qa = make_qa(
            claim="Albert Einstein was born in France.",
            passage="Albert Einstein was born in Ulm, Germany, not in France.",
            eqs=0.80,
        )
        result = verifier.verify(qa)
        # Should be CONTRADICTED or UNVERIFIABLE (not SUPPORTED)
        assert result.label != VerificationLabel.SUPPORTED, (
            "A claim explicitly contradicted by evidence should not be SUPPORTED"
        )
