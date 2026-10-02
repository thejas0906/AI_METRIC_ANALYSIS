"""
tests/test_evidence_quality.py
================================
Unit Tests for the Evidence Quality Assessment Module (Phase 3)
---------------------------------------------------------------
Tests reliability weight lookup, EQS computation, source
classification, and edge case handling.

Normalization formula (K=0.30):
    norm = raw / (raw + 0.30)

This is a calibrated sigmoid normalization chosen so that a
"good" BM25-lite relevance score of ~0.30 maps to ~0.50,
avoiding the compression artifacts of the naive raw/(raw+1)
formula which maps most real scores below 0.33.

Run with:
    pytest tests/test_evidence_quality.py -v
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from config import FrameworkConfig, EVIDENCE_WEIGHTS
from retrieval.models import EvidenceItem, RetrievalResult
from verification.evidence_quality import (
    EvidenceQualityAssessor,
    ScoredEvidence,
    QualityAssessedRetrieval,
)


# ------------------------------------------------------------------
# Normalization constant (must match evidence_quality.py)
# ------------------------------------------------------------------
_K = 0.30  # calibration constant for _normalize_relevance


def expected_norm(raw: float) -> float:
    """Reference implementation of the normalization formula."""
    if raw <= 0:
        return 0.0
    return raw / (raw + _K)


# ------------------------------------------------------------------
# Fixtures
# ------------------------------------------------------------------

@pytest.fixture
def assessor():
    return EvidenceQualityAssessor(FrameworkConfig())


def make_evidence(source_type="wikipedia", relevance=1.0, url=None) -> EvidenceItem:
    """Helper to create a minimal EvidenceItem for testing."""
    return EvidenceItem(
        passage="Test evidence passage.",
        source_url=url or "https://en.wikipedia.org/wiki/Test",
        source_type=source_type,
        page_title="Test Page",
        relevance_score=relevance,
        reliability_weight=EVIDENCE_WEIGHTS.get(source_type, 0.5),
    )


def make_retrieval(claim="Test claim.", evidence_list=None) -> RetrievalResult:
    """Helper to create a minimal RetrievalResult for testing."""
    return RetrievalResult(
        claim=claim,
        evidence=evidence_list or [make_evidence()],
        query_used="test query",
    )


# ------------------------------------------------------------------
# Tests: Reliability Weights
# ------------------------------------------------------------------

class TestReliabilityWeights:

    def test_wikipedia_weight(self, assessor):
        """Wikipedia evidence should get weight 0.80."""
        ev = make_evidence(source_type="wikipedia")
        assert assessor._get_reliability_weight(ev) == pytest.approx(0.80)

    def test_peer_reviewed_weight(self, assessor):
        """Peer-reviewed sources should get weight 1.00."""
        ev = make_evidence(source_type="peer_reviewed")
        assert assessor._get_reliability_weight(ev) == pytest.approx(1.00)

    def test_government_weight(self, assessor):
        """Government sources should get weight 0.90."""
        ev = make_evidence(source_type="government")
        assert assessor._get_reliability_weight(ev) == pytest.approx(0.90)

    def test_news_weight(self, assessor):
        """News sources should get weight 0.60."""
        ev = make_evidence(source_type="news")
        assert assessor._get_reliability_weight(ev) == pytest.approx(0.60)

    def test_unknown_weight_fallback(self, assessor):
        """Unknown source type falls back to URL classification."""
        ev = make_evidence(source_type="unknown_type", url="https://example.com/test")
        assert assessor._get_reliability_weight(ev) == pytest.approx(0.50)


# ------------------------------------------------------------------
# Tests: URL Classification
# ------------------------------------------------------------------

class TestSourceClassification:

    def test_gov_url(self, assessor):
        assert assessor.classify_source("https://www.cdc.gov/health/test") == "government"

    def test_arxiv_url(self, assessor):
        assert assessor.classify_source("https://arxiv.org/abs/2401.00001") == "peer_reviewed"

    def test_wikipedia_url(self, assessor):
        assert assessor.classify_source("https://en.wikipedia.org/wiki/Python") == "wikipedia"

    def test_bbc_url(self, assessor):
        assert assessor.classify_source("https://www.bbc.com/news/science") == "news"

    def test_unknown_url(self, assessor):
        assert assessor.classify_source("https://example.com/random") == "unknown"


# ------------------------------------------------------------------
# Tests: Normalization Formula (K=0.30)
# ------------------------------------------------------------------

class TestNormalizeRelevance:

    def test_zero_relevance(self, assessor):
        """Zero relevance -> 0.0."""
        assert assessor._normalize_relevance(0.0) == pytest.approx(0.0)

    def test_negative_relevance(self, assessor):
        """Negative relevance -> 0.0 (clamped)."""
        assert assessor._normalize_relevance(-1.0) == pytest.approx(0.0)

    def test_small_relevance(self, assessor):
        """Small relevance uses K=0.30 formula."""
        r = 0.05
        assert assessor._normalize_relevance(r) == pytest.approx(expected_norm(r))

    def test_medium_relevance(self, assessor):
        """Medium relevance score."""
        r = 0.30
        # raw=0.30, K=0.30 -> norm = 0.30/0.60 = 0.50
        assert assessor._normalize_relevance(r) == pytest.approx(0.50, rel=1e-4)

    def test_high_relevance(self, assessor):
        """High relevance score."""
        r = 1.0
        assert assessor._normalize_relevance(r) == pytest.approx(expected_norm(r))

    def test_very_high_relevance(self, assessor):
        """Very high relevance approaches 1.0."""
        r = 100.0
        norm = assessor._normalize_relevance(r)
        assert norm > 0.99
        assert norm < 1.0

    def test_always_bounded(self, assessor):
        """Normalized relevance must always be in [0, 1)."""
        for r in [0, 0.01, 0.1, 0.3, 0.5, 1.0, 5.0, 100.0]:
            norm = assessor._normalize_relevance(r)
            assert 0.0 <= norm < 1.0, f"norm({r}) = {norm} out of [0,1)"

    def test_monotone_increasing(self, assessor):
        """Higher raw score -> higher normalized score."""
        scores = [0.0, 0.1, 0.3, 0.5, 1.0, 2.0, 5.0]
        norms  = [assessor._normalize_relevance(r) for r in scores]
        for i in range(len(norms) - 1):
            assert norms[i] <= norms[i+1], (
                f"Not monotone: norm({scores[i]})={norms[i]} > "
                f"norm({scores[i+1]})={norms[i+1]}"
            )


# ------------------------------------------------------------------
# Tests: EQS Computation
# ------------------------------------------------------------------

class TestEQSComputation:

    def test_eqs_is_product(self, assessor):
        """EQS should equal reliability_weight * normalized_relevance."""
        ev = make_evidence(source_type="wikipedia", relevance=1.0)
        rr = make_retrieval(evidence_list=[ev])
        qa = assessor.assess(rr)
        se = qa.scored_evidence[0]
        expected = se.reliability_weight * se.normalized_relevance
        assert se.evidence_quality_score == pytest.approx(expected)

    def test_eqs_bounded(self, assessor):
        """EQS must be in [0, 1]."""
        for source_type in ["wikipedia", "peer_reviewed", "news"]:
            for rel in [0.0, 0.1, 0.5, 1.0, 5.0]:
                ev = make_evidence(source_type=source_type, relevance=rel)
                rr = make_retrieval(evidence_list=[ev])
                qa = assessor.assess(rr)
                eqs = qa.scored_evidence[0].evidence_quality_score if qa.scored_evidence else 0.0
                assert 0.0 <= eqs <= 1.0, f"EQS={eqs} out of [0,1] for {source_type}, rel={rel}"

    def test_best_eqs_is_max(self, assessor):
        """best_eqs should be the maximum EQS among all evidence items."""
        ev1 = make_evidence(source_type="news",      relevance=0.5)
        ev2 = make_evidence(source_type="wikipedia", relevance=2.0)
        rr  = make_retrieval(evidence_list=[ev1, ev2])
        qa  = assessor.assess(rr)
        assert qa.best_eqs == pytest.approx(
            max(se.evidence_quality_score for se in qa.scored_evidence)
        )

    def test_empty_evidence(self, assessor):
        """Empty evidence -> empty scored_evidence and best_eqs=0."""
        rr = RetrievalResult(claim="Test claim.", evidence=[], query_used="test")
        qa = assessor.assess(rr)
        assert qa.scored_evidence == []
        assert qa.best_eqs == pytest.approx(0.0)

    def test_sorted_descending(self, assessor):
        """scored_evidence should be sorted by EQS descending."""
        ev1 = make_evidence(source_type="news",       relevance=0.1)
        ev2 = make_evidence(source_type="wikipedia",  relevance=3.0)
        ev3 = make_evidence(source_type="government", relevance=2.0)
        rr  = make_retrieval(evidence_list=[ev1, ev2, ev3])
        qa  = assessor.assess(rr)
        scores = [se.evidence_quality_score for se in qa.scored_evidence]
        assert scores == sorted(scores, reverse=True)

    def test_css_reachable(self, assessor):
        """
        Verify that CSS = entailment_prob * EQS can reach the SUPPORTED
        threshold (0.75) with realistic evidence.

        With a strong source (peer_reviewed, weight=1.0) and high
        relevance (raw=2.0, norm ~= 0.87), EQS ~= 0.87.
        An entailment_prob of ~0.87 would then give CSS ~= 0.75.
        This proves the SUPPORTED threshold is mathematically reachable.
        """
        ev = make_evidence(source_type="peer_reviewed", relevance=2.0)
        rr = make_retrieval(evidence_list=[ev])
        qa = assessor.assess(rr)
        eqs = qa.scored_evidence[0].evidence_quality_score
        # With entailment_prob=0.87, CSS would be ~= 0.87 * 0.87 = 0.756
        # which exceeds the default css_supported=0.75 threshold.
        simulated_css = 0.87 * eqs
        config = FrameworkConfig()
        assert simulated_css >= config.css_supported, (
            f"SUPPORTED threshold {config.css_supported} not reachable: "
            f"max simulated CSS = {simulated_css:.3f} with EQS = {eqs:.3f}"
        )
