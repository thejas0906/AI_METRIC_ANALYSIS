"""
tests/test_evidence_quality.py
================================
Unit Tests for the Evidence Quality Assessment Module (Phase 3)
---------------------------------------------------------------
Tests reliability weight lookup, EQS computation, source
classification, and edge case handling.

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


# ──────────────────────────────────────────────────────────────────
# Fixtures
# ──────────────────────────────────────────────────────────────────

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


# ──────────────────────────────────────────────────────────────────
# Tests
# ──────────────────────────────────────────────────────────────────

class TestEvidenceQualityAssessor:

    def test_wikipedia_weight(self, assessor):
        """Wikipedia evidence should get weight 0.80."""
        ev = make_evidence(source_type="wikipedia")
        weight = assessor._get_reliability_weight(ev)
        assert weight == pytest.approx(0.80)

    def test_peer_reviewed_weight(self, assessor):
        """Peer-reviewed sources should get weight 1.00."""
        ev = make_evidence(source_type="peer_reviewed")
        weight = assessor._get_reliability_weight(ev)
        assert weight == pytest.approx(1.00)

    def test_government_weight(self, assessor):
        """Government sources should get weight 0.90."""
        ev = make_evidence(source_type="government")
        weight = assessor._get_reliability_weight(ev)
        assert weight == pytest.approx(0.90)

    def test_news_weight(self, assessor):
        """News sources should get weight 0.60."""
        ev = make_evidence(source_type="news")
        weight = assessor._get_reliability_weight(ev)
        assert weight == pytest.approx(0.60)

    def test_unknown_weight_fallback(self, assessor):
        """Unknown source type should fall back to URL classification."""
        ev = make_evidence(source_type="unknown_type", url="https://example.com/test")
        weight = assessor._get_reliability_weight(ev)
        # Falls back to URL classification → "unknown" → 0.50
        assert weight == pytest.approx(0.50)

    def test_gov_url_classification(self, assessor):
        """URLs with '.gov' should be classified as 'government'."""
        source_type = assessor.classify_source("https://www.cdc.gov/health/test")
        assert source_type == "government"

    def test_arxiv_url_classification(self, assessor):
        """arXiv URLs should be classified as 'peer_reviewed'."""
        source_type = assessor.classify_source("https://arxiv.org/abs/2401.00001")
        assert source_type == "peer_reviewed"

    def test_wikipedia_url_classification(self, assessor):
        """Wikipedia URLs should be classified as 'wikipedia'."""
        source_type = assessor.classify_source("https://en.wikipedia.org/wiki/Python")
        assert source_type == "wikipedia"

    def test_news_url_classification(self, assessor):
        """BBC URLs should be classified as 'news'."""
        source_type = assessor.classify_source("https://www.bbc.com/news/science")
        assert source_type == "news"

    def test_normalize_relevance_zero(self, assessor):
        """Zero relevance should normalize to 0.0."""
        assert assessor._normalize_relevance(0.0) == pytest.approx(0.0)

    def test_normalize_relevance_positive(self, assessor):
        """Positive relevance should normalize to (r / r+1)."""
        r = 2.0
        expected = r / (r + 1.0)
        assert assessor._normalize_relevance(r) == pytest.approx(expected)

    def test_normalize_relevance_bounded(self, assessor):
        """Normalized relevance should always be in [0, 1)."""
        for r in [0, 0.5, 1, 5, 100]:
            norm = assessor._normalize_relevance(r)
            assert 0.0 <= norm < 1.0

    def test_eqs_is_product(self, assessor):
        """EQS should equal reliability_weight * normalized_relevance."""
        ev = make_evidence(source_type="wikipedia", relevance=1.0)
        rr = make_retrieval(evidence_list=[ev])
        qa = assessor.assess(rr)
        se = qa.scored_evidence[0]
        expected_eqs = se.reliability_weight * se.normalized_relevance
        assert se.evidence_quality_score == pytest.approx(expected_eqs)

    def test_best_eqs_is_max(self, assessor):
        """best_eqs should be the maximum EQS among all evidence items."""
        ev1 = make_evidence(source_type="news",      relevance=0.5)
        ev2 = make_evidence(source_type="wikipedia", relevance=2.0)
        rr  = make_retrieval(evidence_list=[ev1, ev2])
        qa  = assessor.assess(rr)
        assert qa.best_eqs == pytest.approx(
            max(se.evidence_quality_score for se in qa.scored_evidence)
        )

    def test_empty_evidence_list(self, assessor):
        """Empty evidence should produce empty scored_evidence and best_eqs=0."""
        rr = RetrievalResult(claim="Test claim.", evidence=[], query_used="test")
        qa = assessor.assess(rr)
        assert qa.scored_evidence == []
        assert qa.best_eqs == pytest.approx(0.0)

    def test_sorted_by_eqs_descending(self, assessor):
        """scored_evidence should be sorted by EQS descending."""
        ev1 = make_evidence(source_type="news",      relevance=0.1)
        ev2 = make_evidence(source_type="wikipedia", relevance=3.0)
        ev3 = make_evidence(source_type="government", relevance=2.0)
        rr  = make_retrieval(evidence_list=[ev1, ev2, ev3])
        qa  = assessor.assess(rr)
        scores = [se.evidence_quality_score for se in qa.scored_evidence]
        assert scores == sorted(scores, reverse=True)
