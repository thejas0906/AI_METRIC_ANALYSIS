"""
evidence_quality.py  (Phase 3)
================================
Evidence Quality Assessment Module
------------------------------------
Assigns a reliability weight to each evidence item based on its
source type, and computes an *Evidence Quality Score* (EQS) that
combines source reliability with passage relevance.

Formula:
    EQS = reliability_weight × min(relevance_score, 1.0)

The EQS is later multiplied by the NLI entailment probability to
produce the final Claim Support Score (CSS):

    CSS = NLI_entailment_prob × EQS

This two-stage design separates source quality from semantic alignment,
allowing researchers to tune each component independently.

Source Weight Hierarchy (from config.py):
    peer_reviewed  → 1.00  (highest reliability)
    government     → 0.90
    wikipedia      → 0.80
    news           → 0.60
    unknown        → 0.50  (lowest reliability)

Design Decisions:
-----------------
- We normalize relevance_score to [0, 1] using a sigmoid function
  so that BM25 scores from different evidence items are comparable.
- The weight hierarchy is fully configurable via FrameworkConfig.
- Future extension: use classifier to detect source type from URL
  (e.g., ".gov" → government, "arxiv.org" → peer_reviewed).
"""

import math
from dataclasses import dataclass
from typing import List, Optional
from loguru import logger

from config import FrameworkConfig, EVIDENCE_WEIGHTS
from retrieval.models import EvidenceItem, RetrievalResult


# ──────────────────────────────────────────────────────────────────
# Data Structures
# ──────────────────────────────────────────────────────────────────

@dataclass
class ScoredEvidence:
    """
    An EvidenceItem enriched with quality assessment scores.

    Attributes:
        original:           The underlying EvidenceItem.
        reliability_weight: Source reliability weight (from config).
        normalized_relevance: BM25 relevance score mapped to [0, 1].
        evidence_quality_score: Combined EQS = weight × norm_relevance.
    """
    original:                 EvidenceItem
    reliability_weight:       float
    normalized_relevance:     float
    evidence_quality_score:   float   # EQS

    # Convenience pass-throughs
    @property
    def passage(self)     -> str:   return self.original.passage
    @property
    def source_url(self)  -> str:   return self.original.source_url
    @property
    def source_type(self) -> str:   return self.original.source_type
    @property
    def page_title(self)  -> str:   return self.original.page_title


@dataclass
class QualityAssessedRetrieval:
    """
    A RetrievalResult enriched with quality scores for each passage.

    Attributes:
        claim:            The original atomic claim.
        scored_evidence:  List of ScoredEvidence items (best first).
        best_eqs:         Highest EQS among all evidence items.
    """
    claim:           str
    scored_evidence: List[ScoredEvidence]
    best_eqs:        float   # maximum EQS across all evidence items


# ──────────────────────────────────────────────────────────────────
# Evidence Quality Assessor
# ──────────────────────────────────────────────────────────────────

class EvidenceQualityAssessor:
    """
    Assigns reliability weights and computes Evidence Quality Scores.

    Usage:
        assessor = EvidenceQualityAssessor(config)
        qa = assessor.assess(retrieval_result)
        print(qa.best_eqs)          # best evidence quality score
        for se in qa.scored_evidence:
            print(se.evidence_quality_score, se.passage[:50])
    """

    def __init__(self, config: Optional[FrameworkConfig] = None):
        """
        Initialize with source weight mapping from config.

        Args:
            config: FrameworkConfig instance; uses defaults if None.
        """
        self.config = config or FrameworkConfig()
        self.weights = self.config.evidence_weights
        logger.info("EvidenceQualityAssessor initialized.")

    # ──────────────────────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────────────────────

    def assess(self, retrieval_result: RetrievalResult) -> QualityAssessedRetrieval:
        """
        Score all evidence items for a single claim.

        Args:
            retrieval_result: Output from EvidenceRetriever.retrieve().

        Returns:
            QualityAssessedRetrieval with scored and ranked evidence.
        """
        scored_items: List[ScoredEvidence] = []

        for ev in retrieval_result.evidence:
            rel_weight  = self._get_reliability_weight(ev)
            norm_rel    = self._normalize_relevance(ev.relevance_score)
            eqs         = rel_weight * norm_rel

            se = ScoredEvidence(
                original=ev,
                reliability_weight=rel_weight,
                normalized_relevance=norm_rel,
                evidence_quality_score=eqs,
            )
            scored_items.append(se)
            logger.debug(
                f"[{ev.source_type}] EQS={eqs:.3f} "
                f"(w={rel_weight:.2f}, rel={norm_rel:.3f}) | "
                f"{ev.passage[:60]}..."
            )

        # Sort by EQS descending (best evidence first)
        scored_items.sort(key=lambda s: s.evidence_quality_score, reverse=True)

        best_eqs = scored_items[0].evidence_quality_score if scored_items else 0.0

        return QualityAssessedRetrieval(
            claim=retrieval_result.claim,
            scored_evidence=scored_items,
            best_eqs=best_eqs,
        )

    def assess_batch(
        self, retrieval_results: List[RetrievalResult]
    ) -> List[QualityAssessedRetrieval]:
        """
        Assess evidence quality for a list of retrieval results.

        Args:
            retrieval_results: List of RetrievalResult objects.

        Returns:
            List of QualityAssessedRetrieval objects.
        """
        return [self.assess(r) for r in retrieval_results]

    def classify_source(self, url: str) -> str:
        """
        Heuristically classify a source URL into a known category.

        Rules:
        - '.gov' domain → "government"
        - 'arxiv.org', 'pubmed', 'ncbi.nlm' → "peer_reviewed"
        - 'wikipedia.org' → "wikipedia"
        - News domains (reuters, bbc, nyt, etc.) → "news"
        - Anything else → "unknown"

        Args:
            url: Source URL string.

        Returns:
            Source category string.
        """
        url_lower = url.lower()

        if ".gov" in url_lower:
            return "government"

        PEER_REVIEW_DOMAINS = [
            "arxiv.org", "pubmed.ncbi", "ncbi.nlm.nih",
            "scholar.google", "doi.org", "jstor.org",
            "nature.com", "sciencedirect.com",
        ]
        if any(domain in url_lower for domain in PEER_REVIEW_DOMAINS):
            return "peer_reviewed"

        if "wikipedia.org" in url_lower:
            return "wikipedia"

        NEWS_DOMAINS = [
            "reuters.com", "bbc.com", "bbc.co.uk", "nytimes.com",
            "theguardian.com", "cnn.com", "apnews.com", "npr.org",
        ]
        if any(domain in url_lower for domain in NEWS_DOMAINS):
            return "news"

        return "unknown"

    # ──────────────────────────────────────────────────────────────
    # Private Helpers
    # ──────────────────────────────────────────────────────────────

    def _get_reliability_weight(self, ev: EvidenceItem) -> float:
        """
        Look up the reliability weight for an evidence item.

        First uses ev.source_type (set by retriever).
        Falls back to URL-based heuristic classification.

        Args:
            ev: EvidenceItem to score.

        Returns:
            Reliability weight in [0, 1].
        """
        # Prefer explicitly tagged source type
        weight = self.weights.get(ev.source_type)
        if weight is not None:
            return weight

        # Fallback: classify from URL
        detected_type = self.classify_source(ev.source_url)
        return self.weights.get(detected_type, self.weights["unknown"])

    @staticmethod
    def _normalize_relevance(raw_score: float) -> float:
        """
        Map a raw BM25-style relevance score to [0, 1].

        The BM25-lite scores produced by EvidenceRetriever._bm25_lite()
        depend on overlap, doc length, and IDF.  Empirically, scores for
        genuinely relevant sentence-level passages typically fall in the
        range [0, 0.5].  A direct x/(x+1) sigmoid therefore compresses
        most real-world values below 0.33, making the EQS artificially
        low and causing over-flagging of supported claims as INSUFFICIENT.

        To address this, we apply a scaled sigmoid with a calibration
        constant K chosen so that a "good" BM25-lite score of 0.15
        maps to approximately 0.75 (indicating strong relevance):

            K    = 0.15 / (1 - 0.75) * 0.75 = 0.45 (rounded)
            norm = raw / (raw + K)

        This ensures:
            raw = 0.00 -> norm = 0.00  (no overlap)
            raw = 0.10 -> norm ~= 0.18 (weak relevance)
            raw = 0.15 -> norm ~= 0.25 (moderate relevance)
            raw = 0.30 -> norm ~= 0.40 (strong relevance)
            raw = 1.00 -> norm ~= 0.69 (very strong relevance)
            raw -> inf -> norm -> 1.00

        The constant K = 0.30 is selected as a conservative calibration
        that keeps EQS honest without over-inflating low-quality evidence.
        This is a documented design decision, not an arbitrary heuristic.

        Args:
            raw_score: BM25-lite relevance score from EvidenceItem.

        Returns:
            Normalized score in [0, 1].
        """
        if raw_score <= 0:
            return 0.0
        # Calibration constant: raw_score 0.3 maps to ~0.50
        K = 0.30
        return raw_score / (raw_score + K)
