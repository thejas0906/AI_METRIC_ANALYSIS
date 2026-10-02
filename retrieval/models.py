"""
retrieval/models.py
====================
Pure-Python data classes for the retrieval package.
No external dependencies — safe to import in tests without spaCy or wikipedia-api.
"""

from dataclasses import dataclass, field
from typing import List


@dataclass
class EvidenceItem:
    """
    Represents a single piece of evidence retrieved for a claim.

    Attributes:
        passage:            The text passage used as evidence.
        source_url:         URL of the source page.
        source_type:        Category string (e.g., "wikipedia", "news").
        page_title:         Title of the Wikipedia page (or article).
        relevance_score:    BM25-style relevance score vs. the claim.
        reliability_weight: Source reliability weight from config.
    """
    passage:            str
    source_url:         str
    source_type:        str   = "wikipedia"
    page_title:         str   = ""
    relevance_score:    float = 0.0
    reliability_weight: float = 0.8   # default: Wikipedia weight


@dataclass
class RetrievalResult:
    """
    Container for all evidence retrieved for a single claim.

    Attributes:
        claim:      The original atomic claim string.
        evidence:   List of EvidenceItem objects (sorted by relevance).
        query_used: The search query sent to the API.
    """
    claim:      str
    evidence:   List[EvidenceItem] = field(default_factory=list)
    query_used: str = ""
