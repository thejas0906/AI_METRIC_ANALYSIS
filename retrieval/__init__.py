# retrieval/__init__.py
"""
retrieval package
==================
Phase 1 & 2 of the Hallucination Correction Framework.

Exports:
- ClaimExtractor       → Phase 1: extract atomic claims from LLM output (requires spaCy)
- EvidenceRetriever    → Phase 2: retrieve Wikipedia evidence per claim
- EvidenceItem         → Data class for a single evidence passage
- RetrievalResult      → Data class for all evidence for one claim

Lazy-loaded to avoid forcing spaCy / wikipedia-api installation
unless those classes are explicitly used.
"""

__all__ = [
    "ClaimExtractor",
    "EvidenceRetriever",
    "EvidenceItem",
    "RetrievalResult",
]


def __getattr__(name):
    if name == "ClaimExtractor":
        from retrieval.claim_extractor import ClaimExtractor
        return ClaimExtractor
    if name == "EvidenceRetriever":
        from retrieval.evidence_retriever import EvidenceRetriever
        return EvidenceRetriever
    if name == "EvidenceItem":
        from retrieval.evidence_retriever import EvidenceItem
        return EvidenceItem
    if name == "RetrievalResult":
        from retrieval.evidence_retriever import RetrievalResult
        return RetrievalResult
    raise AttributeError(f"module 'retrieval' has no attribute {name!r}")
