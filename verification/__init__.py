# verification/__init__.py
"""
verification package
======================
Phase 3 & 4 of the Hallucination Correction Framework.

Exports:
- EvidenceQualityAssessor  → Phase 3: compute Evidence Quality Scores (pure Python)
- ScoredEvidence           → Data class: evidence with quality score
- QualityAssessedRetrieval → Data class: all scored evidence for a claim
- NLIVerifier              → Phase 4: NLI-based claim verification (requires torch)
- ClaimVerificationResult  → Data class: verification outcome
- VerificationLabel        → Enum: SUPPORTED / INSUFFICIENT / HALLUCINATED

Lazy-loaded to avoid forcing torch/transformers unless NLIVerifier is used.
"""

__all__ = [
    "EvidenceQualityAssessor",
    "ScoredEvidence",
    "QualityAssessedRetrieval",
    "NLIVerifier",
    "ClaimVerificationResult",
    "VerificationLabel",
    "NLIResult",
]


def __getattr__(name):
    if name in ("EvidenceQualityAssessor", "ScoredEvidence", "QualityAssessedRetrieval"):
        import importlib
        mod = importlib.import_module("verification.evidence_quality")
        return getattr(mod, name)
    if name in ("NLIVerifier", "ClaimVerificationResult", "VerificationLabel", "NLIResult"):
        import importlib
        mod = importlib.import_module("verification.nli_verifier")
        return getattr(mod, name)
    raise AttributeError(f"module 'verification' has no attribute {name!r}")
