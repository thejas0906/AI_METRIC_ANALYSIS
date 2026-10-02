# correction/__init__.py
"""
correction package
===================
Phase 5 & 6 of the Hallucination Correction Framework.

Exports:
- ClaimCorrector         → Phase 5: selectively correct hallucinated claims
- CorrectedClaim         → Data class: result of correction for one claim
- CorrectionStatus       → Enum: PRESERVED / CORRECTED / FLAGGED / FAILED
- ResponseReconstructor  → Phase 6: merge claims into final response
- ReconstructedResponse  → Data class: final response with statistics
"""

from correction.claim_corrector import (
    ClaimCorrector,
    CorrectedClaim,
    CorrectionStatus,
)
from correction.response_reconstructor import (
    ResponseReconstructor,
    ReconstructedResponse,
)

__all__ = [
    "ClaimCorrector",
    "CorrectedClaim",
    "CorrectionStatus",
    "ResponseReconstructor",
    "ReconstructedResponse",
]
