"""
tests/test_pipeline.py
========================
End-to-End and Phase-7 Pipeline Tests
---------------------------------------
Verifies:
1. Mixed response containing supported + contradicted + unverifiable claims.
2. Selective correction: only contradicted claims are modified.
3. Supported and unverifiable claims are preserved/flagged without fabrication.
4. Final reconstruction preserves original response structure.
5. Phase 7 genuinely performs fresh retrieval and handles retrieval failure
   by setting FINAL_VERIFICATION_FAILED (no silent fallback).
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from unittest.mock import MagicMock, patch

from config import FrameworkConfig
from retrieval.claim_extractor import ClaimSpan
from retrieval.models import EvidenceItem, RetrievalResult
from verification.evidence_quality import ScoredEvidence, QualityAssessedRetrieval
from verification.nli_verifier import ClaimVerificationResult, VerificationLabel
from correction.claim_corrector import CorrectedClaim, CorrectionStatus
from hallucination_pipeline import HallucinationCorrectionPipeline, PipelineResult


class TestPipelinePhase7AndMixedResponse:

    def test_mixed_response_selective_correction(self):
        """
        Verify that in a mixed response:
        - Claim 1 (SUPPORTED) is preserved
        - Claim 2 (CONTRADICTED) is corrected
        - Claim 3 (UNVERIFIABLE) is flagged and not fabricated
        - Final reconstruction preserves the structure of unchanged portions
        """
        config = FrameworkConfig()
        pipeline = HallucinationCorrectionPipeline.__new__(HallucinationCorrectionPipeline)
        pipeline.config = config

        orig_response = (
            "Einstein was born in 1879. "
            "He was born in France. "
            "He secretly loved ice cream."
        )

        spans = [
            ClaimSpan(claim_id=0, text="Einstein was born in 1879.", start_char=0, end_char=26, original_text="Einstein was born in 1879."),
            ClaimSpan(claim_id=1, text="He was born in France.", start_char=27, end_char=49, original_text="He was born in France."),
            ClaimSpan(claim_id=2, text="He secretly loved ice cream.", start_char=50, end_char=78, original_text="He secretly loved ice cream."),
        ]

        # Mock claim extractor
        pipeline.claim_extractor = MagicMock()
        pipeline.claim_extractor.extract_spans.return_value = spans

        # Mock retriever & assessor
        pipeline.evidence_retriever = MagicMock()
        pipeline.quality_assessor = MagicMock()

        # Mock NLI Verifier (Phase 4):
        # Claim 0 -> SUPPORTED
        # Claim 1 -> CONTRADICTED
        # Claim 2 -> UNVERIFIABLE
        vr0 = ClaimVerificationResult(claim=spans[0].text, label=VerificationLabel.SUPPORTED, css=0.85)
        vr1 = ClaimVerificationResult(claim=spans[1].text, label=VerificationLabel.CONTRADICTED, css=0.10)
        vr2 = ClaimVerificationResult(claim=spans[2].text, label=VerificationLabel.UNVERIFIABLE, css=0.20)

        pipeline.nli_verifier = MagicMock()
        pipeline.nli_verifier.verify_batch.return_value = [vr0, vr1, vr2]

        # Mock Claim Corrector (Phase 5):
        # Claim 0 -> PRESERVED
        # Claim 1 -> CORRECTED to "He was born in Germany."
        # Claim 2 -> FLAGGED
        cc0 = CorrectedClaim(original_claim=spans[0].text, corrected_claim=spans[0].text, status=CorrectionStatus.PRESERVED, verification_result=vr0)
        cc1 = CorrectedClaim(original_claim=spans[1].text, corrected_claim="He was born in Germany.", status=CorrectionStatus.CORRECTED, verification_result=vr1)
        cc2 = CorrectedClaim(original_claim=spans[2].text, corrected_claim=spans[2].text, status=CorrectionStatus.FLAGGED, verification_result=vr2)

        pipeline.claim_corrector = MagicMock()
        pipeline.claim_corrector.correct_batch.return_value = [cc0, cc1, cc2]

        # Real ResponseReconstructor
        from correction.response_reconstructor import ResponseReconstructor
        pipeline.reconstructor = ResponseReconstructor()

        # Phase 7 fresh verification: fresh retrieval succeeds and verifies the correction
        fresh_vr1 = ClaimVerificationResult(claim="He was born in Germany.", label=VerificationLabel.SUPPORTED, css=0.88)
        fresh_qa = MagicMock()
        fresh_qa.scored_evidence = [MagicMock()]
        pipeline.evidence_retriever.retrieve.return_value = MagicMock()
        pipeline.quality_assessor.assess.return_value = fresh_qa
        pipeline.nli_verifier.verify.return_value = fresh_vr1

        # Run pipeline
        result = pipeline.run(orig_response, verbose=False)

        # Assertions
        assert len(result.corrected_claims) == 3
        assert result.corrected_claims[0].status == CorrectionStatus.PRESERVED
        assert result.corrected_claims[1].status == CorrectionStatus.CORRECTED
        assert result.corrected_claims[2].status == CorrectionStatus.FLAGGED

        # Check final text reconstruction
        expected_final = "Einstein was born in 1879. He was born in Germany. He secretly loved ice cream."
        assert result.final_response == expected_final
        assert "Einstein was born in 1879." in result.final_response
        assert "He secretly loved ice cream." in result.final_response
        assert "France" not in result.final_response

    def test_phase7_fresh_retrieval_failure_marks_final_verification_failed(self):
        """
        When fresh retrieval in Phase 7 fails to find evidence for the corrected claim,
        the status must be marked FINAL_VERIFICATION_FAILED rather than falling back silently.
        """
        config = FrameworkConfig()
        pipeline = HallucinationCorrectionPipeline.__new__(HallucinationCorrectionPipeline)
        pipeline.config = config
        pipeline.evidence_retriever = MagicMock()
        pipeline.quality_assessor = MagicMock()
        pipeline.nli_verifier = MagicMock()
        pipeline.claim_corrector = MagicMock()

        cc = CorrectedClaim(
            original_claim="Einstein was born in France.",
            corrected_claim="Einstein was born in Germany.",
            status=CorrectionStatus.CORRECTED,
            verification_result=ClaimVerificationResult(
                claim="Einstein was born in France.",
                label=VerificationLabel.CONTRADICTED,
                css=0.1,
            ),
        )

        # Simulate fresh retrieval returning NO evidence
        empty_qa = MagicMock()
        empty_qa.scored_evidence = []
        pipeline.evidence_retriever.retrieve.return_value = MagicMock()
        pipeline.quality_assessor.assess.return_value = empty_qa

        final_vr, updated_claims = pipeline._independent_verification([cc], qa_results=[MagicMock()])

        assert len(updated_claims) == 1
        assert updated_claims[0].status == CorrectionStatus.FINAL_VERIFICATION_FAILED
        assert final_vr[0].label == VerificationLabel.UNVERIFIABLE
