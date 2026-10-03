"""
tests/test_response_reconstructor.py
======================================
Unit Tests for ResponseReconstructor (Phase 6)
-----------------------------------------------
Verifies:
1. Only corrected claim spans are modified.
2. Supported and unverifiable text remains byte-for-byte identical.
3. Multi-paragraph structure, indentation, and transitions are preserved.
4. Duplicate claim occurrences in the same document are handled correctly
   via exact character offsets (e.g. only modifying the second occurrence).
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from retrieval.claim_extractor import ClaimSpan
from correction.claim_corrector import CorrectedClaim, CorrectionStatus
from verification.nli_verifier import ClaimVerificationResult, VerificationLabel
from correction.response_reconstructor import ResponseReconstructor


def make_dummy_vr(claim: str, label=VerificationLabel.SUPPORTED, css=1.0):
    return ClaimVerificationResult(claim=claim, label=label, css=css)


class TestResponseReconstructor:

    def test_only_corrected_span_changes(self):
        """Only the contradicted claim span should change; others must be identical."""
        orig = "Einstein was born in 1879. He died in France. He won the Nobel Prize."
        span0 = ClaimSpan(claim_id=0, text="Einstein was born in 1879.", start_char=0, end_char=26, original_text="Einstein was born in 1879.")
        span1 = ClaimSpan(claim_id=1, text="He died in France.", start_char=27, end_char=45, original_text="He died in France.")
        span2 = ClaimSpan(claim_id=2, text="He won the Nobel Prize.", start_char=46, end_char=69, original_text="He won the Nobel Prize.")

        cc0 = CorrectedClaim(
            original_claim=span0.text,
            corrected_claim=span0.text,
            status=CorrectionStatus.PRESERVED,
            verification_result=make_dummy_vr(span0.text, VerificationLabel.SUPPORTED),
        )
        cc1 = CorrectedClaim(
            original_claim=span1.text,
            corrected_claim="He died in Princeton, New Jersey.",
            status=CorrectionStatus.CORRECTED,
            verification_result=make_dummy_vr(span1.text, VerificationLabel.CONTRADICTED),
        )
        cc2 = CorrectedClaim(
            original_claim=span2.text,
            corrected_claim=span2.text,
            status=CorrectionStatus.PRESERVED,
            verification_result=make_dummy_vr(span2.text, VerificationLabel.SUPPORTED),
        )

        reconstructor = ResponseReconstructor()
        result = reconstructor.reconstruct(
            corrected_claims=[cc0, cc1, cc2],
            claim_spans=[span0, span1, span2],
            original_response=orig,
        )

        expected = "Einstein was born in 1879. He died in Princeton, New Jersey. He won the Nobel Prize."
        assert result.final_text == expected
        assert "Einstein was born in 1879." in result.final_text
        assert "He won the Nobel Prize." in result.final_text
        assert "France" not in result.final_text

    def test_supported_text_stays_identical(self):
        """When all claims are supported, the reconstructed response matches original byte-for-byte."""
        orig = "Water boils at 100 degrees Celsius under standard atmospheric pressure. Ice melts at 0 degrees."
        span0 = ClaimSpan(claim_id=0, text="Water boils at 100 degrees Celsius under standard atmospheric pressure.", start_char=0, end_char=71, original_text="Water boils at 100 degrees Celsius under standard atmospheric pressure.")
        span1 = ClaimSpan(claim_id=1, text="Ice melts at 0 degrees.", start_char=72, end_char=95, original_text="Ice melts at 0 degrees.")

        cc0 = CorrectedClaim(
            original_claim=span0.text,
            corrected_claim=span0.text,
            status=CorrectionStatus.PRESERVED,
            verification_result=make_dummy_vr(span0.text, VerificationLabel.SUPPORTED),
        )
        cc1 = CorrectedClaim(
            original_claim=span1.text,
            corrected_claim=span1.text,
            status=CorrectionStatus.PRESERVED,
            verification_result=make_dummy_vr(span1.text, VerificationLabel.SUPPORTED),
        )

        reconstructor = ResponseReconstructor()
        result = reconstructor.reconstruct(
            corrected_claims=[cc0, cc1],
            claim_spans=[span0, span1],
            original_response=orig,
        )
        assert result.final_text == orig

    def test_paragraph_structure_stays_intact(self):
        """Reconstruction must preserve multi-paragraph layout and newlines."""
        orig = "First paragraph starts here.\n\nSecond paragraph has a mistake in 1900.\n\nThird paragraph ends here."
        idx1 = orig.index("Second paragraph has a mistake in 1900.")
        len1 = len("Second paragraph has a mistake in 1900.")
        span1 = ClaimSpan(
            claim_id=1,
            text="Second paragraph has a mistake in 1900.",
            start_char=idx1,
            end_char=idx1 + len1,
            original_text="Second paragraph has a mistake in 1900.",
        )
        cc1 = CorrectedClaim(
            original_claim=span1.text,
            corrected_claim="Second paragraph has been corrected to 1920.",
            status=CorrectionStatus.CORRECTED,
            verification_result=make_dummy_vr(span1.text, VerificationLabel.CONTRADICTED),
        )

        reconstructor = ResponseReconstructor()
        result = reconstructor.reconstruct(
            corrected_claims=[cc1],
            claim_spans=[span1],
            original_response=orig,
        )

        expected = "First paragraph starts here.\n\nSecond paragraph has been corrected to 1920.\n\nThird paragraph ends here."
        assert result.final_text == expected
        assert "\n\n" in result.final_text
        assert result.final_text.startswith("First paragraph starts here.\n\n")
        assert result.final_text.endswith("\n\nThird paragraph ends here.")

    def test_duplicate_claim_occurrences_work_correctly(self):
        """
        When two identical sentences appear in the text, modifying only the
        second occurrence must leave the first occurrence untouched.
        """
        orig = "The company was founded in 1990. The company was founded in 1990."
        span0 = ClaimSpan(
            claim_id=0,
            text="The company was founded in 1990.",
            start_char=0,
            end_char=32,
            original_text="The company was founded in 1990.",
        )
        span1 = ClaimSpan(
            claim_id=1,
            text="The company was founded in 1990.",
            start_char=33,
            end_char=65,
            original_text="The company was founded in 1990.",
        )

        cc0 = CorrectedClaim(
            original_claim=span0.text,
            corrected_claim=span0.text,
            status=CorrectionStatus.PRESERVED,
            verification_result=make_dummy_vr(span0.text, VerificationLabel.SUPPORTED),
        )
        cc1 = CorrectedClaim(
            original_claim=span1.text,
            corrected_claim="The company was founded in 2005.",
            status=CorrectionStatus.CORRECTED,
            verification_result=make_dummy_vr(span1.text, VerificationLabel.CONTRADICTED),
        )

        reconstructor = ResponseReconstructor()
        result = reconstructor.reconstruct(
            corrected_claims=[cc0, cc1],
            claim_spans=[span0, span1],
            original_response=orig,
        )

        expected = "The company was founded in 1990. The company was founded in 2005."
        assert result.final_text == expected
        assert result.final_text.startswith("The company was founded in 1990.")
        assert result.final_text.endswith("The company was founded in 2005.")
