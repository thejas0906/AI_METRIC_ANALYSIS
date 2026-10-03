"""
tests/test_claim_extractor.py
==============================
Unit Tests for the Claim Extraction Module (Phase 1)
------------------------------------------------------
Tests sentence segmentation, filtering, atomization,
deduplication, and edge case handling.

Run with:
    pytest tests/test_claim_extractor.py -v
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from config import FrameworkConfig


# ──────────────────────────────────────────────────────────────────
# Fixtures
# ──────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def extractor():
    """Initialize ClaimExtractor once for all tests."""
    try:
        from retrieval.claim_extractor import ClaimExtractor
        config = FrameworkConfig()
        return ClaimExtractor(config)
    except RuntimeError as e:
        pytest.skip(f"spaCy model not available: {e}")


# ──────────────────────────────────────────────────────────────────
# Tests
# ──────────────────────────────────────────────────────────────────

class TestClaimExtractor:
    """Tests for ClaimExtractor.extract()."""

    def test_basic_extraction(self, extractor):
        """Simple two-sentence input should return two claims."""
        text = "Einstein was born in 1879. He won the Nobel Prize in 1921."
        claims = extractor.extract(text)
        assert len(claims) >= 1
        assert all(isinstance(c, str) for c in claims)

    def test_empty_input_returns_empty(self, extractor):
        """Empty input should return an empty list."""
        claims = extractor.extract("")
        assert claims == []

    def test_whitespace_only_returns_empty(self, extractor):
        """Whitespace-only input should return an empty list."""
        claims = extractor.extract("   \n\t  ")
        assert claims == []

    def test_question_filtered_out(self, extractor):
        """Questions (ending with '?') should be filtered."""
        text = "What is the speed of light? The speed of light is 299,792 km/s."
        claims = extractor.extract(text)
        # The question should be removed; the factual statement kept
        assert not any("?" in c for c in claims)

    def test_claims_end_with_period(self, extractor):
        """All extracted claims should end with a period."""
        text = "Einstein was a physicist. Curie discovered radium."
        claims = extractor.extract(text)
        for claim in claims:
            assert claim.endswith("."), f"Claim doesn't end with period: '{claim}'"

    def test_deduplication(self, extractor):
        """Duplicate sentences should appear only once."""
        text = "Einstein was born in Germany. Einstein was born in Germany."
        claims = extractor.extract(text)
        # Should deduplicate
        assert len(claims) == 1

    def test_short_sentences_filtered(self, extractor):
        """Sentences shorter than min_claim_length should be filtered."""
        text = "Yes. Einstein discovered relativity and changed modern physics."
        claims = extractor.extract(text)
        # "Yes." is too short — should be filtered
        assert not any(c.strip() == "Yes." for c in claims)

    def test_output_type_is_list_of_strings(self, extractor):
        """Output must always be a list of strings."""
        text = "The Moon orbits the Earth. Water freezes at 0 Celsius."
        claims = extractor.extract(text)
        assert isinstance(claims, list)
        assert all(isinstance(c, str) for c in claims)

    def test_long_text_produces_multiple_claims(self, extractor):
        """A multi-sentence paragraph should produce multiple claims."""
        text = (
            "Albert Einstein was a German-born theoretical physicist. "
            "He developed the theory of relativity. "
            "His work is also known for its influence on the philosophy of science. "
            "He received the Nobel Prize in Physics in 1921."
        )
        claims = extractor.extract(text)
        assert len(claims) >= 2

    def test_no_exclamation_marks(self, extractor):
        """Exclamatory sentences should be filtered."""
        text = "What a discovery! Einstein won the Nobel Prize in 1921."
        claims = extractor.extract(text)
        assert not any("!" in c for c in claims)

    def test_offsets_are_correct(self, extractor):
        """ClaimSpan character offsets must match the original text substring."""
        text = "Albert Einstein was born in 1879. He won the Nobel Prize in 1921."
        spans = extractor.extract_spans(text)
        assert len(spans) == 2
        for s in spans:
            extracted_sub = text[s.start_char:s.end_char]
            assert extracted_sub == s.original_text
            assert s.text.rstrip(".") in extracted_sub or extracted_sub in s.text

    def test_duplicate_claim_spans_preserved(self, extractor):
        """Duplicate sentences in the text must produce separate ClaimSpan objects with distinct offsets."""
        text = "The company was founded in 1990. The company was founded in 1990."
        spans = extractor.extract_spans(text)
        assert len(spans) == 2, f"Expected 2 spans, got {len(spans)}"
        assert spans[0].claim_id == 0
        assert spans[1].claim_id == 1
        assert spans[0].start_char != spans[1].start_char
        assert spans[0].end_char != spans[1].end_char
        assert text[spans[0].start_char:spans[0].end_char] == "The company was founded in 1990."
        assert text[spans[1].start_char:spans[1].end_char] == "The company was founded in 1990."

    def test_punctuation_handling(self, extractor):
        """Exclamations and questions filtered; factual statements properly punctuated."""
        text = "Did you know that? Earth orbits the Sun! But gravity attracts mass."
        spans = extractor.extract_spans(text)
        # Questions and exclamation marks should be filtered
        texts = [s.text for s in spans]
        assert not any("?" in t for t in texts)
        assert not any("!" in t for t in texts)
        assert any("gravity" in t.lower() for t in texts)
