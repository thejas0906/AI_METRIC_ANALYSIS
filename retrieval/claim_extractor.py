"""
claim_extractor.py  (Phase 1)
==============================
Claim Extraction Module
-----------------------
Converts a free-form LLM-generated answer into a list of
atomic factual claims that retain their ORIGINAL CHARACTER OFFSETS
in the source response.

Each extracted claim is represented as a ClaimSpan:
    ClaimSpan(
        claim_id   = int,         # zero-based index
        text       = str,         # the atomic claim text
        start_char = int,         # start offset in original_response
        end_char   = int,         # end offset in original_response
        original_text = str,      # the original sentence (before atomization)
    )

This span information enables the ResponseReconstructor to perform
true span-level surgery on the original response: replacing ONLY the
hallucinated spans and leaving all surrounding text intact.

Pipeline:
  LLM Answer
      |
  Sentence Segmentation (spaCy, preserving char offsets)
      |
  Sentence Filtering  (length, punctuation, question removal)
      |
  Claim Atomization   (coordinate clause splitting)
      |
  ClaimSpan creation (with start_char/end_char)
      |
  List[ClaimSpan]

Design Decisions:
-----------------
- We use spaCy's sentence boundary detector which exposes character
  offsets (sent.start_char, sent.end_char) directly.
- Atomization (coordinate clause splitting) preserves the parent
  sentence's offsets as the span, since the atomic claim is a
  *logical sub-unit* of that sentence span.
- The `extract()` method still returns List[str] for backward
  compatibility with tests that predate this change.
- `extract_spans()` is the primary method used by the pipeline.
"""

import re
import spacy
from dataclasses import dataclass
from typing import List, Optional
from loguru import logger

from config import FrameworkConfig


@dataclass
class ClaimSpan:
    """
    An atomic factual claim with its location in the original response.

    Attributes:
        claim_id:      Zero-based index of this claim in the extraction order.
        text:          The atomic claim text (normalized, ends with '.').
        start_char:    Start character offset in the original LLM response.
        end_char:      End character offset in the original LLM response.
        original_text: The raw sentence text before atomization.
    """
    claim_id:      int
    text:          str
    start_char:    int
    end_char:      int
    original_text: str = ""


# Alias for research specification compliance
Claim = ClaimSpan


class ClaimExtractor:
    """
    Extracts atomic factual claims from LLM-generated text, returning
    ClaimSpan / Claim objects that retain character offsets in the original response.

    Usage (primary):
        extractor = ClaimExtractor(config)
        spans = extractor.extract_spans("Einstein was born in 1879 in Germany.")
        for s in spans:
            print(s.text, s.start_char, s.end_char)

    Usage (legacy / backward compat):
        claims = extractor.extract("Einstein was born in 1879 in Germany.")
        # Returns List[str]
    """

    def __init__(self, config: Optional[FrameworkConfig] = None):
        self.config = config or FrameworkConfig()
        logger.info(f"Loading spaCy model: {self.config.spacy_model}")
        try:
            self.nlp = spacy.load(self.config.spacy_model)
        except OSError:
            raise RuntimeError(
                f"spaCy model '{self.config.spacy_model}' not found.\n"
                f"Install it with: python -m spacy download {self.config.spacy_model}"
            )
        logger.info("ClaimExtractor initialized successfully.")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def extract_spans(self, llm_answer: str) -> List[ClaimSpan]:
        """
        Main entry point. Extract atomic factual claims WITH character
        offsets into the original response text.

        Preserves multiple occurrences of duplicate claim text so that each
        occurrence maintains its distinct character span and unique claim_id.

        Args:
            llm_answer: Raw text output from the LLM.

        Returns:
            List of ClaimSpan objects with start_char/end_char offsets.
        """
        if not llm_answer or not llm_answer.strip():
            logger.warning("Empty LLM answer received -- no claims extracted.")
            return []

        logger.debug(f"Extracting claim spans from answer ({len(llm_answer)} chars).")

        doc = self.nlp(llm_answer)
        claim_spans: List[ClaimSpan] = []
        claim_id = 0

        for sent in doc.sents:
            sent_text = sent.text.strip()
            if not sent_text:
                continue

            # Skip non-factual sentences (questions, exclamations, transitions)
            if not self._is_factual(sent_text):
                continue

            # Character offsets in the original string
            sent_start = sent.start_char
            sent_end   = sent.end_char

            # Atomize the sentence into sub-claims
            atomic_texts = self._atomize(sent_text)

            for atomic in atomic_texts:
                if len(atomic) < self.config.min_claim_length:
                    continue

                # Locate sub-clause offset within sentence if possible
                clean_clause = atomic.rstrip(".")
                sub_offset = sent_text.find(clean_clause)
                if sub_offset != -1 and len(atomic_texts) > 1:
                    span_start = sent_start + sub_offset
                    span_end   = span_start + len(clean_clause)
                    orig_span_text = llm_answer[span_start:span_end]
                else:
                    span_start = sent_start
                    span_end   = sent_end
                    orig_span_text = sent_text

                span = ClaimSpan(
                    claim_id=claim_id,
                    text=atomic,
                    start_char=span_start,
                    end_char=span_end,
                    original_text=orig_span_text,
                )
                claim_spans.append(span)
                claim_id += 1

        logger.info(f"Extracted {len(claim_spans)} claim spans.")
        return claim_spans

    def extract(self, llm_answer: str, dedup: bool = True) -> List[str]:
        """
        Backward-compatible method: returns claim strings.

        Args:
            llm_answer: Raw text output from the LLM.
            dedup:      If True, deduplicates identical claim strings (legacy behavior).

        Returns:
            List of atomic factual claim strings.
        """
        spans = self.extract_spans(llm_answer)
        if not dedup:
            return [cs.text for cs in spans]

        seen = set()
        deduped = []
        for cs in spans:
            key = cs.text.lower().rstrip(".")
            if key not in seen:
                seen.add(key)
                deduped.append(cs.text)
        return deduped

    # ------------------------------------------------------------------
    # Private Helpers
    # ------------------------------------------------------------------

    def _is_factual(self, text: str) -> bool:
        """
        Return True if the sentence is a declarative factual claim.
        Questions, exclamations, and transition phrases are excluded.
        """
        if text.endswith("?") or text.endswith("!"):
            return False
        if len(text) < self.config.min_claim_length:
            return False

        TRANSITION_PREFIXES = {
            "however", "therefore", "in conclusion", "in summary",
            "to summarize", "overall", "on the other hand",
            "firstly", "secondly", "finally", "additionally",
            "in other words", "as mentioned", "as stated",
        }
        lower = text.lower()
        if any(lower.startswith(p) for p in TRANSITION_PREFIXES):
            return False
        return True

    def _atomize(self, sentence: str) -> List[str]:
        """
        Split a sentence containing coordinate clauses into
        individual atomic claims.

        Strategy: parse with spaCy, find coordinating conjunctions
        attached to root verb phrases, split at those points.
        """
        doc = self.nlp(sentence)

        root = None
        coord_verbs = []

        for token in doc:
            if token.dep_ == "ROOT":
                root = token
            if token.dep_ == "conj" and token.head.dep_ == "ROOT":
                coord_verbs.append(token)

        if not coord_verbs or root is None:
            return [self._clean_claim(sentence)]

        claims = []
        prev_end = 0

        for cv in coord_verbs:
            cc_token = None
            for tok in cv.lefts:
                if tok.dep_ == "cc":
                    cc_token = tok
                    break

            split_idx = cc_token.i if cc_token else cv.i
            part1 = doc[prev_end:split_idx].text.strip()
            if part1:
                claims.append(self._clean_claim(part1))
            prev_end = cv.i

        final_part = doc[prev_end:].text.strip()
        if final_part:
            claims.append(self._clean_claim(final_part))

        valid = [c for c in claims if len(c) >= self.config.min_claim_length]
        return valid if valid else [self._clean_claim(sentence)]

    def _clean_claim(self, text: str) -> str:
        """Normalize whitespace and ensure claim ends with a period."""
        text = re.sub(r"\s+", " ", text).strip()
        if text and not text.endswith("."):
            text += "."
        return text
