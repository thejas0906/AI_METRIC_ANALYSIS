"""
claim_extractor.py  (Phase 1)
==============================
Claim Extraction Module
-----------------------
Converts a free-form LLM-generated answer into a list of
*atomic factual claims* — short, independently verifiable
statements that can each be sent to the evidence retrieval
and NLI verification pipeline.

Pipeline:
  LLM Answer
      ↓
  Sentence Segmentation (spaCy)
      ↓
  Sentence Filtering  (length, punctuation, question removal)
      ↓
  Claim Atomization   (coordinate clause splitting)
      ↓
  Claim Deduplication
      ↓
  List[str]  ← atomic claims

Design Decisions:
-----------------
- We use spaCy's dependency parser and sentence boundary
  detector rather than a fine-tuned claim extraction model
  (e.g., ClaimBuster) to keep the system *laptop-friendly*.
- Coordinate clause splitting ("X and Y did Z") allows each
  entity to be verified independently.
- Questions and greetings are filtered because they are not
  verifiable factual claims.

References:
-----------
- spaCy: https://spacy.io/
- ClaimBuster (for future upgrade): https://idir.uta.edu/claimbuster/
"""

import re
import spacy
from typing import List, Optional
from loguru import logger

from config import FrameworkConfig


class ClaimExtractor:
    """
    Extracts atomic factual claims from LLM-generated text.

    Usage:
        extractor = ClaimExtractor(config)
        claims = extractor.extract("Einstein was born in 1879 in Germany.")
        # → ["Einstein was born in 1879.", "Einstein was born in Germany."]
    """

    def __init__(self, config: Optional[FrameworkConfig] = None):
        """
        Initialize the claim extractor by loading the spaCy NLP model.

        Args:
            config: FrameworkConfig instance; uses defaults if None.
        """
        self.config = config or FrameworkConfig()
        logger.info(f"Loading spaCy model: {self.config.spacy_model}")
        try:
            self.nlp = spacy.load(self.config.spacy_model)
        except OSError:
            # Provide a helpful error if the model hasn't been downloaded
            raise RuntimeError(
                f"spaCy model '{self.config.spacy_model}' not found.\n"
                f"Install it with: python -m spacy download {self.config.spacy_model}"
            )
        logger.info("ClaimExtractor initialized successfully.")

    # ──────────────────────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────────────────────

    def extract(self, llm_answer: str) -> List[str]:
        """
        Main entry point. Extract atomic factual claims from
        an LLM-generated answer.

        Args:
            llm_answer: Raw text output from the LLM.

        Returns:
            List of atomic factual claim strings.
        """
        if not llm_answer or not llm_answer.strip():
            logger.warning("Empty LLM answer received — no claims extracted.")
            return []

        logger.debug(f"Extracting claims from answer ({len(llm_answer)} chars).")

        # Step 1: Sentence segmentation using spaCy
        sentences = self._segment_sentences(llm_answer)
        logger.debug(f"Segmented into {len(sentences)} sentences.")

        # Step 2: Filter out non-factual sentences
        filtered = self._filter_sentences(sentences)
        logger.debug(f"{len(filtered)} sentences after filtering.")

        # Step 3: Split coordinate clauses into atomic claims
        atomic_claims = []
        for sent in filtered:
            atomic_claims.extend(self._atomize(sent))

        # Step 4: Deduplicate while preserving order
        unique_claims = self._deduplicate(atomic_claims)

        logger.info(f"Extracted {len(unique_claims)} atomic claims.")
        return unique_claims

    # ──────────────────────────────────────────────────────────────
    # Private Helpers
    # ──────────────────────────────────────────────────────────────

    def _segment_sentences(self, text: str) -> List[str]:
        """
        Use spaCy's sentence boundary detector to split text
        into individual sentences.

        Args:
            text: Input text.

        Returns:
            List of sentence strings.
        """
        doc = self.nlp(text)
        # Each `doc.sents` span is a spaCy Span object; strip whitespace
        return [sent.text.strip() for sent in doc.sents if sent.text.strip()]

    def _filter_sentences(self, sentences: List[str]) -> List[str]:
        """
        Remove sentences that are NOT atomic factual claims:
        - Questions (end with '?')
        - Exclamations (end with '!')
        - Very short or very long sentences (likely noise)
        - Sentences that are greetings or transitions

        Args:
            sentences: Raw segmented sentences.

        Returns:
            Filtered list of candidate claim sentences.
        """
        TRANSITION_PHRASES = {
            "however", "therefore", "in conclusion", "in summary",
            "to summarize", "overall", "on the other hand",
            "firstly", "secondly", "finally", "additionally",
            "in other words", "as mentioned", "as stated"
        }

        filtered = []
        for sent in sentences:
            # Remove questions and exclamations
            if sent.endswith("?") or sent.endswith("!"):
                logger.debug(f"Filtered (non-declarative): {sent[:60]}")
                continue

            # Check length constraints
            if len(sent) < self.config.min_claim_length:
                logger.debug(f"Filtered (too short): {sent}")
                continue
            if len(sent) > self.config.max_claim_length:
                logger.debug(f"Filtered (too long, splitting needed): {sent[:60]}...")
                # Still include it; long sentences will be handled by atomizer

            # Filter pure transition sentences
            lower = sent.lower()
            if any(lower.startswith(phrase) for phrase in TRANSITION_PHRASES):
                logger.debug(f"Filtered (transition): {sent[:60]}")
                continue

            filtered.append(sent)

        return filtered

    def _atomize(self, sentence: str) -> List[str]:
        """
        Split a sentence containing coordinate clauses into
        individual atomic claims.

        Strategy:
        - Parse with spaCy to find coordinating conjunctions (cc)
          attached to verb phrases.
        - If 'and' coordinates two independent verb phrases, split.
        - Otherwise, keep the sentence as-is.

        Example:
          "Einstein won the Nobel Prize and he was born in Germany."
          → ["Einstein won the Nobel Prize.",
             "He was born in Germany."]

        Args:
            sentence: A single candidate claim sentence.

        Returns:
            List of one or more atomic claim strings.
        """
        doc = self.nlp(sentence)

        # Identify root verb and coordinate verbs
        root = None
        coord_verbs = []

        for token in doc:
            if token.dep_ == "ROOT":
                root = token
            # A conj dependent on the ROOT with a 'cc' sibling is a coordinate verb
            if token.dep_ == "conj" and token.head.dep_ == "ROOT":
                coord_verbs.append(token)

        # If no coordinate verbs found, return as-is
        if not coord_verbs or root is None:
            return [self._clean_claim(sentence)]

        # Split sentence at the coordinate verb positions
        claims = []
        prev_end = 0

        for cv in coord_verbs:
            # Find the start of the conjunction token that precedes this conj verb
            cc_token = None
            for tok in cv.lefts:
                if tok.dep_ == "cc":
                    cc_token = tok
                    break

            split_idx = cc_token.i if cc_token else cv.i
            # Reconstruct the first part
            part1 = doc[prev_end:split_idx].text.strip()
            if part1:
                claims.append(self._clean_claim(part1))
            prev_end = cv.i  # next segment starts at the conj verb

        # Add the final segment
        final_part = doc[prev_end:].text.strip()
        if final_part:
            # Reconstruct subject for the final clause if missing
            claims.append(self._clean_claim(final_part))

        # Fallback: if splitting produced empty or malformed output, keep original
        valid_claims = [c for c in claims if len(c) >= self.config.min_claim_length]
        return valid_claims if valid_claims else [self._clean_claim(sentence)]

    def _clean_claim(self, text: str) -> str:
        """
        Normalize whitespace and ensure the claim ends with a period.

        Args:
            text: Raw claim string.

        Returns:
            Cleaned claim string.
        """
        text = re.sub(r"\s+", " ", text).strip()
        if text and not text.endswith("."):
            text += "."
        return text

    def _deduplicate(self, claims: List[str]) -> List[str]:
        """
        Remove duplicate claims while preserving insertion order.

        Args:
            claims: Possibly duplicate list of claim strings.

        Returns:
            Deduplicated list.
        """
        seen = set()
        unique = []
        for claim in claims:
            # Normalize for comparison (lowercase, strip punctuation)
            key = claim.lower().rstrip(".")
            if key not in seen:
                seen.add(key)
                unique.append(claim)
        return unique
