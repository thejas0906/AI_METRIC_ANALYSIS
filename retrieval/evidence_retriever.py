"""
evidence_retriever.py  (Phase 2)
=================================
Evidence Retrieval Module
--------------------------
For each atomic claim, retrieves supporting evidence passages
from Wikipedia. Evidence is returned as structured EvidenceItem
objects containing the passage text, source URL, and metadata.

Pipeline:
  Claim (str)
      ↓
  Keyword Extraction  (spaCy named entities + noun chunks)
      ↓
  Wikipedia Search Query Construction
      ↓
  Wikipedia API Call (top-K results)
      ↓
  Passage Segmentation + Relevance Scoring
      ↓
  List[EvidenceItem]

Design Decisions:
-----------------
- Wikipedia API is used because it is free, comprehensive,
  and requires no authentication — ideal for a laptop prototype.
- Named entity extraction provides focused search queries.
- Passages are scored by term overlap with the claim (BM25-lite)
  to rank the most relevant evidence first.
- Future extension: add a `source_type` field to EvidenceItem
  and route to government / academic APIs based on claim category.

References:
-----------
- wikipedia-api: https://pypi.org/project/Wikipedia-API/
- MediaWiki API: https://www.mediawiki.org/wiki/API:Search
"""

import re
import math
import time
from dataclasses import dataclass, field
from typing import List, Optional
from loguru import logger

try:
    import wikipediaapi
except ImportError:
    wikipediaapi = None

try:
    import spacy
except ImportError:
    spacy = None

from config import FrameworkConfig, EVIDENCE_WEIGHTS
from retrieval.models import EvidenceItem, RetrievalResult


# ──────────────────────────────────────────────────────────────────
# Data Structures  (imported from retrieval.models for reusability)
# ──────────────────────────────────────────────────────────────────
# EvidenceItem and RetrievalResult are defined in retrieval/models.py
# and re-exported here for backward compatibility.
__all__ = ["EvidenceItem", "RetrievalResult", "EvidenceRetriever"]


# ──────────────────────────────────────────────────────────────────
# Main Retriever Class
# ──────────────────────────────────────────────────────────────────

class EvidenceRetriever:
    """
    Retrieves evidence passages from Wikipedia for each factual claim.

    Usage:
        retriever = EvidenceRetriever(config)
        result = retriever.retrieve("Einstein was born in 1879.")
        for ev in result.evidence:
            print(ev.passage, ev.source_url)
    """

    def __init__(self, config: Optional[FrameworkConfig] = None):
        """
        Initialize the retriever with Wikipedia API client and spaCy.

        Args:
            config: FrameworkConfig instance; uses defaults if None.
        """
        self.config = config or FrameworkConfig()

        if wikipediaapi is None:
            raise RuntimeError(
                "wikipedia-api is required for EvidenceRetriever.\n"
                "Install it with: pip install wikipedia-api"
            )

        # Wikipedia-API client (requires user agent per API policy)
        user_agent = getattr(
            self.config,
            "wikipedia_user_agent",
            "SelectiveHallucinationCorrectionBot/1.0 (https://github.com/thejas0906/AI_METRIC_ANALYSIS; hallucination-research@example.com)"
        )
        self.wiki = wikipediaapi.Wikipedia(
            user_agent=user_agent,
            language=getattr(self.config, "wikipedia_language", "en"),
            extract_format=wikipediaapi.ExtractFormat.WIKI,
        )

        if spacy is None:
            raise RuntimeError(
                "spaCy is required for EvidenceRetriever.\n"
                "Install it with: pip install spacy && python -m spacy download en_core_web_sm"
            )

        # Load spaCy for keyword/entity extraction from claims
        logger.info(f"Loading spaCy model for keyword extraction: {self.config.spacy_model}")
        try:
            self.nlp = spacy.load(self.config.spacy_model)
        except OSError:
            raise RuntimeError(
                f"spaCy model '{self.config.spacy_model}' not found.\n"
                f"Install it: python -m spacy download {self.config.spacy_model}"
            )

        logger.info("EvidenceRetriever initialized.")

    # ──────────────────────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────────────────────

    def retrieve(self, claim: str) -> RetrievalResult:
        """
        Retrieve top-K evidence passages from Wikipedia for a single claim.

        Args:
            claim: An atomic factual claim string.

        Returns:
            RetrievalResult with claim, evidence list, and query used.
        """
        # Step 1: Build a focused search query from claim entities
        query = self._build_query(claim)
        logger.debug(f"Query for claim '{claim[:50]}': '{query}'")

        result = RetrievalResult(claim=claim, query_used=query)

        # Step 2: Search Wikipedia and retrieve candidate pages
        pages = self._search_wikipedia(query)

        # Step 3: For each page, extract the most relevant passages
        all_evidence: List[EvidenceItem] = []
        for page_title, page_url in pages:
            passages = self._extract_passages(page_title, claim)
            all_evidence.extend(passages)
            # Respect Wikipedia rate limits
            time.sleep(0.1)

        # Step 4: Sort by relevance and keep top-K
        all_evidence.sort(key=lambda e: e.relevance_score, reverse=True)
        result.evidence = all_evidence[: self.config.wikipedia_top_k]

        logger.info(
            f"Retrieved {len(result.evidence)} evidence items for: "
            f"'{claim[:50]}...'"
        )
        return result

    def retrieve_batch(self, claims: List[str]) -> List[RetrievalResult]:
        """
        Retrieve evidence for a list of claims.

        Args:
            claims: List of atomic factual claim strings.

        Returns:
            List of RetrievalResult objects, one per claim.
        """
        results = []
        for i, claim in enumerate(claims):
            logger.info(f"Retrieving evidence for claim {i+1}/{len(claims)}")
            results.append(self.retrieve(claim))
        return results

    # ──────────────────────────────────────────────────────────────
    # Private Helpers
    # ──────────────────────────────────────────────────────────────

    def _build_query(self, claim: str) -> str:
        """
        Extract named entities and key noun phrases from the claim
        to construct a focused Wikipedia search query.

        Strategy:
        1. Parse claim with spaCy.
        2. Extract named entities (PERSON, ORG, GPE, DATE, etc.)
        3. If no entities, fall back to noun chunks.
        4. Join top-3 keywords as the query.

        Args:
            claim: Atomic factual claim string.

        Returns:
            Search query string.
        """
        doc = self.nlp(claim)

        # Collect named entities (e.g., "Albert Einstein", "Germany", "1879")
        entities = [ent.text for ent in doc.ents]

        if entities:
            # Use the first 3 entities as the query
            query = " ".join(entities[:3])
        else:
            # Fallback: use noun chunks (e.g., "the Nobel Prize")
            noun_chunks = [chunk.root.text for chunk in doc.noun_chunks]
            query = " ".join(noun_chunks[:3]) if noun_chunks else claim[:80]

        return query.strip()

    def _search_wikipedia(self, query: str) -> List[tuple]:
        """
        Search Wikipedia for pages matching the query.
        Returns a list of (page_title, page_url) tuples.

        Uses MediaWiki's search API via the wikipedia-api library.

        Args:
            query: Search query string.

        Returns:
            List of (title, url) tuples for candidate pages.
        """
        import requests

        # Wikipedia OpenSearch API endpoint
        api_url = "https://en.wikipedia.org/w/api.php"
        params = {
            "action": "opensearch",
            "search": query,
            "limit": 3,          # top-3 candidate pages
            "namespace": 0,
            "format": "json",
        }

        headers = {
            "User-Agent": getattr(
                self.config,
                "wikipedia_user_agent",
                "SelectiveHallucinationCorrectionBot/1.0 (https://github.com/thejas0906/AI_METRIC_ANALYSIS; hallucination-research@example.com)"
            )
        }

        for attempt in range(3):
            try:
                resp = requests.get(
                    api_url,
                    params=params,
                    timeout=10,
                    headers=headers,
                )
                if resp.status_code == 429:
                    wait_time = (attempt + 1) * 2
                    logger.warning(f"Wikipedia 429 rate limit. Waiting {wait_time}s before retry...")
                    time.sleep(wait_time)
                    continue
                resp.raise_for_status()
                data = resp.json()
                # OpenSearch returns: [query, [titles], [descriptions], [urls]]
                titles = data[1]
                urls   = data[3]
                return list(zip(titles, urls))
            except Exception as e:
                if attempt == 2:
                    logger.warning(f"Wikipedia search failed for query '{query}': {e}")
                time.sleep(1)
        return []

    def _extract_passages(
        self, page_title: str, claim: str
    ) -> List[EvidenceItem]:
        """
        Fetch the Wikipedia page and extract relevant text passages.

        Splits the page summary into sentences and scores each sentence
        by term overlap with the claim (a lightweight BM25-style score).

        Args:
            page_title: Title of the Wikipedia page to fetch.
            claim:      The claim we are searching evidence for.

        Returns:
            List of EvidenceItem objects with scored passages.
        """
        # Fetch page via wikipedia-api
        page = self.wiki.page(page_title)
        if not page.exists():
            logger.debug(f"Wikipedia page not found: {page_title}")
            return []

        # Split page text into sentences using simple regex
        full_text = page.summary
        sentences = re.split(r"(?<=[.!?])\s+", full_text)

        evidence_items = []
        claim_tokens = set(claim.lower().split())

        for sent in sentences:
            if len(sent) < 20:   # skip trivially short sentences
                continue

            # Compute simple term-overlap relevance score
            sent_tokens = set(sent.lower().split())
            overlap = len(claim_tokens & sent_tokens)
            score = self._bm25_lite(overlap, len(sent_tokens), len(claim_tokens))

            item = EvidenceItem(
                passage=sent.strip(),
                source_url=page.fullurl,
                source_type="wikipedia",
                page_title=page.title,
                relevance_score=score,
                reliability_weight=EVIDENCE_WEIGHTS.get("wikipedia", 0.8),
            )
            evidence_items.append(item)

        return evidence_items

    @staticmethod
    def _bm25_lite(overlap: int, doc_len: int, query_len: int) -> float:
        """
        A simplified BM25-style relevance score based on term overlap.

        BM25 formula (simplified):
            score = IDF * (overlap * (k+1)) / (overlap + k * (1 - b + b * dl/avg_dl))

        Here we use fixed constants for a lightweight approximation:
            k = 1.5, b = 0.75, avg_dl = 20 tokens

        Args:
            overlap:    Number of shared tokens between claim and passage.
            doc_len:    Number of tokens in the passage.
            query_len:  Number of tokens in the claim.

        Returns:
            A float relevance score ≥ 0.
        """
        if overlap == 0 or doc_len == 0:
            return 0.0

        k = 1.5
        b = 0.75
        avg_dl = 20.0  # average document length in tokens

        # IDF approximation: log(1 + 1/query_len) scales for short queries
        idf = math.log(1 + 1.0 / max(query_len, 1))

        tf = (overlap * (k + 1)) / (
            overlap + k * (1 - b + b * (doc_len / avg_dl))
        )

        return idf * tf
