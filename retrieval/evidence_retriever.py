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

import os
import re
import math
import time
from dataclasses import dataclass, field
from typing import List, Optional, Dict
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
        self._current_response: str = ""
        self._current_topic: str = ""
        self._dominant_subject: Optional[str] = None

        # Persistent Wikipedia cache
        cache_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data"
        )
        os.makedirs(cache_dir, exist_ok=True)
        self._cache_file = os.path.join(cache_dir, "wikipedia_cache.json")
        self._search_cache: Dict[str, list] = {}
        self._page_cache: Dict[str, dict] = {}
        if os.path.exists(self._cache_file):
            try:
                import json
                with open(self._cache_file, "r", encoding="utf-8") as f:
                    _cdata = json.load(f)
                    self._search_cache = _cdata.get("searches", {})
                    self._page_cache = _cdata.get("pages", {})
            except Exception:
                pass

    def _save_cache(self) -> None:
        """Persist Wikipedia cache to disk."""
        try:
            import json
            with open(self._cache_file, "w", encoding="utf-8") as f:
                json.dump({"searches": self._search_cache, "pages": self._page_cache}, f)
        except Exception:
            pass

    # ──────────────────────────────────────────────────────────────
    # Context Management
    # ──────────────────────────────────────────────────────────────

    def set_context(self, response: str = "", topic: str = "") -> None:
        """
        Set the active response and topic context for the retriever.
        Used to resolve pronouns and preserve dominant named entities across atomic claims.
        """
        self._current_response = response or ""
        self._current_topic = topic or ""
        self._dominant_subject = self._extract_dominant_subject(self._current_response, self._current_topic)

    def _extract_dominant_subject(self, response: str = "", topic: str = "") -> Optional[str]:
        """
        Extract the dominant named entity or subject from the query topic or original response.
        Used to resolve pronoun-only claims and anchor retrieval to the correct topic.
        """
        # 1. First check topic/query
        if topic and topic.strip():
            doc_topic = self.nlp(topic.strip())
            for ent in doc_topic.ents:
                if ent.label_ in {"PERSON", "ORG", "GPE", "WORK_OF_ART", "PRODUCT", "EVENT"}:
                    return ent.text.strip()
            for chunk in doc_topic.noun_chunks:
                clean_words = [
                    t.text for t in chunk 
                    if not t.is_stop and t.pos_ in {"NOUN", "PROPN"}
                ]
                if clean_words:
                    return " ".join(clean_words).strip()

        # 2. Check response
        if response and response.strip():
            doc_resp = self.nlp(response.strip())
            ent_counts: Dict[str, int] = {}
            for ent in doc_resp.ents:
                if ent.label_ in {"PERSON", "ORG", "WORK_OF_ART"}:
                    ent_text = ent.text.strip()
                    ent_counts[ent_text] = ent_counts.get(ent_text, 0) + 1
            if ent_counts:
                return max(ent_counts, key=ent_counts.get)
            
            for sent in doc_resp.sents:
                for ent in sent.ents:
                    if ent.label_ not in {"CARDINAL", "ORDINAL", "DATE", "TIME", "PERCENT", "QUANTITY"}:
                        return ent.text.strip()
                for chunk in sent.noun_chunks:
                    clean_words = [
                        t.text for t in chunk 
                        if not t.is_stop and t.pos_ in {"NOUN", "PROPN"}
                    ]
                    if clean_words:
                        return " ".join(clean_words).strip()
                break

        return None

    # ──────────────────────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────────────────────

    def retrieve(
        self,
        claim: str,
        context: Optional[str] = None,
        topic: Optional[str] = None,
    ) -> RetrievalResult:
        """
        Retrieve top-K evidence passages from Wikipedia for a single claim.

        Args:
            claim:   An atomic factual claim string.
            context: Optional full response text.
            topic:   Optional query/topic.

        Returns:
            RetrievalResult with claim, evidence list, and query used.
        """
        if context or topic:
            self.set_context(response=context or "", topic=topic or "")

        # Step 1: Build a focused search query from claim entities
        query = self._build_query(claim, context=context, topic=topic)
        logger.info(f"Generated query: '{query}'")

        result = RetrievalResult(claim=claim, query_used=query)

        # Step 2: Search Wikipedia and retrieve candidate pages
        pages = self._search_wikipedia(query)
        page_titles = [p[0] for p in pages]
        logger.info(f"Retrieved page titles: {page_titles}")

        # Step 3: For each page, extract the most relevant passages
        all_evidence: List[EvidenceItem] = []
        for page_title, page_url in pages:
            passages = self._extract_passages(page_title, claim)
            all_evidence.extend(passages)
            # Respect Wikipedia rate limits
            time.sleep(0.1)

        # Step 4: Sort by relevance and keep top-K (prioritize non-zero relevance)
        all_evidence.sort(key=lambda e: e.relevance_score, reverse=True)
        relevant_evidence = [e for e in all_evidence if e.relevance_score > 0]
        result.evidence = (relevant_evidence if relevant_evidence else all_evidence)[: self.config.wikipedia_top_k]

        logger.info(f"Retrieved evidence count: {len(result.evidence)}")
        return result

    def retrieve_batch(
        self,
        claims: List[str],
        context: Optional[str] = None,
        topic: Optional[str] = None,
    ) -> List[RetrievalResult]:
        """
        Retrieve evidence for a list of claims.

        Args:
            claims:  List of atomic factual claim strings.
            context: Optional full response text.
            topic:   Optional query/topic.

        Returns:
            List of RetrievalResult objects, one per claim.
        """
        if context or topic:
            self.set_context(response=context or "", topic=topic or "")
        results = []
        for i, claim in enumerate(claims):
            logger.info(f"Retrieving evidence for claim {i+1}/{len(claims)}")
            results.append(self.retrieve(claim, context=context, topic=topic))
        return results

    # ──────────────────────────────────────────────────────────────
    # Private Helpers
    # ──────────────────────────────────────────────────────────────

    def _build_query(
        self,
        claim: str,
        context: Optional[str] = None,
        topic: Optional[str] = None,
    ) -> str:
        """
        Extract named entities and key noun phrases from the claim
        to construct a focused Wikipedia search query.

        Improvements:
        - Resolves pronoun-only claims ("He", "She", "They", etc.) using the dominant subject.
        - Preserves dominant named entities from the original response/topic.
        - Filters out low-information entities like CARDINAL values ("one", "two").
        - Adds descriptive noun chunks when entities are sparse.
        """
        if context or topic:
            self.set_context(response=context or "", topic=topic or "")

        doc = self.nlp(claim)

        LOW_INFO_LABELS = {"CARDINAL", "ORDINAL", "PERCENT", "QUANTITY", "TIME"}
        NUMBER_WORDS = {
            "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten",
            "first", "second", "third", "fourth", "fifth"
        }
        PRONOUNS = {"he", "she", "they", "it", "his", "her", "their", "its", "him", "them"}

        # 1. Collect high-information named entities from claim
        valid_entities: List[str] = []
        for ent in doc.ents:
            if ent.label_ in LOW_INFO_LABELS:
                continue
            cleaned = ent.text.strip()
            if cleaned.lower() in NUMBER_WORDS or cleaned.isdigit():
                continue
            if cleaned not in valid_entities:
                valid_entities.append(cleaned)

        # 2. Check for pronoun subjects or missing dominant subject
        first_tokens = [t.text.lower() for t in doc[:3]]
        has_pronoun = any(p in PRONOUNS for p in first_tokens)
        
        dominant = self._dominant_subject or self._extract_dominant_subject(self._current_response, self._current_topic)

        query_tokens: List[str] = []

        # If dominant subject exists, preserve/anchor with it:
        # especially if the claim starts with a pronoun, lacks high-value entities, or lacks the dominant subject
        if dominant:
            dominant_lower = dominant.lower()
            claim_has_dominant = dominant_lower in claim.lower()
            if has_pronoun or not claim_has_dominant or not valid_entities:
                query_tokens.append(dominant)

        for ent in valid_entities:
            # Avoid redundant duplicate of dominant subject
            if dominant and (ent.lower() in dominant.lower() or dominant.lower() in ent.lower()):
                if dominant not in query_tokens:
                    query_tokens.append(dominant)
                continue
            if ent not in query_tokens:
                query_tokens.append(ent)

        # 3. If query has fewer than 3 terms, enrich with informative noun chunks
        if len(query_tokens) < 3:
            for chunk in doc.noun_chunks:
                clean_chunk_words = [
                    t.text for t in chunk
                    if not t.is_stop 
                    and t.pos_ in {"NOUN", "PROPN"}
                    and t.text.lower() not in NUMBER_WORDS
                    and t.text.lower() not in PRONOUNS
                ]
                chunk_str = " ".join(clean_chunk_words).strip()
                if (
                    chunk_str 
                    and chunk_str.lower() not in " ".join(query_tokens).lower() 
                    and len(chunk_str) > 2
                ):
                    query_tokens.append(chunk_str)
                    if len(query_tokens) >= 4:
                        break

        # Fallback if still empty
        if not query_tokens:
            content_words = [
                t.text for t in doc
                if not t.is_stop and not t.is_punct 
                and t.text.lower() not in NUMBER_WORDS 
                and t.text.lower() not in PRONOUNS
            ]
            query = " ".join(content_words[:4]) if content_words else claim[:80]
        else:
            query = " ".join(query_tokens[:4])

        return query.strip()

    def _search_wikipedia(self, query: str) -> List[tuple]:
        """
        Search Wikipedia for pages matching the query using full-text search.
        Returns a list of (page_title, page_url) tuples.

        Uses MediaWiki's full-text search API:
            action=query
            list=search
            srsearch=<query>

        Args:
            query: Search query string.

        Returns:
            List of (title, url) tuples for candidate pages.
        """
        # Check search cache first
        cache_key = f"{query}__topk{self.config.wikipedia_top_k}"
        if cache_key in self._search_cache:
            return [tuple(x) for x in self._search_cache[cache_key]]

        import requests

        api_url = "https://en.wikipedia.org/w/api.php"
        limit = max(self.config.wikipedia_top_k, 3)
        params = {
            "action": "query",
            "list": "search",
            "srsearch": query,
            "srlimit": limit,
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
                search_results = data.get("query", {}).get("search", [])
                pages = []
                for item in search_results:
                    title = item.get("title", "").strip()
                    if title:
                        url = f"https://en.wikipedia.org/wiki/{title.replace(' ', '_')}"
                        pages.append((title, url))

                # Also include dominant subject page if known and exists
                if self._dominant_subject:
                    dom_title = self._dominant_subject.strip()
                    existing_lower = {p[0].lower() for p in pages}
                    if dom_title.lower() not in existing_lower:
                        try:
                            dom_page = self.wiki.page(dom_title)
                            if dom_page.exists():
                                pages.append((dom_page.title, dom_page.fullurl))
                        except Exception:
                            pass

                self._search_cache[cache_key] = pages
                self._save_cache()
                return pages
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
        # Check page cache
        if page_title in self._page_cache:
            p_data = self._page_cache[page_title]
            if not p_data.get("exists", False):
                return []
            full_text = p_data.get("summary", "")
            page_url = p_data.get("fullurl", "")
            page_real_title = p_data.get("title", page_title)
        else:
            page = self.wiki.page(page_title)
            if not page.exists():
                logger.debug(f"Wikipedia page not found: {page_title}")
                self._page_cache[page_title] = {"exists": False}
                self._save_cache()
                return []
            full_text = page.summary
            page_url = page.fullurl
            page_real_title = page.title
            self._page_cache[page_title] = {
                "exists": True,
                "summary": full_text,
                "fullurl": page_url,
                "title": page_real_title,
            }
            self._save_cache()

        # Split page text into sentences using simple regex
        sentences = re.split(r"(?<=[.!?])\s+", full_text)

        evidence_items = []
        
        # Focus term overlap on meaningful content tokens (filtering trivial stopwords)
        from spacy.lang.en.stop_words import STOP_WORDS
        claim_tokens = {
            re.sub(r"\W+", "", w.lower()) 
            for w in claim.split() 
            if w.lower() not in STOP_WORDS and len(w) > 1
        }
        claim_tokens.discard("")
        if not claim_tokens:
            claim_tokens = {re.sub(r"\W+", "", w.lower()) for w in claim.split() if len(w) > 1}
            claim_tokens.discard("")

        for sent in sentences:
            if len(sent) < 20:   # skip trivially short sentences
                continue

            sent_tokens = {
                re.sub(r"\W+", "", w.lower()) 
                for w in sent.split() 
                if w.lower() not in STOP_WORDS and len(w) > 1
            }
            sent_tokens.discard("")
            overlap = len(claim_tokens & sent_tokens)
            score = self._bm25_lite(overlap, len(sent_tokens), len(claim_tokens))

            item = EvidenceItem(
                passage=sent.strip(),
                source_url=page_url,
                source_type="wikipedia",
                page_title=page_real_title,
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

