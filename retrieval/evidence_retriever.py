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

        # Step 4: Sort by relevance and keep top-K (prioritize non-zero relevance and non-redundant passages)
        all_evidence.sort(key=lambda e: e.relevance_score, reverse=True)
        unique_evidence: List[EvidenceItem] = []
        seen_texts: List[str] = []
        for ev in all_evidence:
            if ev.relevance_score <= 0 and unique_evidence:
                continue
            norm_p = re.sub(r"\W+", " ", ev.passage.lower()).strip()
            # Skip if nearly identical or heavily overlapping with already selected passage
            is_dup = False
            for prev in seen_texts:
                if norm_p in prev or prev in norm_p or (len(set(norm_p.split()) & set(prev.split())) / max(len(set(norm_p.split())), 1) > 0.85):
                    is_dup = True
                    break
            if not is_dup:
                unique_evidence.append(ev)
                seen_texts.append(norm_p)
                if len(unique_evidence) >= self.config.wikipedia_top_k:
                    break

        result.evidence = unique_evidence if unique_evidence else all_evidence[: self.config.wikipedia_top_k]

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
        to construct a focused, entity-centric Wikipedia search query.

        Enhancements:
        - Entity-centric anchoring: preserves proper entities (PERSON, ORG, GPE, EVENT).
        - Clean pronoun resolution: resolves 3rd-person pronouns using dominant subject without possessive corruption.
        - Prevents topic bleed: does not inject dominant topic entity when claim already has its own distinct named entity.
        - Preserves 4-digit years (e.g. 1879, 1921, 1953, 1969, 1705, 1914) essential for factual discriminators.
        - Filters out filler words and punctuation.
        """
        if context or topic:
            self.set_context(response=context or "", topic=topic or "")

        doc = self.nlp(claim)

        NUMBER_WORDS = {
            "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten",
            "first", "second", "third", "fourth", "fifth"
        }
        PRONOUNS = {"he", "she", "they", "it", "his", "her", "their", "its", "him", "them"}
        HIGH_VALUE_ENT_LABELS = {"PERSON", "ORG", "GPE", "LOC", "EVENT", "WORK_OF_ART", "FAC", "NORP"}

        # 1. Clean dominant subject
        raw_dom = self._dominant_subject or self._extract_dominant_subject(self._current_response, self._current_topic)
        dominant = None
        if raw_dom:
            dominant = re.sub(r"['’]s\b", "", raw_dom).strip()
            if dominant.lower() in {"the", "a", "an"}:
                dominant = None

        # 2. Collect high-information named entities and 4-digit years
        valid_entities: List[str] = []
        for ent in doc.ents:
            if ent.label_ in HIGH_VALUE_ENT_LABELS:
                cleaned = re.sub(r"['’]s\b", "", ent.text).strip()
                if cleaned.lower() not in NUMBER_WORDS and not cleaned.isdigit() and cleaned not in valid_entities:
                    valid_entities.append(cleaned)
            elif ent.label_ in {"DATE", "CARDINAL"}:
                m = re.search(r"\b(1\d{3}|20\d{2})\b", ent.text)
                if m and m.group(1) not in valid_entities:
                    valid_entities.append(m.group(1))

        # Check for 4-digit years in raw text
        for yr in re.findall(r"\b(1\d{3}|20\d{2})\b", claim):
            if yr not in valid_entities:
                valid_entities.append(yr)

        # 3. Check for pronoun subjects
        first_tokens = [t.text.lower() for t in doc[:3]]
        has_pronoun = any(p in PRONOUNS for p in first_tokens)

        query_tokens: List[str] = []

        # Anchor with dominant subject when:
        # - Claim begins with a pronoun (e.g. "He worked as a patent clerk...")
        # - Claim has NO high-value named entities
        # - Claim explicitly mentions the dominant subject
        if dominant:
            dom_lower = dominant.lower()
            claim_has_dom = dom_lower in claim.lower()
            if has_pronoun or not valid_entities or claim_has_dom:
                query_tokens.append(dominant)

        for ent in valid_entities:
            if dominant and ent.lower() == dominant.lower():
                if dominant not in query_tokens:
                    query_tokens.append(dominant)
                continue
            if ent not in query_tokens:
                query_tokens.append(ent)

        # 4. Enrich with informative noun chunks if query is short
        if len(query_tokens) < 4:
            for chunk in doc.noun_chunks:
                clean_chunk_words = [
                    t.text for t in chunk
                    if not t.is_stop
                    and t.pos_ in {"NOUN", "PROPN"}
                    and t.text.lower() not in NUMBER_WORDS
                    and t.text.lower() not in PRONOUNS
                    and len(t.text) > 2
                ]
                chunk_str = " ".join(clean_chunk_words).strip()
                if (
                    chunk_str
                    and not any(chunk_str.lower() in q.lower() for q in query_tokens)
                    and chunk_str.lower() not in {"day", "this day", "stay", "pursuit"}
                ):
                    query_tokens.append(chunk_str)
                    if len(query_tokens) >= 5:
                        break

        # Fallback if empty
        if not query_tokens:
            content_words = [
                t.text for t in doc
                if not t.is_stop and not t.is_punct
                and t.text.lower() not in NUMBER_WORDS
                and t.text.lower() not in PRONOUNS
            ]
            query = " ".join(content_words[:5]) if content_words else claim[:80]
        else:
            query = " ".join(query_tokens[:5])

        return query.strip()

    def _search_wikipedia(self, query: str) -> List[tuple]:
        """
        Search Wikipedia for pages matching the query using full-text search.
        Returns a list of (page_title, page_url) tuples.

        Uses MediaWiki's full-text search API:
            action=query
            list=search
            srsearch=<query>
        """
        # Search limit: expanded to 6 candidates for broader recall
        search_limit = max(getattr(self.config, "wikipedia_top_k", 3), 6)
        cache_key = f"{query}__topk{search_limit}"
        if cache_key in self._search_cache:
            return [tuple(x) for x in self._search_cache[cache_key]]

        import requests

        api_url = "https://en.wikipedia.org/w/api.php"
        params = {
            "action": "query",
            "list": "search",
            "srsearch": query,
            "srlimit": search_limit,
            "format": "json",
        }

        headers = {
            "User-Agent": getattr(
                self.config,
                "wikipedia_user_agent",
                "SelectiveHallucinationCorrectionBot/1.0 (https://github.com/thejas0906/AI_METRIC_ANALYSIS; hallucination-research@example.com)"
            )
        }

        for attempt in range(2):
            try:
                resp = requests.get(
                    api_url,
                    params=params,
                    timeout=3.0,
                    headers=headers,
                )
                if resp.status_code == 429:
                    time.sleep(0.5)
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
                    dom_clean = re.sub(r"['’]s\b", "", self._dominant_subject).strip()
                    existing_lower = {p[0].lower() for p in pages}
                    if dom_clean.lower() not in existing_lower:
                        try:
                            dom_page = self.wiki.page(dom_clean)
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
        Fetch the Wikipedia page and extract relevant text passages using:
        1. Lead section and summary extraction.
        2. Relevant body paragraph scanning (matching claim tokens).
        3. Multi-sentence evidence windowing (1-sentence, 2-sentence, 3-sentence windows).
        4. BM25-lite term overlap scoring with lead priority and token coverage boosts.

        Args:
            page_title: Title of the Wikipedia page to fetch.
            claim:      The claim we are searching evidence for.

        Returns:
            List of EvidenceItem objects with scored passages.
        """
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

        # Check page cache
        need_body_fetch = False
        if page_title in self._page_cache:
            p_data = self._page_cache[page_title]
            if not p_data.get("exists", False):
                return []
            summary_text = p_data.get("summary", "")
            page_url = p_data.get("fullurl", "")
            page_real_title = p_data.get("title", page_title)
            body_paragraphs = p_data.get("body_paragraphs", [])
            need_body_fetch = False
        else:
            need_body_fetch = True

        if need_body_fetch:
            try:
                page = self.wiki.page(page_title)
                if not page.exists():
                    logger.debug(f"Wikipedia page not found: {page_title}")
                    self._page_cache[page_title] = {"exists": False}
                    self._save_cache()
                    return []
                summary_text = page.summary
                page_url = page.fullurl
                page_real_title = page.title
                
                # Extract matching body paragraphs from full text
                body_paragraphs = []
                full_text = getattr(page, "text", "")
                if full_text:
                    raw_paras = full_text.split("\n\n")
                    for p in raw_paras[:40]:  # scan first 40 paragraphs
                        p_str = p.strip()
                        if len(p_str) < 40 or p_str.startswith("==") or p_str.startswith("See also"):
                            continue
                        p_lower = p_str.lower()
                        # Keep paragraph if it has overlap with claim content tokens
                        if any(ct in p_lower for ct in claim_tokens):
                            body_paragraphs.append(p_str)
                            if len(body_paragraphs) >= 8:
                                break

                self._page_cache[page_title] = {
                    "exists": True,
                    "summary": summary_text,
                    "fullurl": page_url,
                    "title": page_real_title,
                    "body_paragraphs": body_paragraphs,
                    "has_scanned_body": True,
                }
                self._save_cache()
            except Exception as e:
                logger.warning(f"Error fetching page '{page_title}': {e}")
                return []

        evidence_items = []

        # Prepare pools of text: lead paragraphs vs body paragraphs
        lead_paras = [p.strip() for p in summary_text.split("\n") if len(p.strip()) > 30]
        if not lead_paras and summary_text:
            lead_paras = [summary_text.strip()]

        # Helper to process paragraph into single sentences and multi-sentence sliding windows
        def process_paragraph(para: str, is_lead: bool):
            sents = [s.strip() for s in re.split(r"(?<=[.!?])\s+", para) if len(s.strip()) >= 15]
            if not sents:
                return

            candidates = []
            # 1. Single sentences
            for s in sents:
                if len(s) >= 20:
                    candidates.append(s)

            # 2. Two-sentence windows
            for i in range(len(sents) - 1):
                win2 = f"{sents[i]} {sents[i+1]}"
                if len(win2) <= 450:
                    candidates.append(win2)

            # 3. Three-sentence windows
            for i in range(len(sents) - 2):
                win3 = f"{sents[i]} {sents[i+1]} {sents[i+2]}"
                if len(win3) <= 600:
                    candidates.append(win3)

            for cand in candidates:
                cand_tokens = {
                    re.sub(r"\W+", "", w.lower())
                    for w in cand.split()
                    if w.lower() not in STOP_WORDS and len(w) > 1
                }
                cand_tokens.discard("")
                overlap = len(claim_tokens & cand_tokens)
                if overlap == 0:
                    continue

                raw_score = self._bm25_lite(overlap, len(cand_tokens), len(claim_tokens))
                
                # Priority boosts
                score = raw_score
                if is_lead:
                    score *= 1.20  # lead definition priority
                coverage = overlap / max(len(claim_tokens), 1)
                if coverage >= 0.70:
                    score *= 1.25  # high token coverage bonus

                item = EvidenceItem(
                    passage=cand.strip(),
                    source_url=page_url,
                    source_type="wikipedia",
                    page_title=page_real_title,
                    relevance_score=score,
                    reliability_weight=EVIDENCE_WEIGHTS.get("wikipedia", 0.8),
                )
                evidence_items.append(item)

        # Process lead paragraphs
        for lp in lead_paras:
            process_paragraph(lp, is_lead=True)

        # Process matching body paragraphs
        for bp in body_paragraphs:
            process_paragraph(bp, is_lead=False)

        return evidence_items

    @staticmethod
    def _bm25_lite(overlap: int, doc_len: int, query_len: int) -> float:
        """
        A simplified BM25-style relevance score based on term overlap.

        BM25 formula (simplified):
            score = IDF * (overlap * (k+1)) / (overlap + k * (1 - b + b * dl/avg_dl))

        Calibrated for windowed passages:
            k = 1.5, b = 0.75, avg_dl = 45 tokens

        Args:
            overlap:    Number of shared tokens between claim and passage.
            doc_len:    Number of tokens in the passage.
            query_len:  Number of tokens in the claim.

        Returns:
            A float relevance score >= 0.
        """
        if overlap == 0 or doc_len == 0:
            return 0.0

        k = 1.5
        b = 0.75
        avg_dl = 45.0  # calibrated average document length for windowed passages

        # IDF approximation: log(1 + 1/query_len) scales for short queries
        idf = math.log(1 + 1.0 / max(query_len, 1))

        tf = (overlap * (k + 1)) / (
            overlap + k * (1 - b + b * (doc_len / avg_dl))
        )

        return idf * tf


