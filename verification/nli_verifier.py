"""
nli_verifier.py  (Phase 4)
===========================
NLI-Based Claim Verification Module
--------------------------------------
Uses facebook/bart-large-mnli for genuine pairwise NLI inference.

For each (evidence_passage, claim) pair the model receives them as a
sequence-pair and outputs three class logits:

    label 0  ->  CONTRADICTION
    label 1  ->  NEUTRAL
    label 2  ->  ENTAILMENT

Source: facebook/bart-large-mnli config.json id2label mapping.

This module does NOT use the HuggingFace zero-shot-classification
pipeline, which treats the hypothesis as a candidate label string
formatted into a template and therefore does NOT perform genuine
sequence-pair NLI.  Instead we call AutoTokenizer +
AutoModelForSequenceClassification directly, so the model receives:

    tokenizer(premise, hypothesis, truncation=True, ...)

Verification labels:
    SUPPORTED     -- CSS >= css_supported:    evidence entails claim
    CONTRADICTED  -- contradiction_prob dominates AND evidence is strong
    UNVERIFIABLE  -- insufficient or neutral evidence; do NOT correct

CSS = entailment_prob * EQS  (Claim Support Score)

Classification thresholds (from FrameworkConfig):
    CSS >= 0.75         ->  SUPPORTED
    0.40 <= CSS < 0.75  ->  UNVERIFIABLE
    CSS <  0.40         ->  further analysis: check contradiction_prob
        contradiction_prob > 0.5  ->  CONTRADICTED
        else                      ->  UNVERIFIABLE

References:
-----------
- BART-large-mnli: https://huggingface.co/facebook/bart-large-mnli
- MNLI benchmark:  https://cims.nyu.edu/~sbowman/multinli/
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional
from loguru import logger

import torch
import torch.nn.functional as F
try:
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
except ImportError:
    AutoTokenizer = None
    AutoModelForSequenceClassification = None

from config import FrameworkConfig
from verification.evidence_quality import QualityAssessedRetrieval, ScoredEvidence


# ------------------------------------------------------------------
# Enums & Data Structures
# ------------------------------------------------------------------

class VerificationLabel(str, Enum):
    """
    Possible outcomes of NLI-based claim verification.

    SUPPORTED:    Evidence strongly entails the claim.
    CONTRADICTED: Evidence contradicts the claim (hallucinated/wrong).
    UNVERIFIABLE: Evidence is insufficient to confirm or deny.

    The distinction between CONTRADICTED and UNVERIFIABLE is critical:
    a claim should only be CORRECTED when evidence actively contradicts
    it, not merely when retrieval is weak.

    Backward compat aliases:
        HALLUCINATED          = CONTRADICTED
        INSUFFICIENT_EVIDENCE = UNVERIFIABLE
    """
    SUPPORTED             = "SUPPORTED"
    CONTRADICTED          = "CONTRADICTED"
    UNVERIFIABLE          = "UNVERIFIABLE"
    # Backward compat aliases (same string value, used in old code)
    HALLUCINATED          = "CONTRADICTED"
    INSUFFICIENT_EVIDENCE = "UNVERIFIABLE"


@dataclass
class NLIResult:
    """
    NLI scores for a single (evidence, claim) pair.

    Attributes:
        entailment_prob:    P(entailment | evidence, claim)
        neutral_prob:       P(neutral | evidence, claim)
        contradiction_prob: P(contradiction | evidence, claim)
        evidence_passage:   The evidence text used as premise.
    """
    entailment_prob:    float
    neutral_prob:       float
    contradiction_prob: float
    evidence_passage:   str


@dataclass
class ClaimVerificationResult:
    """
    Full verification outcome for a single atomic claim.

    Attributes:
        claim:          The atomic factual claim being verified.
        label:          SUPPORTED / CONTRADICTED / UNVERIFIABLE
        css:            Claim Support Score (best NLI entailment x EQS)
        best_nli:       The NLIResult from the highest-scoring evidence.
        best_evidence:  The ScoredEvidence item that produced best CSS.
        all_nli:        NLI results for all evidence items.
        css_breakdown:  Per-evidence CSS values for transparency.
    """
    claim:          str
    label:          VerificationLabel
    css:            float
    best_nli:       Optional[NLIResult]      = None
    best_evidence:  Optional[ScoredEvidence] = None
    all_nli:        List[NLIResult]           = field(default_factory=list)
    css_breakdown:  List[float]               = field(default_factory=list)


# ------------------------------------------------------------------
# NLI Verifier
# ------------------------------------------------------------------

class NLIVerifier:
    """
    Verifies factual claims against evidence using BART-large-mnli.

    Performs genuine pairwise NLI inference (NOT zero-shot classification).
    The tokenizer encodes:

        premise    = evidence passage
        hypothesis = atomic claim

    as a sequence pair.  The model returns logits over 3 classes:
        index 0 -> contradiction
        index 1 -> neutral
        index 2 -> entailment

    Probabilities are extracted by integer index -- NOT by label string
    lookup -- to guarantee correctness with the confirmed id2label:
        {0: "contradiction", 1: "neutral", 2: "entailment"}

    Usage:
        verifier = NLIVerifier(config)
        result = verifier.verify(qa_result)
        print(result.label, result.css)
    """

    # Class-level cache: model loaded once and shared across instances
    _tokenizer_cache:   Optional[AutoTokenizer] = None
    _model_cache:       Optional[AutoModelForSequenceClassification] = None
    _cached_model_name: Optional[str] = None

    def __init__(self, config: Optional[FrameworkConfig] = None):
        self.config = config or FrameworkConfig()
        self._load_nli_model()
        logger.info("NLIVerifier initialized (pairwise NLI mode).")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def verify(self, qa_result: QualityAssessedRetrieval) -> ClaimVerificationResult:
        """
        Verify a single claim against all of its retrieved evidence.

        For each (evidence_passage, claim) pair, runs pairwise NLI and computes:
            CSS_i = entailment_prob_i * EQS_i

        Final CSS = max(CSS_i) over all evidence items.
        Label assigned using _classify() which distinguishes
        SUPPORTED / CONTRADICTED / UNVERIFIABLE.

        Args:
            qa_result: QualityAssessedRetrieval from Phase 3.

        Returns:
            ClaimVerificationResult with label and CSS.
        """
        claim = qa_result.claim
        logger.debug(f"Verifying claim: '{claim[:60]}'")

        if not qa_result.scored_evidence:
            logger.warning(
                f"No evidence for claim: '{claim[:60]}' -- marking UNVERIFIABLE"
            )
            return ClaimVerificationResult(
                claim=claim,
                label=VerificationLabel.UNVERIFIABLE,
                css=0.0,
            )

        all_nli:  List[NLIResult] = []
        all_css:  List[float]     = []
        best_css  = 0.0
        best_nli  = None
        best_ev   = None
        max_contradiction = 0.0

        for se in qa_result.scored_evidence:
            nli_result = self._run_nli(premise=se.passage, hypothesis=claim)
            css_i = min(1.0, max(0.0, float(nli_result.entailment_prob * se.evidence_quality_score)))

            all_nli.append(nli_result)
            all_css.append(css_i)

            logger.debug(
                f"  NLI: ent={nli_result.entailment_prob:.3f}, "
                f"neu={nli_result.neutral_prob:.3f}, "
                f"con={nli_result.contradiction_prob:.3f}, "
                f"EQS={se.evidence_quality_score:.3f}, CSS={css_i:.3f}"
            )

            if css_i > best_css:
                best_css = css_i
                best_nli = nli_result
                best_ev  = se

            # Track strongest contradiction signal (bounded in [0, 1])
            weighted_contradiction = min(
                1.0, max(0.0, float(nli_result.contradiction_prob * se.evidence_quality_score))
            )
            if weighted_contradiction > max_contradiction:
                max_contradiction = weighted_contradiction

        label = self._classify(best_css, max_contradiction)

        logger.info(
            f"Claim: '{claim[:50]}' -> {label} "
            f"(CSS={best_css:.3f}, max_contra={max_contradiction:.3f})"
        )

        return ClaimVerificationResult(
            claim=claim,
            label=label,
            css=best_css,
            best_nli=best_nli,
            best_evidence=best_ev,
            all_nli=all_nli,
            css_breakdown=all_css,
        )

    def verify_batch(
        self, qa_results: List[QualityAssessedRetrieval]
    ) -> List[ClaimVerificationResult]:
        """Verify a list of claims."""
        results = []
        for i, qa in enumerate(qa_results):
            logger.info(f"Verifying claim {i+1}/{len(qa_results)}")
            results.append(self.verify(qa))
        return results

    # ------------------------------------------------------------------
    # Private: Model Loading
    # ------------------------------------------------------------------

    def _load_nli_model(self) -> None:
        """
        Load the BART-large-mnli tokenizer and model for pairwise NLI.

        Uses class-level cache to avoid reloading on subsequent
        instantiations. The id2label mapping from the model config is:
            {0: "contradiction", 1: "neutral", 2: "entailment"}

        We use integer indices -- not label name strings -- to extract
        probabilities, ensuring correctness even if string labels change.
        """
        model_name = self.config.nli_model_name

        if (
            NLIVerifier._model_cache is not None
            and NLIVerifier._cached_model_name == model_name
        ):
            logger.info(f"Reusing cached NLI model: {model_name}")
            self._tokenizer = NLIVerifier._tokenizer_cache
            self._model     = NLIVerifier._model_cache
            self._device    = self._resolve_device()
            return

        if AutoTokenizer is None or AutoModelForSequenceClassification is None:
            raise RuntimeError(
                "transformers is required for NLIVerifier.\n"
                "Install it with: pip install transformers"
            )

        logger.info(
            f"Loading NLI model: {model_name} "
            f"(device={self.config.nli_device}) -- this may take a minute..."
        )

        self._tokenizer = AutoTokenizer.from_pretrained(model_name)
        self._model = AutoModelForSequenceClassification.from_pretrained(
            model_name
        )

        self._device = self._resolve_device()
        self._model.to(self._device)
        self._model.eval()

        # Log and validate the id2label at load time
        id2label = self._model.config.id2label
        logger.info(f"NLI model id2label: {id2label}")
        # Expected: {0: "contradiction", 1: "neutral", 2: "entailment"}

        expected = {0: "contradiction", 1: "neutral", 2: "entailment"}
        for idx, expected_label in expected.items():
            actual = id2label.get(idx, "")
            if actual.lower() != expected_label:
                logger.warning(
                    f"Unexpected id2label at index {idx}: "
                    f"got {actual!r}, expected {expected_label!r}. "
                    f"Full mapping: {id2label}"
                )

        NLIVerifier._tokenizer_cache   = self._tokenizer
        NLIVerifier._model_cache       = self._model
        NLIVerifier._cached_model_name = model_name
        logger.info("NLI model loaded and cached.")

    def _resolve_device(self) -> torch.device:
        """Return the torch device based on config and hardware availability."""
        if self.config.nli_device == "cuda" and torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")

    # ------------------------------------------------------------------
    # Private: Pairwise NLI Inference
    # ------------------------------------------------------------------

    def _run_nli(self, premise: str, hypothesis: str) -> NLIResult:
        """
        Run pairwise NLI inference using BART-large-mnli.

        The tokenizer encodes the sequence pair and the model returns
        logits over 3 classes indexed as:
            0 -> contradiction
            1 -> neutral
            2 -> entailment

        Softmax converts logits to probabilities extracted by integer index.

        IMPORTANT: Both premise AND hypothesis reach the model as a
        genuine sequence pair.  The claim text is the hypothesis;
        the evidence passage is the premise.  This is verified by the
        NLI unit tests in tests/test_nli_verifier.py.

        Args:
            premise:    Evidence passage (the "fact source").
            hypothesis: The claim being verified.

        Returns:
            NLIResult with probabilities for each label.
        """
        max_chars  = self.config.nli_max_length
        premise    = premise[:max_chars]
        hypothesis = hypothesis[:max_chars // 4]

        try:
            inputs = self._tokenizer(
                premise,
                hypothesis,
                return_tensors="pt",
                truncation=True,
                max_length=1024,
                padding=False,
            )
            inputs = {k: v.to(self._device) for k, v in inputs.items()}

            with torch.no_grad():
                logits = self._model(**inputs).logits  # shape: [1, 3]

            probs = F.softmax(logits, dim=-1).squeeze(0)  # shape: [3]

            # id2label: {0: "contradiction", 1: "neutral", 2: "entailment"}
            contradiction_prob = float(probs[0].item())
            neutral_prob       = float(probs[1].item())
            entailment_prob    = float(probs[2].item())

            return NLIResult(
                entailment_prob=entailment_prob,
                neutral_prob=neutral_prob,
                contradiction_prob=contradiction_prob,
                evidence_passage=premise,
            )

        except Exception as e:
            logger.error(f"NLI inference failed: {e}")
            return NLIResult(
                entailment_prob=0.0,
                neutral_prob=1.0,
                contradiction_prob=0.0,
                evidence_passage=premise,
            )

    # ------------------------------------------------------------------
    # Private: Classification
    # ------------------------------------------------------------------

    def _classify(self, css: float, max_contradiction: float) -> VerificationLabel:
        """
        Map CSS + contradiction signal to a VerificationLabel.

        Logic:
            strong entailment    (CSS >= css_supported)             -> SUPPORTED
            strong contradiction (max_contra >= contradiction_thresh) -> CONTRADICTED
            neither sufficiently strong                              -> UNVERIFIABLE

        A retrieval failure or neutral evidence must NOT automatically
        become a hallucination / contradiction: if evidence is absent or
        weak, both CSS and max_contradiction are low, which maps to UNVERIFIABLE.

        Args:
            css:               Best Claim Support Score across evidence (bounded in [0, 1]).
            max_contradiction: Best weighted contradiction score (bounded in [0, 1]).

        Returns:
            VerificationLabel enum value (SUPPORTED, CONTRADICTED, or UNVERIFIABLE).
        """
        contra_thresh = getattr(
            self.config, "contradiction_threshold", self.config.css_insufficient
        )
        if css >= self.config.css_supported:
            return VerificationLabel.SUPPORTED
        elif max_contradiction >= contra_thresh and css < self.config.css_insufficient:
            return VerificationLabel.CONTRADICTED
        elif max_contradiction >= contra_thresh and max_contradiction > css:
            return VerificationLabel.CONTRADICTED
        else:
            return VerificationLabel.UNVERIFIABLE
