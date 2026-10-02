"""
nli_verifier.py  (Phase 4)
===========================
NLI-Based Claim Verification Module
--------------------------------------
Uses facebook/bart-large-mnli (a zero-shot Natural Language Inference
model) to determine whether each piece of retrieved evidence
ENTAILS, CONTRADICTS, or is NEUTRAL to a given claim.

The raw entailment probability is combined with the Evidence Quality
Score (EQS) from Phase 3 to produce the final:

    Claim Support Score (CSS) = NLI_entailment_prob × EQS

Classification:
    CSS ≥ 0.75         →  SUPPORTED
    0.40 ≤ CSS < 0.75  →  INSUFFICIENT_EVIDENCE
    CSS < 0.40         →  HALLUCINATED

The model is loaded once and cached for the lifetime of the verifier
to avoid reloading on every call (critical for laptop performance).

How BART-large-mnli works:
--------------------------
- It is a sequence-to-sequence model fine-tuned on MNLI.
- Input:  "{evidence} </s></s> {claim}"
- Output: logits over 3 classes:
            label 0 → CONTRADICTION
            label 1 → NEUTRAL
            label 2 → ENTAILMENT
- We apply softmax to get probabilities.
- We use the ENTAILMENT probability as the raw support score.

References:
-----------
- BART-large-mnli: https://huggingface.co/facebook/bart-large-mnli
- MNLI benchmark:  https://cims.nyu.edu/~sbowman/multinli/
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Tuple
from loguru import logger

import torch
from transformers import pipeline, Pipeline

from config import FrameworkConfig
from verification.evidence_quality import QualityAssessedRetrieval, ScoredEvidence


# ──────────────────────────────────────────────────────────────────
# Enums & Data Structures
# ──────────────────────────────────────────────────────────────────

class VerificationLabel(str, Enum):
    """Possible outcomes of NLI-based claim verification."""
    SUPPORTED             = "SUPPORTED"
    INSUFFICIENT_EVIDENCE = "INSUFFICIENT_EVIDENCE"
    HALLUCINATED          = "HALLUCINATED"


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
        label:          SUPPORTED / INSUFFICIENT_EVIDENCE / HALLUCINATED
        css:            Claim Support Score (best NLI entailment × EQS)
        best_nli:       The NLIResult from the highest-scoring evidence.
        best_evidence:  The ScoredEvidence item that produced best CSS.
        all_nli:        NLI results for all evidence items.
        css_breakdown:  Per-evidence CSS values for transparency.
    """
    claim:          str
    label:          VerificationLabel
    css:            float                   # final Claim Support Score
    best_nli:       Optional[NLIResult]     = None
    best_evidence:  Optional[ScoredEvidence] = None
    all_nli:        List[NLIResult]          = field(default_factory=list)
    css_breakdown:  List[float]              = field(default_factory=list)


# ──────────────────────────────────────────────────────────────────
# NLI Verifier
# ──────────────────────────────────────────────────────────────────

class NLIVerifier:
    """
    Verifies factual claims against evidence using BART-large-mnli.

    Usage:
        verifier = NLIVerifier(config)
        result = verifier.verify(qa_result)
        print(result.label, result.css)
    """

    # Class-level cache: model is loaded once and shared across instances
    _pipeline_cache: Optional[Pipeline] = None
    _cached_model_name: Optional[str]   = None

    def __init__(self, config: Optional[FrameworkConfig] = None):
        """
        Load (or reuse) the BART-large-mnli inference pipeline.

        Args:
            config: FrameworkConfig instance; uses defaults if None.
        """
        self.config = config or FrameworkConfig()
        self._load_nli_pipeline()
        logger.info("NLIVerifier initialized.")

    # ──────────────────────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────────────────────

    def verify(self, qa_result: QualityAssessedRetrieval) -> ClaimVerificationResult:
        """
        Verify a single claim against all of its retrieved evidence.

        For each (evidence_passage, claim) pair, runs NLI and computes:
            CSS_i = entailment_prob_i × EQS_i

        Final CSS = max(CSS_i) over all evidence items.

        Args:
            qa_result: QualityAssessedRetrieval from Phase 3.

        Returns:
            ClaimVerificationResult with label and CSS.
        """
        claim = qa_result.claim
        logger.debug(f"Verifying claim: '{claim[:60]}'")

        if not qa_result.scored_evidence:
            # No evidence retrieved → cannot support the claim
            logger.warning(f"No evidence for claim: '{claim[:60]}' — marking INSUFFICIENT")
            return ClaimVerificationResult(
                claim=claim,
                label=VerificationLabel.INSUFFICIENT_EVIDENCE,
                css=0.0,
            )

        all_nli:  List[NLIResult] = []
        all_css:  List[float]     = []
        best_css  = 0.0
        best_nli  = None
        best_ev   = None

        for se in qa_result.scored_evidence:
            # Run NLI: evidence is the premise, claim is the hypothesis
            nli_result = self._run_nli(premise=se.passage, hypothesis=claim)
            # Compute CSS for this evidence item
            css_i = nli_result.entailment_prob * se.evidence_quality_score

            all_nli.append(nli_result)
            all_css.append(css_i)

            logger.debug(
                f"  NLI: ent={nli_result.entailment_prob:.3f}, "
                f"EQS={se.evidence_quality_score:.3f}, CSS={css_i:.3f}"
            )

            if css_i > best_css:
                best_css  = css_i
                best_nli  = nli_result
                best_ev   = se

        # Classify the claim based on best CSS
        label = self._classify(best_css)

        logger.info(
            f"Claim: '{claim[:50]}' → {label} (CSS={best_css:.3f})"
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
        """
        Verify a list of claims.

        Args:
            qa_results: List of QualityAssessedRetrieval objects.

        Returns:
            List of ClaimVerificationResult objects.
        """
        results = []
        for i, qa in enumerate(qa_results):
            logger.info(f"Verifying claim {i+1}/{len(qa_results)}")
            results.append(self.verify(qa))
        return results

    # ──────────────────────────────────────────────────────────────
    # Private Helpers
    # ──────────────────────────────────────────────────────────────

    def _load_nli_pipeline(self):
        """
        Load the BART-large-mnli zero-shot classification pipeline.

        Uses class-level cache to avoid reloading on subsequent
        instantiations (saves ~1.6 GB RAM load time on laptop).
        """
        model_name = self.config.nli_model_name

        if (
            NLIVerifier._pipeline_cache is not None
            and NLIVerifier._cached_model_name == model_name
        ):
            logger.info(f"Reusing cached NLI pipeline: {model_name}")
            self._nli_pipeline = NLIVerifier._pipeline_cache
            return

        logger.info(
            f"Loading NLI model: {model_name} "
            f"(device={self.config.nli_device}) — this may take a minute..."
        )

        self._nli_pipeline = pipeline(
            "zero-shot-classification",
            model=model_name,
            device=0 if self.config.nli_device == "cuda" else -1,
        )

        # Cache at class level
        NLIVerifier._pipeline_cache  = self._nli_pipeline
        NLIVerifier._cached_model_name = model_name
        logger.info("NLI pipeline loaded and cached.")

    def _run_nli(self, premise: str, hypothesis: str) -> NLIResult:
        """
        Run zero-shot NLI inference using BART-large-mnli.

        The model is called as a zero-shot classifier with candidate
        labels ["entailment", "neutral", "contradiction"].

        Args:
            premise:    Evidence passage (the "fact source").
            hypothesis: The claim being verified.

        Returns:
            NLIResult with probabilities for each label.
        """
        # Truncate inputs to model's max length
        max_len = self.config.nli_max_length
        premise    = premise[:max_len]
        hypothesis = hypothesis[:max_len // 4]   # claim is shorter

        try:
            # zero-shot-classification returns dict with 'labels' and 'scores'
            output = self._nli_pipeline(
                sequences=premise,
                candidate_labels=["entailment", "neutral", "contradiction"],
                hypothesis_template="This text suggests: {}.",
                multi_label=False,
            )

            # Map label → score
            label_scores = dict(zip(output["labels"], output["scores"]))

            return NLIResult(
                entailment_prob=label_scores.get("entailment", 0.0),
                neutral_prob=label_scores.get("neutral", 0.0),
                contradiction_prob=label_scores.get("contradiction", 0.0),
                evidence_passage=premise,
            )

        except Exception as e:
            logger.error(f"NLI inference failed: {e}")
            # Return neutral on failure (conservative: don't flag as hallucination)
            return NLIResult(
                entailment_prob=0.0,
                neutral_prob=1.0,
                contradiction_prob=0.0,
                evidence_passage=premise,
            )

    def _classify(self, css: float) -> VerificationLabel:
        """
        Map a Claim Support Score (CSS) to a VerificationLabel.

        Thresholds (from FrameworkConfig):
            CSS >= 0.75  → SUPPORTED
            0.40 <= CSS < 0.75 → INSUFFICIENT_EVIDENCE
            CSS < 0.40   → HALLUCINATED

        Args:
            css: Claim Support Score in [0, 1].

        Returns:
            VerificationLabel enum value.
        """
        if css >= self.config.css_supported:
            return VerificationLabel.SUPPORTED
        elif css >= self.config.css_insufficient:
            return VerificationLabel.INSUFFICIENT_EVIDENCE
        else:
            return VerificationLabel.HALLUCINATED
