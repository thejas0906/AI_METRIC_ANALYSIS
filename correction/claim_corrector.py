"""
claim_corrector.py  (Phase 5)
==============================
Selective Claim Correction Module
------------------------------------
Corrects ONLY hallucinated claims using evidence retrieved in Phase 2.

The key design principle:
  ✅ Supported claims  → PRESERVED exactly as-is
  ❌ Hallucinated claims → CORRECTED via LLM + evidence grounding
  ⚠️  Insufficient evidence → FLAGGED but not forcibly corrected

Correction Strategy:
--------------------
1. For each HALLUCINATED claim, construct a correction prompt that:
   - Shows the original (hallucinated) claim
   - Provides the top retrieved evidence passages
   - Instructs the LLM to rewrite ONLY this claim using the evidence

2. The corrected claim is verified again in Phase 7 (independent
   verification). If CSS ≥ 0.75, the correction is accepted.
   Otherwise, one more correction attempt is made.

Two correction backends:
    "openai"  → GPT-3.5/4 via OpenAI API (best quality, requires key)
    "local"   → Rule-based fallback (no API required, lower quality)

The local fallback extracts key facts from the evidence passage and
substitutes them into the claim structure — useful for demonstrations
without an API key.

Design Decisions:
-----------------
- We deliberately do NOT modify SUPPORTED claims, even if a
  "better" phrasing exists. This ensures the Claim Preservation
  Rate (CPR) stays at 100% for supported content.
- The correction prompt is carefully engineered to prevent the
  LLM from wandering beyond the scope of the single claim.
- We set temperature=0.3 for deterministic, factual corrections.
"""

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional
from loguru import logger

from config import FrameworkConfig
from verification.nli_verifier import ClaimVerificationResult, VerificationLabel
from verification.evidence_quality import QualityAssessedRetrieval


# ──────────────────────────────────────────────────────────────────
# Enums & Data Structures
# ──────────────────────────────────────────────────────────────────

class CorrectionStatus(str, Enum):
    """
    Outcome of the correction attempt for a single claim.

    PRESERVED:                Claim is SUPPORTED; text is unchanged.
    CORRECTED:                Claim was CONTRADICTED; successfully corrected.
    FLAGGED:                  Claim is UNVERIFIABLE; flagged but not corrected.
                              Do NOT invent a replacement for unverifiable claims.
    FAILED:                   Correction was attempted but exceeded max iterations.
    FINAL_VERIFICATION_FAILED: Phase 7 fresh retrieval failed; cannot independently
                              verify the corrected claim. Distinct from FAILED.
    """
    PRESERVED                  = "PRESERVED"
    CORRECTED                  = "CORRECTED"
    FLAGGED                    = "FLAGGED"
    FAILED                     = "FAILED"
    FINAL_VERIFICATION_FAILED  = "FINAL_VERIFICATION_FAILED"


@dataclass
class CorrectedClaim:
    """
    Result of the selective correction process for one atomic claim.

    Attributes:
        original_claim:    The claim as extracted from the LLM answer.
        corrected_claim:   The rewritten claim (or original if preserved).
        status:            PRESERVED / CORRECTED / FLAGGED / FAILED /
                           FINAL_VERIFICATION_FAILED
        verification_result: Phase 4 verification result.
        evidence_used:     Evidence passage used for correction.
        correction_iterations: Number of correction attempts made.
        was_modified:      True if the claim text actually changed.
    """
    original_claim:        str
    corrected_claim:       str
    status:                CorrectionStatus
    verification_result:   ClaimVerificationResult
    evidence_used:         str = ""
    correction_iterations: int = 0
    was_modified:          bool = False

    def __post_init__(self):
        self.was_modified = self.original_claim != self.corrected_claim


# ──────────────────────────────────────────────────────────────────
# Claim Corrector
# ──────────────────────────────────────────────────────────────────

class ClaimCorrector:
    """
    Selectively corrects hallucinated claims using LLM + evidence.

    Usage:
        corrector = ClaimCorrector(config)
        corrected = corrector.correct(vr, qa_result)
        print(corrected.status, corrected.corrected_claim)
    """

    def __init__(self, config: Optional[FrameworkConfig] = None):
        """
        Initialize the corrector and optionally the OpenAI client.

        Args:
            config: FrameworkConfig instance; uses defaults if None.
        """
        self.config = config or FrameworkConfig()
        self._openai_client = None

        if self.config.correction_backend == "openai":
            self._init_openai()

        logger.info(
            f"ClaimCorrector initialized "
            f"(backend={self.config.correction_backend})"
        )

    # ──────────────────────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────────────────────

    def correct(
        self,
        verification_result: ClaimVerificationResult,
        qa_result: QualityAssessedRetrieval,
    ) -> CorrectedClaim:
        """
        Selectively correct a single claim based on its verification label.

        Logic:
        - SUPPORTED          → preserve as-is (no LLM call)
        - INSUFFICIENT_EVIDENCE → flag but do not forcibly correct
        - HALLUCINATED       → attempt correction with evidence

        Args:
            verification_result: Output from Phase 4 NLI verifier.
            qa_result:           Output from Phase 3 quality assessor.

        Returns:
            CorrectedClaim with status and (possibly) corrected text.
        """
        claim = verification_result.claim
        label = verification_result.label

        if label == VerificationLabel.SUPPORTED:
            # ✅ Claim is supported — preserve exactly
            logger.info(f"PRESERVED: '{claim[:60]}'")
            return CorrectedClaim(
                original_claim=claim,
                corrected_claim=claim,
                status=CorrectionStatus.PRESERVED,
                verification_result=verification_result,
            )

        if label in (VerificationLabel.UNVERIFIABLE, VerificationLabel.INSUFFICIENT_EVIDENCE):
            # ⚠️  Not enough evidence to confirm or deny — flag it
            logger.info(f"FLAGGED (insufficient evidence): '{claim[:60]}'")
            return CorrectedClaim(
                original_claim=claim,
                corrected_claim=claim,
                status=CorrectionStatus.FLAGGED,
                verification_result=verification_result,
            )

        # HALLUCINATED → attempt correction
        logger.info(f"CORRECTING: '{claim[:60]}'")
        return self._attempt_correction(claim, qa_result)

    def correct_batch(
        self,
        verification_results: List[ClaimVerificationResult],
        qa_results: List[QualityAssessedRetrieval],
    ) -> List[CorrectedClaim]:
        """
        Correct a list of claims.

        Args:
            verification_results: List of Phase 4 results.
            qa_results:           Corresponding Phase 3 results.

        Returns:
            List of CorrectedClaim objects.
        """
        assert len(verification_results) == len(qa_results), (
            "verification_results and qa_results must have the same length"
        )

        corrected = []
        for vr, qa in zip(verification_results, qa_results):
            corrected.append(self.correct(vr, qa))
        return corrected

    # ──────────────────────────────────────────────────────────────
    # Correction Logic
    # ──────────────────────────────────────────────────────────────

    def _attempt_correction(
        self,
        claim: str,
        qa_result: QualityAssessedRetrieval,
    ) -> CorrectedClaim:
        """
        Attempt to correct a hallucinated claim using evidence.

        Strategy:
        1. Build an evidence context from the top-K passages.
        2. Call the correction backend (OpenAI or local).
        3. Return the CorrectedClaim.

        Note: Phase 7 independent re-verification happens OUTSIDE
        this module, in the main pipeline (hallucination_pipeline.py).

        Args:
            claim:     The hallucinated claim to correct.
            qa_result: Evidence for this claim.

        Returns:
            CorrectedClaim with CORRECTED or FAILED status.
        """
        # Build evidence context from top evidence passages
        evidence_passages = [
            se.passage for se in qa_result.scored_evidence
        ]
        evidence_context = "\n".join(
            f"[Evidence {i+1}] {p}"
            for i, p in enumerate(evidence_passages[:3])
        )
        evidence_used = evidence_context

        if not evidence_passages:
            logger.warning(f"No evidence available for correction: '{claim[:60]}'")
            return CorrectedClaim(
                original_claim=claim,
                corrected_claim=claim,
                status=CorrectionStatus.FAILED,
                verification_result=ClaimVerificationResult(
                    claim=claim,
                    label=VerificationLabel.CONTRADICTED,
                    css=0.0,
                ),
                evidence_used="",
                correction_iterations=0,
            )

        # Dispatch to appropriate backend
        if self.config.correction_backend == "openai" and self._openai_client:
            corrected_text = self._correct_with_openai(claim, evidence_context)
        else:
            corrected_text = self._correct_with_local(claim, evidence_passages)

        status = (
            CorrectionStatus.CORRECTED
            if corrected_text != claim
            else CorrectionStatus.FAILED
        )

        logger.info(
            f"Correction result: '{corrected_text[:80]}' "
            f"(status={status})"
        )

        return CorrectedClaim(
            original_claim=claim,
            corrected_claim=corrected_text,
            status=status,
            verification_result=ClaimVerificationResult(
                claim=corrected_text,
                label=VerificationLabel.CONTRADICTED,   # re-verified in Phase 7
                css=0.0,
            ),
            evidence_used=evidence_used,
            correction_iterations=1,
        )

    # ──────────────────────────────────────────────────────────────
    # Backend: OpenAI
    # ──────────────────────────────────────────────────────────────

    def _init_openai(self):
        """
        Initialize the OpenAI client if an API key is available.
        Silently falls back to local if the key is missing.
        """
        try:
            import openai
            if self.config.openai_api_key:
                self._openai_client = openai.OpenAI(
                    api_key=self.config.openai_api_key
                )
                logger.info("OpenAI client initialized.")
            else:
                logger.warning(
                    "OPENAI_API_KEY not set — "
                    "falling back to local correction."
                )
        except ImportError:
            logger.warning("openai package not installed — using local backend.")

    def _correct_with_openai(self, claim: str, evidence_context: str) -> str:
        """
        Use GPT-3.5/4 to rewrite a hallucinated claim using evidence.

        Prompt Design:
        - Explicit instruction to rewrite ONLY this one claim.
        - Evidence is provided as grounding context.
        - Temperature=0.3 for factual, deterministic output.
        - max_tokens=150 to prevent over-generation.

        Args:
            claim:            The hallucinated claim to correct.
            evidence_context: Top evidence passages as a string.

        Returns:
            Corrected claim string from GPT.
        """
        prompt = (
            "You are a factual accuracy editor. Your task is to rewrite "
            "ONE factual claim to make it accurate based on the provided evidence.\n\n"
            "RULES:\n"
            "1. Rewrite ONLY the given claim — do not add information beyond the evidence.\n"
            "2. Keep the corrected claim concise (1 sentence).\n"
            "3. Do NOT include quotes, explanations, or extra commentary.\n"
            "4. If the evidence does not support a correction, return the claim unchanged.\n\n"
            f"ORIGINAL CLAIM:\n{claim}\n\n"
            f"EVIDENCE:\n{evidence_context}\n\n"
            "CORRECTED CLAIM:"
        )

        try:
            response = self._openai_client.chat.completions.create(
                model=self.config.openai_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=150,
            )
            corrected = response.choices[0].message.content.strip()
            # Ensure it ends with a period
            if corrected and not corrected.endswith("."):
                corrected += "."
            return corrected if corrected else claim

        except Exception as e:
            logger.error(f"OpenAI correction failed: {e}")
            return self._correct_with_local(claim, [evidence_context])

    # ──────────────────────────────────────────────────────────────
    # Backend: Local (Rule-Based Fallback)
    # ──────────────────────────────────────────────────────────────

    def _correct_with_local(
        self, claim: str, evidence_passages: List[str]
    ) -> str:
        """
        Rule-based claim correction as a fallback when no API is available.

        Strategy:
        1. Find the most relevant sentence in evidence.
        2. Extract key numerical or named-entity facts.
        3. Substitute detected errors in the claim.

        This is a heuristic approach — quality is lower than LLM correction
        but demonstrates the pipeline works end-to-end without an API key.

        Args:
            claim:             The hallucinated claim.
            evidence_passages: List of relevant evidence passage strings.

        Returns:
            Heuristically corrected claim string.
        """
        if not evidence_passages:
            return claim

        # Use the first (most relevant) evidence passage
        best_evidence = evidence_passages[0]

        # Strategy: if the best evidence passage is factually dense,
        # use it as the corrected claim directly (as a last resort)
        # In a real system, this would use entity substitution.
        corrected = self._substitute_facts(claim, best_evidence)
        return corrected

    @staticmethod
    def _substitute_facts(claim: str, evidence: str) -> str:
        """
        Attempt simple fact substitution:
        - Replace years/numbers in the claim with those found in evidence
          if they differ.
        - This is a DEMO-quality heuristic — not production-ready.

        Args:
            claim:    Original hallucinated claim.
            evidence: Evidence passage with (presumably) correct facts.

        Returns:
            Claim with substituted facts, or original if no substitution.
        """
        # Extract 4-digit years from claim and evidence
        claim_years    = re.findall(r"\b(1[0-9]{3}|20[0-9]{2})\b", claim)
        evidence_years = re.findall(r"\b(1[0-9]{3}|20[0-9]{2})\b", evidence)

        result = claim

        if claim_years and evidence_years and claim_years[0] != evidence_years[0]:
            # Substitute the first year found in the claim
            result = result.replace(claim_years[0], evidence_years[0], 1)
            logger.debug(
                f"Year substitution: {claim_years[0]} → {evidence_years[0]}"
            )

        # If no substitution was made, return a truncated evidence sentence
        # as a fallback (better than returning an unchanged hallucination)
        if result == claim and len(evidence) > 20:
            # Extract first sentence of evidence as the corrected claim
            first_sentence = evidence.split(".")[0].strip() + "."
            if len(first_sentence) > 15:
                return first_sentence

        return result
