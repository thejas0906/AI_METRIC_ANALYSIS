"""
hallucination_pipeline.py
==========================
Main Pipeline Orchestrator
---------------------------
Ties all 7 phases of the Selective Evidence-Guided Hallucination
Correction Framework into a single, end-to-end pipeline.

Full Pipeline:
  User Query
      |
  [Phase 0] LLM Response (provided as input)
      |
  [Phase 1] Claim Extraction       (ClaimExtractor -> List[ClaimSpan])
      |
  [Phase 2] Evidence Retrieval     (EvidenceRetriever)
      |
  [Phase 3] Quality Assessment     (EvidenceQualityAssessor)
      |
  [Phase 4] NLI Verification       (NLIVerifier)
      |
  [Phase 5] Selective Correction   (ClaimCorrector)
      |
  [Phase 6] Response Reconstruction (ResponseReconstructor -- span surgery)
      |
  [Phase 7] Independent Verification (fresh retrieval + NLI)
      |
  Final Verified Response

Central research contribution:
    MINIMUM NECESSARY CORRECTION -- only contradicted claim spans are
    replaced; supported text is preserved byte-for-byte.

Phase 7 (Independent Verification):
    After correction, each CORRECTED claim is re-verified with FRESHLY
    retrieved evidence.  The original Phase-1 evidence is NOT reused
    because the corrected claim may reference different facts.
    If fresh retrieval fails, the claim is marked
    FINAL_VERIFICATION_FAILED rather than silently falling back.

Design Decisions:
-----------------
- The pipeline is implemented as a class with a single `.run()` method.
- All intermediate results are stored in `PipelineResult`.
- Component initialization is done once in __init__ for efficiency.
- Logging is verbose by default to aid research reproducibility.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from loguru import logger
from tqdm import tqdm

from config import FrameworkConfig
from retrieval.claim_extractor import ClaimExtractor, ClaimSpan
from retrieval.evidence_retriever import EvidenceRetriever, RetrievalResult
from verification.evidence_quality import EvidenceQualityAssessor, QualityAssessedRetrieval
from verification.nli_verifier import NLIVerifier, ClaimVerificationResult, VerificationLabel
from correction.claim_corrector import ClaimCorrector, CorrectedClaim, CorrectionStatus
from correction.response_reconstructor import ResponseReconstructor, ReconstructedResponse
from evaluation.metrics import EvaluationMetrics, CorrectionOutcome


# ------------------------------------------------------------------
# Pipeline Result Data Structure
# ------------------------------------------------------------------

@dataclass
class PipelineResult:
    """
    Complete result of running the hallucination correction pipeline
    on a single LLM response.

    Attributes:
        original_response:       The raw LLM output.
        claim_spans:             Span-aware claim objects from Phase 1.
        extracted_claims:        Plain claim strings (for backward compat).
        retrieval_results:       Evidence per claim from Phase 2.
        quality_results:         Scored evidence from Phase 3.
        verification_results:    Initial NLI results from Phase 4.
        corrected_claims:        Correction outcomes from Phase 5.
        reconstructed_response:  Final merged response from Phase 6.
        final_verification:      Re-verification results from Phase 7.
        final_response:          The final verified response text.
        pipeline_metadata:       Timing, stats, and configuration info.
    """
    original_response:       str
    claim_spans:             List[ClaimSpan]                = field(default_factory=list)
    extracted_claims:        List[str]                      = field(default_factory=list)
    retrieval_results:       List[RetrievalResult]          = field(default_factory=list)
    quality_results:         List[QualityAssessedRetrieval] = field(default_factory=list)
    verification_results:    List[ClaimVerificationResult]  = field(default_factory=list)
    corrected_claims:        List[CorrectedClaim]           = field(default_factory=list)
    reconstructed_response:  Optional[ReconstructedResponse] = None
    final_verification:      List[ClaimVerificationResult]  = field(default_factory=list)
    final_response:          str                            = ""
    pipeline_metadata:       Dict[str, Any]                 = field(default_factory=dict)


# ------------------------------------------------------------------
# Main Pipeline Class
# ------------------------------------------------------------------

class HallucinationCorrectionPipeline:
    """
    End-to-end Selective Evidence-Guided Hallucination Correction.

    Usage:
        pipeline = HallucinationCorrectionPipeline(config)
        result = pipeline.run(llm_response, query="What is the capital of France?")
        print(result.final_response)
        print(result.reconstructed_response.correction_report)
    """

    def __init__(self, config: Optional[FrameworkConfig] = None):
        """
        Initialize all pipeline components.

        The NLI model (BART-large-mnli) is loaded here once and
        reused across all pipeline calls via class-level caching
        in NLIVerifier.

        Args:
            config: FrameworkConfig instance; uses defaults if None.
        """
        self.config = config or FrameworkConfig()

        logger.info("=" * 55)
        logger.info("  Initializing Hallucination Correction Pipeline")
        logger.info("=" * 55)

        # Phase 1: Claim Extraction (returns ClaimSpan objects)
        self.claim_extractor = ClaimExtractor(self.config)

        # Phase 2: Evidence Retrieval
        self.evidence_retriever = EvidenceRetriever(self.config)

        # Phase 3: Evidence Quality Assessment
        self.quality_assessor = EvidenceQualityAssessor(self.config)

        # Phase 4 & 7: NLI Verification (shared instance -> shared model cache)
        self.nli_verifier = NLIVerifier(self.config)

        # Phase 5: Selective Claim Correction
        self.claim_corrector = ClaimCorrector(self.config)

        # Phase 6: Response Reconstruction
        self.reconstructor = ResponseReconstructor()

        logger.info("Pipeline initialized. Ready to process responses.")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(
        self,
        llm_response: str,
        query: str = "",
        verbose: bool = True,
    ) -> PipelineResult:
        """
        Run the full 7-phase hallucination correction pipeline.

        Args:
            llm_response: The raw text output from an LLM.
            query:        The original user query (for context; optional).
            verbose:      If True, print phase headers to stdout.

        Returns:
            PipelineResult with all intermediate and final outputs.
        """
        import time
        start_time = time.time()

        result = PipelineResult(original_response=llm_response)
        result.pipeline_metadata = {
            "query": query,
            "model": self.config.nli_model_name,
            "css_supported_threshold":    self.config.css_supported,
            "css_insufficient_threshold": self.config.css_insufficient,
        }

        if verbose:
            self._print_phase("PHASE 1: CLAIM EXTRACTION")

        # -- Phase 1: Claim Extraction (span-aware) -------------------
        result.claim_spans = self.claim_extractor.extract_spans(llm_response)
        result.extracted_claims = [cs.text for cs in result.claim_spans]
        logger.info(f"Phase 1 complete: {len(result.extracted_claims)} claims extracted.")

        if not result.extracted_claims:
            logger.warning("No claims extracted -- returning original response.")
            result.final_response = llm_response
            return result

        # -- Phase 2: Evidence Retrieval ------------------------------
        if verbose:
            self._print_phase("PHASE 2: EVIDENCE RETRIEVAL")

        self.evidence_retriever.set_context(response=llm_response, topic=query)
        result.retrieval_results = []
        for claim_text in tqdm(
            result.extracted_claims, desc="Retrieving evidence", disable=not verbose
        ):
            rr = self.evidence_retriever.retrieve(
                claim_text, context=llm_response, topic=query
            )
            result.retrieval_results.append(rr)

        logger.info("Phase 2 complete: evidence retrieved for all claims.")

        # -- Phase 3: Evidence Quality Assessment ---------------------
        if verbose:
            self._print_phase("PHASE 3: EVIDENCE QUALITY ASSESSMENT")

        result.quality_results = self.quality_assessor.assess_batch(
            result.retrieval_results
        )
        logger.info("Phase 3 complete: evidence quality scores computed.")

        # -- Phase 4: NLI Verification --------------------------------
        if verbose:
            self._print_phase("PHASE 4: NLI VERIFICATION")

        result.verification_results = self.nli_verifier.verify_batch(
            result.quality_results
        )
        logger.info("Phase 4 complete: claims classified by NLI.")

        if verbose:
            self._print_verification_summary(result.verification_results)

        # -- Phase 5: Selective Correction ----------------------------
        if verbose:
            self._print_phase("PHASE 5: SELECTIVE CLAIM CORRECTION")

        result.corrected_claims = self.claim_corrector.correct_batch(
            result.verification_results,
            result.quality_results,
        )
        logger.info("Phase 5 complete: hallucinated claims corrected.")

        # -- Phase 6: Response Reconstruction (span surgery) ----------
        if verbose:
            self._print_phase("PHASE 6: RESPONSE RECONSTRUCTION")

        result.reconstructed_response = self.reconstructor.reconstruct(
            corrected_claims=result.corrected_claims,
            claim_spans=result.claim_spans,
            annotate=False,
            original_response=llm_response,
        )
        logger.info("Phase 6 complete: response reconstructed via span surgery.")

        # -- Phase 7: Independent Verification (fresh retrieval) ------
        if verbose:
            self._print_phase("PHASE 7: INDEPENDENT VERIFICATION (FRESH)")

        result.final_verification, result.corrected_claims = (
            self._independent_verification(
                result.corrected_claims,
                result.quality_results,
            )
        )

        # Rebuild final response after Phase 7 adjustments
        final_reconstructed = self.reconstructor.reconstruct(
            corrected_claims=result.corrected_claims,
            claim_spans=result.claim_spans,
            annotate=False,
            original_response=llm_response,
        )
        result.reconstructed_response = final_reconstructed
        result.final_response = final_reconstructed.final_text

        elapsed = time.time() - start_time
        result.pipeline_metadata["elapsed_seconds"] = round(elapsed, 2)

        if verbose:
            self._print_final_summary(result)

        logger.info(
            f"Pipeline complete in {elapsed:.1f}s. "
            f"Final response: {len(result.final_response)} chars."
        )

        return result

    # ------------------------------------------------------------------
    # Phase 7: Independent Verification (EXACTLY ONE DEFINITION)
    # ------------------------------------------------------------------

    def _independent_verification(
        self,
        corrected_claims: List[CorrectedClaim],
        qa_results: List[QualityAssessedRetrieval],
    ) -> tuple:
        """
        Phase 7: Re-verify CORRECTED claims with FRESH evidence retrieval.

        This is the only definition of _independent_verification in this
        class.  The method performs genuinely independent verification:

        1. For each CORRECTED claim, fresh evidence is retrieved from
           Wikipedia using the corrected claim text as the query.
        2. Evidence quality is re-assessed on the fresh evidence.
        3. Pairwise NLI is run on the corrected claim vs fresh evidence.
        4. If CSS >= css_supported: correction is accepted.
        5. If CSS < css_supported and iterations remain: retry correction.
        6. If max iterations exhausted: revert to original claim.

        When fresh retrieval fails (network error, rate limit, etc.), the
        claim is marked with status FINAL_VERIFICATION_FAILED and the
        original claim text is preserved.  This failure is NOT silently
        treated as a successful verification.

        Preserved and flagged claims are NOT re-verified (they already
        passed or were never corrected).

        Args:
            corrected_claims: List of CorrectedClaim from Phase 5.
            qa_results:       Phase-1 quality-assessed evidence (kept for
                              retry context only; NOT used for verification).

        Returns:
            Tuple of (final_verification_results, updated_corrected_claims)
        """
        from verification.evidence_quality import QualityAssessedRetrieval as QAR

        final_verifications: List[ClaimVerificationResult] = []
        updated_corrected: List[CorrectedClaim] = list(corrected_claims)

        for i, cc in enumerate(corrected_claims):
            if cc.status != CorrectionStatus.CORRECTED:
                # Preserved and flagged claims skip re-verification.
                placeholder = ClaimVerificationResult(
                    claim=cc.corrected_claim,
                    label=(
                        VerificationLabel.SUPPORTED
                        if cc.status == CorrectionStatus.PRESERVED
                        else VerificationLabel.UNVERIFIABLE
                    ),
                    css=1.0 if cc.status == CorrectionStatus.PRESERVED else 0.0,
                )
                final_verifications.append(placeholder)
                continue

            # Phase 7: Fresh retrieval for the corrected claim text
            logger.info(
                f"Phase 7: Fresh retrieval for claim "
                f"[{i+1}/{len(corrected_claims)}]: '{cc.corrected_claim[:60]}'"
            )

            fresh_retrieval_succeeded = False
            try:
                fresh_retrieval = self.evidence_retriever.retrieve(cc.corrected_claim)
                fresh_qa = self.quality_assessor.assess(fresh_retrieval)
                fresh_retrieval_succeeded = bool(fresh_qa.scored_evidence)
            except Exception as e:
                logger.warning(
                    f"Phase 7: Fresh retrieval FAILED ({e}). "
                    f"Marking as FINAL_VERIFICATION_FAILED."
                )

            if not fresh_retrieval_succeeded:
                # Do NOT silently fall back to stale evidence.
                # Mark as failed and preserve the corrected text as-is.
                from correction.claim_corrector import CorrectionStatus as CS
                updated_corrected[i].status = CS.FINAL_VERIFICATION_FAILED
                placeholder = ClaimVerificationResult(
                    claim=cc.corrected_claim,
                    label=VerificationLabel.UNVERIFIABLE,
                    css=0.0,
                )
                final_verifications.append(placeholder)
                logger.warning(
                    f"Phase 7: FINAL_VERIFICATION_FAILED for "
                    f"'{cc.corrected_claim[:60]}'"
                )
                continue

            re_verification = self.nli_verifier.verify(fresh_qa)
            final_verifications.append(re_verification)

            if re_verification.css >= self.config.css_supported:
                logger.info(
                    f"Phase 7: Correction ACCEPTED "
                    f"(CSS={re_verification.css:.3f}): '{cc.corrected_claim[:60]}'"
                )

            elif cc.correction_iterations < self.config.max_correction_iterations:
                logger.info(
                    f"Phase 7: CSS too low ({re_verification.css:.3f}), "
                    f"retrying correction with fresh evidence..."
                )
                retry = self.claim_corrector.correct(
                    re_verification, fresh_qa
                )
                retry.correction_iterations = cc.correction_iterations + 1
                updated_corrected[i] = retry

            else:
                # Max iterations reached -- revert to original claim
                logger.warning(
                    f"Phase 7: Correction FAILED after "
                    f"{cc.correction_iterations} attempts. "
                    f"Reverting to original: '{cc.original_claim[:60]}'"
                )
                from correction.claim_corrector import CorrectionStatus as CS
                updated_corrected[i].status = CS.FAILED
                updated_corrected[i].corrected_claim = cc.original_claim

        logger.info("Phase 7 complete: independent fresh-evidence verification done.")
        return final_verifications, updated_corrected

    # ------------------------------------------------------------------
    # Batch Evaluation Mode
    # ------------------------------------------------------------------

    def evaluate(
        self,
        samples: list,
        verbose: bool = True,
    ) -> Dict[str, Any]:
        """
        Run the pipeline on a list of DatasetSample or ResponseEvaluationSample objects
        and compute evaluation metrics.

        The primary unit of evaluation is a full LLM response containing multiple
        factual claims, which are each verified and selectively corrected. The final
        reconstructed response preserves unaffected sections verbatim.

        Args:
            samples:  List of DatasetSample or ResponseEvaluationSample objects.
            verbose:  If True, show progress bar.

        Returns:
            Dict with 'metrics' (MetricsResult) and 'pipeline_outputs'.
        """
        from evaluation.metrics import EvaluationMetrics, CorrectionOutcome

        pipeline_outputs = []
        correction_outcomes = []
        response_reports = []

        for sample_idx, sample in enumerate(tqdm(samples, desc="Evaluating", disable=not verbose)):
            response_text = getattr(sample, "response", sample.claim)
            query_text = getattr(sample, "question", getattr(sample, "context", ""))
            result = self.run(response_text, query=query_text, verbose=False)

            # Resolve response ID
            resp_id = getattr(sample, "response_id", "")
            if not resp_id and hasattr(sample, "metadata") and isinstance(sample.metadata, dict):
                resp_id = sample.metadata.get("response_id", "")
            if not resp_id:
                resp_id = f"resp_{sample_idx + 1:03d}"

            # Response-level counts
            total_claims = len(result.extracted_claims)
            supported_count = sum(
                1 for vr in result.verification_results
                if vr.label == VerificationLabel.SUPPORTED
            )
            contradicted_count = sum(
                1 for vr in result.verification_results
                if vr.label == VerificationLabel.CONTRADICTED
            )
            unverifiable_count = sum(
                1 for vr in result.verification_results
                if vr.label == VerificationLabel.UNVERIFIABLE
            )
            corrected_count = sum(
                1 for cc in result.corrected_claims
                if cc.status == CorrectionStatus.CORRECTED
            )
            preserved_count = sum(
                1 for cc in result.corrected_claims
                if cc.status == CorrectionStatus.PRESERVED
            )
            failed_count = sum(
                1 for cc in result.corrected_claims
                if cc.status in (CorrectionStatus.FAILED, CorrectionStatus.FINAL_VERIFICATION_FAILED)
            )

            # Calculate response-level CPR, UMR, RPS
            cpr = (preserved_count / supported_count) if supported_count > 0 else 1.0
            umr = (1.0 - cpr) if supported_count > 0 else 0.0
            rps = getattr(result.reconstructed_response, "rps", 1.0) if result.reconstructed_response else 1.0

            response_reports.append({
                "response_id":          resp_id,
                "query":                query_text,
                "total_claims":         total_claims,
                "supported_claims":     supported_count,
                "contradicted_claims":  contradicted_count,
                "unverifiable_claims":  unverifiable_count,
                "corrected_claims":     corrected_count,
                "preserved_claims":     preserved_count,
                "failed_corrections":   failed_count,
                "cpr":                  cpr,
                "umr":                  umr,
                "rps":                  rps,
                "original_response":    response_text,
                "final_response":       result.final_response,
            })

            gold_claims = getattr(sample, "gold_claims", None)
            if gold_claims:
                # Multi-claim response evaluation: evaluate every extracted claim
                for i, span in enumerate(result.claim_spans):
                    vr = result.verification_results[i] if i < len(result.verification_results) else None
                    cc = result.corrected_claims[i] if i < len(result.corrected_claims) else None
                    fv = result.final_verification[i] if i < len(result.final_verification) else None

                    # Match with gold claim by text overlap or positional index
                    matched_gold = None
                    for gc in gold_claims:
                        if (
                            gc.claim_text.lower().rstrip(".") in span.text.lower().rstrip(".")
                            or span.text.lower().rstrip(".") in gc.claim_text.lower().rstrip(".")
                        ):
                            matched_gold = gc
                            break
                    if not matched_gold and i < len(gold_claims):
                        matched_gold = gold_claims[i]

                    gold_is_contradicted = (
                        (matched_gold.gold_label == "CONTRADICTED") if matched_gold else False
                    )
                    gold_is_supported = (
                        (matched_gold.gold_label == "SUPPORTED") if matched_gold else True
                    )

                    pred_contradicted = (
                        (vr.label == VerificationLabel.CONTRADICTED) if vr else False
                    )
                    was_corrected = (
                        (cc.status == CorrectionStatus.CORRECTED) if cc else False
                    )
                    was_preserved = (
                        (cc.status == CorrectionStatus.PRESERVED) if cc else True
                    )
                    was_unverifiable = (
                        (vr.label == VerificationLabel.UNVERIFIABLE) if vr else False
                    )

                    post_verified = (
                        (fv.css >= self.config.css_supported) if fv else False
                    )

                    is_gt_correct = None
                    if matched_gold and matched_gold.gold_correction and cc:
                        cand = cc.corrected_claim.strip().lower().rstrip(".")
                        tgt = matched_gold.gold_correction.strip().lower().rstrip(".")
                        is_gt_correct = (cand == tgt) or (tgt in cand) or (cand in tgt)

                    outcome = CorrectionOutcome(
                        claim=span.text,
                        ground_truth_label=gold_is_contradicted,
                        predicted_label=pred_contradicted,
                        was_corrected=was_corrected,
                        post_correction_verified=post_verified,
                        was_preserved=was_preserved,
                        original_was_supported=gold_is_supported,
                        was_unverifiable=was_unverifiable,
                        gold_label=matched_gold.gold_label if matched_gold else "",
                        gold_correction=matched_gold.gold_correction if matched_gold else None,
                        is_ground_truth_correct=is_gt_correct,
                    )
                    correction_outcomes.append(outcome)

            else:
                # Single-claim or legacy sample format
                items_to_eval = result.claim_spans if result.claim_spans else [None]
                for i, span in enumerate(items_to_eval):
                    vr = (
                        result.verification_results[i]
                        if (result.verification_results and i < len(result.verification_results))
                        else None
                    )
                    cc = (
                        result.corrected_claims[i]
                        if (result.corrected_claims and i < len(result.corrected_claims))
                        else None
                    )
                    fv = (
                        result.final_verification[i]
                        if (result.final_verification and i < len(result.final_verification))
                        else None
                    )

                    pred_contradicted = (
                        (vr.label == VerificationLabel.CONTRADICTED) if vr else False
                    )
                    was_corrected = (
                        (cc.status == CorrectionStatus.CORRECTED) if cc else False
                    )
                    was_preserved = (
                        (cc.status == CorrectionStatus.PRESERVED) if cc else True
                    )
                    was_unverifiable = (
                        (vr.label == VerificationLabel.UNVERIFIABLE) if vr else False
                    )
                    post_verified = (
                        (fv.css >= self.config.css_supported) if fv else False
                    )

                    outcome = CorrectionOutcome(
                        claim=span.text if span else sample.claim,
                        ground_truth_label=sample.ground_truth,
                        predicted_label=pred_contradicted,
                        was_corrected=was_corrected,
                        post_correction_verified=post_verified,
                        was_preserved=was_preserved,
                        original_was_supported=not sample.ground_truth,
                        was_unverifiable=was_unverifiable,
                    )
                    correction_outcomes.append(outcome)

            pipeline_outputs.append({
                "response_id":       resp_id,
                "original_response": response_text,
                "final_response":    result.final_response,
                "claims_extracted":  result.extracted_claims,
                "supported_count":   supported_count,
                "contradicted_count": contradicted_count,
                "unverifiable_count": unverifiable_count,
                "corrected_count":   corrected_count,
                "preserved_count":   preserved_count,
                "failed_count":      failed_count,
            })

        metrics_calculator = EvaluationMetrics()
        ground_truth = [o.ground_truth_label for o in correction_outcomes]
        predictions  = [o.predicted_label    for o in correction_outcomes]
        metrics = metrics_calculator.compute(
            ground_truth, predictions, correction_outcomes, response_reports=response_reports
        )

        return {
            "metrics":             metrics,
            "pipeline_outputs":    pipeline_outputs,
            "response_reports":    response_reports,
            "correction_outcomes": correction_outcomes,
        }

    # ------------------------------------------------------------------
    # Display Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _print_phase(title: str) -> None:
        """Print a formatted phase header."""
        print(f"\n{'-' * 55}")
        print(f"  {title}")
        print(f"{'-' * 55}")

    @staticmethod
    def _print_verification_summary(
        verification_results: List[ClaimVerificationResult],
    ) -> None:
        """Print a summary of Phase 4 verification results."""
        labels = [vr.label.value for vr in verification_results]
        supported    = labels.count("SUPPORTED")
        unverifiable = labels.count("UNVERIFIABLE")
        contradicted = labels.count("CONTRADICTED")
        total = len(labels)

        print(f"\n  Verification Summary ({total} claims):")
        print(f"  SUPPORTED:    {supported}")
        print(f"  UNVERIFIABLE: {unverifiable}")
        print(f"  CONTRADICTED: {contradicted}")

    @staticmethod
    def _print_final_summary(result: PipelineResult) -> None:
        """Print the final pipeline summary."""
        print(f"\n{'=' * 55}")
        print("  PIPELINE COMPLETE")
        print(f"{'=' * 55}")
        print(f"  Claims Extracted : {len(result.extracted_claims)}")
        if result.reconstructed_response:
            print(f"  Preserved        : {len(result.reconstructed_response.preserved_claims)}")
            print(f"  Corrected        : {len(result.reconstructed_response.corrected_claims)}")
            print(f"  Flagged          : {len(result.reconstructed_response.flagged_claims)}")
        elapsed = result.pipeline_metadata.get("elapsed_seconds", "?")
        print(f"  Time Elapsed     : {elapsed}s")
        print(f"\n  FINAL RESPONSE:")
        print(f"  {result.final_response}")
        print(f"{'=' * 55}\n")
