"""
hallucination_pipeline.py
==========================
Main Pipeline Orchestrator
---------------------------
Ties all 7 phases of the Selective Evidence-Guided Hallucination
Correction Framework into a single, end-to-end pipeline.

Full Pipeline:
  User Query
      ↓
  [Phase 0] LLM Response (provided as input)
      ↓
  [Phase 1] Claim Extraction       (ClaimExtractor)
      ↓
  [Phase 2] Evidence Retrieval     (EvidenceRetriever)
      ↓
  [Phase 3] Quality Assessment     (EvidenceQualityAssessor)
      ↓
  [Phase 4] NLI Verification       (NLIVerifier)
      ↓
  [Phase 5] Selective Correction   (ClaimCorrector)
      ↓
  [Phase 6] Response Reconstruction (ResponseReconstructor)
      ↓
  [Phase 7] Independent Verification (NLIVerifier, again)
      ↓
  Final Verified Response

Phase 7 (Independent Verification):
    After correction, each modified claim is re-verified by the NLI
    verifier. If CSS ≥ 0.75, the correction is accepted. If not, one
    more correction attempt is made (up to MAX_CORRECTION_ITERATIONS).

Design Decisions:
-----------------
- The pipeline is implemented as a class with a single `.run()` method
  for clean integration into experiment scripts.
- All intermediate results are stored in `PipelineResult` for
  full transparency and debugging.
- Component initialization is done once in __init__ for efficiency;
  the expensive NLI model is loaded only once.
- Logging is verbose by default to aid research reproducibility.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from loguru import logger
from tqdm import tqdm

from config import FrameworkConfig
from retrieval.claim_extractor import ClaimExtractor
from retrieval.evidence_retriever import EvidenceRetriever, RetrievalResult
from verification.evidence_quality import EvidenceQualityAssessor, QualityAssessedRetrieval
from verification.nli_verifier import NLIVerifier, ClaimVerificationResult, VerificationLabel
from correction.claim_corrector import ClaimCorrector, CorrectedClaim, CorrectionStatus
from correction.response_reconstructor import ResponseReconstructor, ReconstructedResponse
from evaluation.metrics import EvaluationMetrics, CorrectionOutcome


# ──────────────────────────────────────────────────────────────────
# Pipeline Result Data Structure
# ──────────────────────────────────────────────────────────────────

@dataclass
class PipelineResult:
    """
    Complete result of running the hallucination correction pipeline
    on a single LLM response.

    Attributes:
        original_response:       The raw LLM output.
        extracted_claims:        Atomic claims from Phase 1.
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
    extracted_claims:        List[str]                       = field(default_factory=list)
    retrieval_results:       List[RetrievalResult]           = field(default_factory=list)
    quality_results:         List[QualityAssessedRetrieval]  = field(default_factory=list)
    verification_results:    List[ClaimVerificationResult]   = field(default_factory=list)
    corrected_claims:        List[CorrectedClaim]            = field(default_factory=list)
    reconstructed_response:  Optional[ReconstructedResponse] = None
    final_verification:      List[ClaimVerificationResult]   = field(default_factory=list)
    final_response:          str                             = ""
    pipeline_metadata:       Dict[str, Any]                  = field(default_factory=dict)


# ──────────────────────────────────────────────────────────────────
# Main Pipeline Class
# ──────────────────────────────────────────────────────────────────

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

        # Phase 1: Claim Extraction
        self.claim_extractor = ClaimExtractor(self.config)

        # Phase 2: Evidence Retrieval
        self.evidence_retriever = EvidenceRetriever(self.config)

        # Phase 3: Evidence Quality Assessment
        self.quality_assessor = EvidenceQualityAssessor(self.config)

        # Phase 4 & 7: NLI Verification (shared instance → shared model cache)
        self.nli_verifier = NLIVerifier(self.config)

        # Phase 5: Selective Claim Correction
        self.claim_corrector = ClaimCorrector(self.config)

        # Phase 6: Response Reconstruction
        self.reconstructor = ResponseReconstructor()

        logger.info("Pipeline initialized. Ready to process responses.")

    # ──────────────────────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────────────────────

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

        # ── Phase 1: Claim Extraction ────────────────────────────
        result.extracted_claims = self.claim_extractor.extract(llm_response)
        logger.info(f"Phase 1 complete: {len(result.extracted_claims)} claims extracted.")

        if not result.extracted_claims:
            logger.warning("No claims extracted — returning original response.")
            result.final_response = llm_response
            return result

        # ── Phase 2: Evidence Retrieval ──────────────────────────
        if verbose:
            self._print_phase("PHASE 2: EVIDENCE RETRIEVAL")

        result.retrieval_results = []
        for claim in tqdm(result.extracted_claims, desc="Retrieving evidence", disable=not verbose):
            rr = self.evidence_retriever.retrieve(claim)
            result.retrieval_results.append(rr)

        logger.info("Phase 2 complete: evidence retrieved for all claims.")

        # ── Phase 3: Evidence Quality Assessment ─────────────────
        if verbose:
            self._print_phase("PHASE 3: EVIDENCE QUALITY ASSESSMENT")

        result.quality_results = self.quality_assessor.assess_batch(
            result.retrieval_results
        )
        logger.info("Phase 3 complete: evidence quality scores computed.")

        # ── Phase 4: NLI Verification ─────────────────────────
        if verbose:
            self._print_phase("PHASE 4: NLI VERIFICATION")

        result.verification_results = self.nli_verifier.verify_batch(
            result.quality_results
        )
        logger.info("Phase 4 complete: claims classified by NLI.")

        # Print Phase 4 summary
        if verbose:
            self._print_verification_summary(result.verification_results)

        # ── Phase 5: Selective Correction ─────────────────────────
        if verbose:
            self._print_phase("PHASE 5: SELECTIVE CLAIM CORRECTION")

        result.corrected_claims = self.claim_corrector.correct_batch(
            result.verification_results,
            result.quality_results,
        )
        logger.info("Phase 5 complete: hallucinated claims corrected.")

        # ── Phase 6: Response Reconstruction ──────────────────────
        if verbose:
            self._print_phase("PHASE 6: RESPONSE RECONSTRUCTION")

        result.reconstructed_response = self.reconstructor.reconstruct(
            result.corrected_claims,
            annotate=False,
            original_response=llm_response,
        )
        logger.info("Phase 6 complete: response reconstructed.")

        # ── Phase 7: Independent Verification ─────────────────────
        if verbose:
            self._print_phase("PHASE 7: INDEPENDENT VERIFICATION")

        result.final_verification, result.corrected_claims = (
            self._independent_verification(
                result.corrected_claims,
                result.quality_results,
            )
        )

        # Rebuild final response after Phase 7 adjustments
        final_reconstructed = self.reconstructor.reconstruct(
            result.corrected_claims,
            annotate=False,
        )
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

    # ──────────────────────────────────────────────────────────────
    # Phase 7: Independent Verification
    # ──────────────────────────────────────────────────────────────

    def _independent_verification(
        self,
        corrected_claims: List[CorrectedClaim],
        qa_results: List[QualityAssessedRetrieval],
    ) -> tuple:
        """
        Phase 7: Re-verify all corrected claims with NLI.

        For each CORRECTED claim:
        - Build a temporary QualityAssessedRetrieval with the corrected text
        - Run NLI verification on it
        - If CSS ≥ 0.75: accept correction
        - If CSS < 0.75 and iterations < MAX_CORRECTION_ITERATIONS:
          try one more correction

        Preserved claims are not re-verified (they were already supported).

        Args:
            corrected_claims: List of CorrectedClaim from Phase 5.
            qa_results:       Corresponding quality-assessed evidence.

        Returns:
            Tuple of (final_verification_results, updated_corrected_claims)
        """
        final_verifications: List[ClaimVerificationResult] = []
        updated_corrected: List[CorrectedClaim] = list(corrected_claims)

        for i, cc in enumerate(corrected_claims):
            if cc.status != CorrectionStatus.CORRECTED:
                # Non-corrected claims don't need re-verification
                # Create a placeholder result for tracking
                placeholder = ClaimVerificationResult(
                    claim=cc.corrected_claim,
                    label=VerificationLabel.SUPPORTED
                    if cc.status == CorrectionStatus.PRESERVED
                    else VerificationLabel.INSUFFICIENT_EVIDENCE,
                    css=1.0 if cc.status == CorrectionStatus.PRESERVED else 0.0,
                )
                final_verifications.append(placeholder)
                continue

            # ── Re-verify the corrected claim ─────────────────────
            qa = qa_results[i]

            # Create a temporary QA result with the corrected claim text
            from verification.evidence_quality import QualityAssessedRetrieval as QAR
            temp_qa = QAR(
                claim=cc.corrected_claim,     # use the corrected text
                scored_evidence=qa.scored_evidence,
                best_eqs=qa.best_eqs,
            )

            re_verification = self.nli_verifier.verify(temp_qa)
            final_verifications.append(re_verification)

            if re_verification.css >= self.config.css_supported:
                # ✅ Correction accepted — update the verification result
                logger.info(
                    f"Phase 7: Correction ACCEPTED "
                    f"(CSS={re_verification.css:.3f}): '{cc.corrected_claim[:60]}'"
                )

            elif cc.correction_iterations < self.config.max_correction_iterations:
                # 🔄 Retry correction once more
                logger.info(
                    f"Phase 7: CSS too low ({re_verification.css:.3f}), "
                    f"retrying correction..."
                )
                retry = self.claim_corrector.correct(
                    re_verification, qa
                )
                retry.correction_iterations = cc.correction_iterations + 1
                updated_corrected[i] = retry

            else:
                # ❌ Max iterations reached — revert to original
                logger.warning(
                    f"Phase 7: Correction FAILED after "
                    f"{cc.correction_iterations} attempts. "
                    f"Reverting to original: '{cc.original_claim[:60]}'"
                )
                from correction.claim_corrector import CorrectionStatus as CS
                updated_corrected[i].status = CS.FAILED
                updated_corrected[i].corrected_claim = cc.original_claim

        logger.info("Phase 7 complete: independent verification done.")
        return final_verifications, updated_corrected

    # ──────────────────────────────────────────────────────────────
    # Batch Evaluation Mode
    # ──────────────────────────────────────────────────────────────

    def evaluate(
        self,
        samples: list,
        verbose: bool = True,
    ) -> Dict[str, Any]:
        """
        Run the pipeline on a list of DatasetSample objects and
        compute evaluation metrics.

        This method is designed for use with evaluation datasets
        (FEVER, TruthfulQA, custom) where ground-truth labels are
        available.

        Args:
            samples:  List of DatasetSample objects from DatasetLoader.
            verbose:  If True, show progress bar.

        Returns:
            Dict with 'metrics' (MetricsResult) and 'pipeline_outputs'.
        """
        from evaluation.metrics import EvaluationMetrics, CorrectionOutcome
        from evaluation.dataset_loader import DatasetSample

        pipeline_outputs = []
        correction_outcomes = []

        for sample in tqdm(samples, desc="Evaluating", disable=not verbose):
            # Run pipeline on the claim text directly
            # (In real use, this would be an LLM response; here we use the claim)
            result = self.run(sample.claim, verbose=False)

            # Determine if the framework detected this as hallucinated
            if result.verification_results:
                primary_vr = result.verification_results[0]
                predicted_hallucinated = (
                    primary_vr.label == VerificationLabel.HALLUCINATED
                )
                was_corrected = (
                    result.corrected_claims and
                    result.corrected_claims[0].status == CorrectionStatus.CORRECTED
                )
                post_verified = False
                if result.final_verification:
                    post_verified = (
                        result.final_verification[0].css >= self.config.css_supported
                    )
                was_preserved = (
                    result.corrected_claims and
                    result.corrected_claims[0].status == CorrectionStatus.PRESERVED
                )
            else:
                predicted_hallucinated = False
                was_corrected = False
                post_verified = False
                was_preserved = True

            outcome = CorrectionOutcome(
                claim=sample.claim,
                ground_truth_label=sample.ground_truth,
                predicted_label=predicted_hallucinated,
                was_corrected=was_corrected,
                post_correction_verified=post_verified,
                was_preserved=was_preserved,
                original_was_supported=not sample.ground_truth,
            )
            correction_outcomes.append(outcome)
            pipeline_outputs.append({
                "claim": sample.claim,
                "ground_truth": sample.ground_truth,
                "predicted": predicted_hallucinated,
                "was_corrected": was_corrected,
                "post_correction_verified": post_verified,
                "was_preserved": was_preserved,
                "final_response": result.final_response,
            })

        # Compute metrics
        metrics_calculator = EvaluationMetrics()
        ground_truth = [o.ground_truth_label for o in correction_outcomes]
        predictions  = [o.predicted_label    for o in correction_outcomes]
        metrics = metrics_calculator.compute(ground_truth, predictions, correction_outcomes)

        return {
            "metrics": metrics,
            "pipeline_outputs": pipeline_outputs,
        }

    # ──────────────────────────────────────────────────────────────
    # Display Helpers
    # ──────────────────────────────────────────────────────────────

    @staticmethod
    def _print_phase(title: str) -> None:
        """Print a formatted phase header."""
        print(f"\n{'─' * 55}")
        print(f"  {title}")
        print(f"{'─' * 55}")

    @staticmethod
    def _print_verification_summary(
        verification_results: List[ClaimVerificationResult],
    ) -> None:
        """Print a summary of Phase 4 verification results."""
        labels = [vr.label.value for vr in verification_results]
        supported   = labels.count("SUPPORTED")
        insufficient = labels.count("INSUFFICIENT_EVIDENCE")
        hallucinated = labels.count("HALLUCINATED")
        total = len(labels)

        print(f"\n  Verification Summary ({total} claims):")
        print(f"  ✅ Supported:              {supported}")
        print(f"  ⚠️  Insufficient Evidence:  {insufficient}")
        print(f"  ❌ Hallucinated:           {hallucinated}")

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
