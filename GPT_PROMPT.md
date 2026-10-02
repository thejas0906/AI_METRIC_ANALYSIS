# GPT Context Prompt
# Selective Evidence-Guided Hallucination Correction Framework
# ============================================================
# HOW TO USE:
#   1. Open this file
#   2. Copy everything from the "---" line below to the end
#   3. Paste into ChatGPT / GPT-4
#   4. Replace [YOUR QUESTION HERE] at the bottom with your question
# ============================================================

---

You are being given context about a Python research framework that was
already fully implemented. Here is a complete description of what was built.

=======================================================================
PROJECT: Selective Evidence-Guided Hallucination Correction Framework
=======================================================================

GOAL:
A modular, end-to-end Python pipeline for detecting and selectively
correcting hallucinations in LLM-generated text without regenerating
the entire response. Only unsupported claims are corrected; verified
claims are preserved verbatim.

-----------------------------------------------------------------------
PIPELINE (7 Phases)
-----------------------------------------------------------------------

Phase 1 - Claim Extraction  (retrieval/claim_extractor.py)
  - Input: Raw LLM-generated text
  - Uses spaCy (en_core_web_sm) for sentence segmentation
  - Filters questions, exclamations, and transition phrases
  - Splits coordinate clauses into atomic claims
    Example: "X and Y did Z" becomes two separate claims
  - Deduplicates and normalizes output
  - Output: List of atomic factual claim strings

Phase 2 - Evidence Retrieval  (retrieval/evidence_retriever.py)
  - For each claim, extracts named entities/noun chunks via spaCy
  - Queries Wikipedia OpenSearch API to find candidate pages
  - Fetches page summaries via wikipedia-api library
  - Scores each sentence with a BM25-lite term-overlap relevance score
  - Returns top-K evidence passages per claim (default K=3)
  - Output: RetrievalResult (claim + evidence list + source URLs)

Phase 3 - Evidence Quality Assessment  (verification/evidence_quality.py)
  - Assigns source reliability weights:
      peer_reviewed = 1.00
      government    = 0.90
      wikipedia     = 0.80
      news          = 0.60
      unknown       = 0.50
  - URL heuristic classifier:
      .gov domains       -> government
      arxiv.org, pubmed  -> peer_reviewed
      bbc.com, reuters   -> news
      wikipedia.org      -> wikipedia
  - Normalizes BM25 scores to [0,1] using: score / (score + 1)
  - Evidence Quality Score (EQS) = reliability_weight x normalized_relevance
  - Output: ScoredEvidence list sorted by EQS descending

Phase 4 - NLI-Based Claim Verification  (verification/nli_verifier.py)
  - Model: facebook/bart-large-mnli (zero-shot classification, ~1.6 GB)
  - Runs NLI inference on each (evidence_passage, claim) pair
  - Candidate labels: entailment / neutral / contradiction
  - Claim Support Score: CSS = NLI_entailment_prob x EQS
  - Final CSS = max(CSS_i) across all evidence items for the claim
  - Classification thresholds:
      CSS >= 0.75              -> SUPPORTED
      0.40 <= CSS < 0.75       -> INSUFFICIENT_EVIDENCE
      CSS < 0.40               -> HALLUCINATED
  - Output: ClaimVerificationResult with label + CSS score

Phase 5 - Selective Claim Correction  (correction/claim_corrector.py)
  - SUPPORTED claims       -> PRESERVED exactly (no modification at all)
  - INSUFFICIENT_EVIDENCE  -> FLAGGED (kept as-is, marked uncertain)
  - HALLUCINATED claims    -> CORRECTED using one of two backends:

      Backend 1 - OpenAI (GPT-3.5 / GPT-4):
        Rewrites ONLY the hallucinated claim using evidence context.
        Temperature=0.3, max_tokens=150.
        Prompt instructs: "rewrite ONLY this one claim, 1 sentence."

      Backend 2 - Local rule-based (no API key needed):
        Substitutes mismatched years/numbers found in evidence,
        or falls back to using the first evidence sentence as
        the corrected claim text.

  - Output: CorrectedClaim
    Status options: PRESERVED / CORRECTED / FLAGGED / FAILED

Phase 6 - Response Reconstruction  (correction/response_reconstructor.py)
  - Merges corrected + preserved + flagged claims in original order
  - Joins claims as a natural-sounding paragraph (not bullet points)
  - annotate=True mode embeds [CORRECTED: was '...'] inline
  - Generates a human-readable correction report:
      - Count of preserved / corrected / flagged / failed claims
      - "Original -> Corrected" diff for each changed claim
  - Output: ReconstructedResponse with final_text + correction_report

Phase 7 - Independent Verification  (hallucination_pipeline.py)
  - Re-runs Phase 4 NLI verification on every CORRECTED claim
  - If re-verified CSS >= 0.75: correction accepted
  - If CSS < 0.75 and iterations < max_correction_iterations: retry
  - If max iterations reached: revert to original claim text
  - Rebuilds and returns the final verified response

-----------------------------------------------------------------------
EVALUATION METRICS  (evaluation/metrics.py)
-----------------------------------------------------------------------

All 8 metrics implemented with full unit test coverage:

1. Hallucination Detection Accuracy (HDA)
      HDA = (TP + TN) / Total Claims

2. Precision
      Precision = TP / (TP + FP)

3. Recall
      Recall = TP / (TP + FN)

4. F1 Score
      F1 = 2 x Precision x Recall / (Precision + Recall)

5. Correction Success Rate (CSR)
      CSR = Successfully Corrected Hallucinations / Total Hallucinations
      Only counts corrections that passed Phase 7 re-verification.

6. Claim Preservation Rate (CPR)
      CPR = Preserved Supported Claims / Total Supported Claims

7. Unnecessary Modification Rate (UMR)
      UMR = Modified Supported Claims / Total Supported Claims
      Note: UMR = 1 - CPR (they always sum to 1.0)

8. Final Response Accuracy (FRA)
      FRA = Verified Correct Claims in Final Response / Total Claims

-----------------------------------------------------------------------
DATASETS SUPPORTED  (evaluation/dataset_loader.py)
-----------------------------------------------------------------------

FEVER v1.0 (HuggingFace datasets):
    SUPPORTS  -> ground_truth = False  (claim is not hallucinated)
    REFUTES   -> ground_truth = True   (claim is hallucinated)

TruthfulQA generation (HuggingFace datasets):
    best_answer       -> ground_truth = False
    incorrect_answers -> ground_truth = True

Custom CSV:
    Any file with columns: claim, label (0 or 1), context (optional)

Offline fallback:
    10 hand-crafted TruthfulQA-style examples built directly into code.
    Works without internet or HuggingFace access (useful for demos).

-----------------------------------------------------------------------
PROJECT FILE STRUCTURE
-----------------------------------------------------------------------

AI_METRIC_ANALYSIS/
|
+-- config.py                    <- All thresholds, model names, source weights
+-- hallucination_pipeline.py    <- Main 7-phase orchestrator
+-- run_experiment.py            <- CLI: demo / fever / truthfulqa / custom
+-- compute_metrics.py           <- Metrics tool: LaTeX / CSV / chart output
+-- requirements.txt             <- All Python dependencies
+-- .env.example                 <- API key and device config template
+-- conftest.py / pytest.ini     <- Test configuration
|
+-- retrieval/
|   +-- models.py                <- Pure-Python data classes (no heavy deps)
|   +-- claim_extractor.py       <- Phase 1: spaCy claim extraction
|   +-- evidence_retriever.py    <- Phase 2: Wikipedia + BM25-lite scoring
|
+-- verification/
|   +-- evidence_quality.py      <- Phase 3: EQS computation
|   +-- nli_verifier.py          <- Phase 4: BART-large-mnli CSS classification
|
+-- correction/
|   +-- claim_corrector.py       <- Phase 5: Selective correction
|   +-- response_reconstructor.py <- Phase 6: Merge and rebuild response
|
+-- evaluation/
|   +-- metrics.py               <- All 8 research metrics
|   +-- dataset_loader.py        <- FEVER / TruthfulQA / custom CSV loaders
|
+-- data/
|   +-- custom_claims.csv        <- 21 hand-labelled example claims
|
+-- tests/
    +-- test_metrics.py          <- 20 unit tests (all passing)
    +-- test_evidence_quality.py <- 16 unit tests (all passing)
    +-- test_claim_extractor.py  <- 10 unit tests (requires spaCy)

-----------------------------------------------------------------------
KEY DESIGN DECISIONS
-----------------------------------------------------------------------

- No model training required. Pretrained models only.
- Laptop-friendly: runs entirely on CPU with 8 GB RAM.
- facebook/bart-large-mnli (~1.6 GB) downloads automatically on first run.
- All packages use lazy imports. Heavy dependencies (spaCy, torch,
  transformers) only load when actually used, not at import time.
- OpenAI API key is OPTIONAL. Local rule-based corrector works without it.
- All CSS thresholds are configurable via the FrameworkConfig dataclass.
- Wikipedia API is free and requires no authentication.
- BM25-lite scoring keeps retrieval fast without Elasticsearch.
- EvidenceItem and RetrievalResult live in retrieval/models.py (pure Python)
  so tests can import them without triggering spaCy or network calls.

-----------------------------------------------------------------------
SAMPLE RESULTS (illustrative - actual values depend on configuration)
-----------------------------------------------------------------------

+------------------------------------------+--------+--------+------------+
| Metric                                   | Demo   | FEVER  | TruthfulQA |
+------------------------------------------+--------+--------+------------+
| Hallucination Detection Accuracy (HDA)   | 0.8000 | 0.7600 | 0.7200     |
| Precision                                | 0.8333 | 0.8000 | 0.7500     |
| Recall                                   | 0.8333 | 0.8000 | 0.7500     |
| F1 Score                                 | 0.8333 | 0.8000 | 0.7500     |
| Correction Success Rate (CSR)            | 0.7500 | 0.6500 | 0.6000     |
| Claim Preservation Rate (CPR)            | 0.8571 | 0.8000 | 0.8333     |
| Unnecessary Modification Rate (UMR)      | 0.1429 | 0.2000 | 0.1667     |
| Final Response Accuracy (FRA)            | 0.8000 | 0.7200 | 0.7000     |
+------------------------------------------+--------+--------+------------+

-----------------------------------------------------------------------
HOW TO RUN
-----------------------------------------------------------------------

# Install dependencies
pip install -r requirements.txt
python -m spacy download en_core_web_sm

# Run demo (downloads BART model ~1.6 GB on first run)
python run_experiment.py

# Run on FEVER benchmark (requires HuggingFace internet access)
python run_experiment.py --dataset fever --max-samples 30

# Run on TruthfulQA benchmark
python run_experiment.py --dataset truthfulqa --max-samples 40

# Use OpenAI for higher-quality corrections
set OPENAI_API_KEY=sk-your-key-here
python run_experiment.py --correction-backend openai

# Compute metrics from saved results
python compute_metrics.py --input results/fever_results.json
python compute_metrics.py --input results/fever_results.json --latex
python compute_metrics.py --input results/fever_results.json --csv out.csv
python compute_metrics.py --input results/fever_results.json --chart

# Run unit tests (no ML models required, runs in under 1 second)
python -m pytest tests/test_metrics.py tests/test_evidence_quality.py -v
# Result: 36 passed in 0.42s

-----------------------------------------------------------------------
EXAMPLE PIPELINE USAGE IN PYTHON
-----------------------------------------------------------------------

from config import FrameworkConfig
from hallucination_pipeline import HallucinationCorrectionPipeline

config = FrameworkConfig(
    css_supported=0.75,
    css_insufficient=0.40,
    correction_backend="local",   # or "openai"
    wikipedia_top_k=3,
)

pipeline = HallucinationCorrectionPipeline(config)

llm_response = """
Albert Einstein was a German physicist born in 1879.
He won the Nobel Prize in Physics in 1925.
He worked at Harvard University until his death.
"""

result = pipeline.run(llm_response, query="Tell me about Einstein.")

print(result.final_response)
print(result.reconstructed_response.correction_report)

for vr in result.verification_results:
    print(f"[{vr.label.value}] CSS={vr.css:.3f} | {vr.claim}")

=======================================================================
Given the above context about this project, [YOUR QUESTION HERE]
=======================================================================
