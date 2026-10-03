# Selective Evidence-Guided Hallucination Correction Framework

> **A modular, lightweight Python research framework for post-generation hallucination detection and selective correction in Large Language Models (LLMs).**

---

## Overview

Most existing hallucination mitigation approaches either:
1. **Detect hallucinations only** — without correcting them
2. **Regenerate the entire response** — losing accurate, supported content

This framework implements a **selective correction strategy**: only unsupported claims are corrected, while factually supported claims are preserved verbatim.

```
User Query
    ↓
LLM Response
    ↓
[Phase 1] Claim Extraction         (spaCy NLP)
    ↓
[Phase 2] Evidence Retrieval       (Wikipedia API)
    ↓
[Phase 3] Evidence Quality Score   (Source Reliability Weights)
    ↓
[Phase 4] NLI Verification         (facebook/bart-large-mnli)
    ↓
Supported / Contradicted / Insufficient Evidence
    ↓
[Phase 5] Selective Claim Correction (LLM or rule-based)
    ↓
[Phase 6] Response Reconstruction
    ↓
[Phase 7] Independent Re-Verification
    ↓
Final Verified Response
```

---

## Project Structure

```
AI_METRIC_ANALYSIS/
│
├── config.py                     # Central configuration (thresholds, weights, models)
├── hallucination_pipeline.py     # Main pipeline orchestrator (Phases 1–7)
├── run_experiment.py             # Example experiment script with CLI
├── compute_metrics.py            # Standalone metrics calculator & visualizer
├── requirements.txt              # Python dependencies
├── .env.example                  # Environment variable template
│
├── retrieval/                    # Phase 1 & 2
│   ├── __init__.py
│   ├── claim_extractor.py        # Phase 1: Atomic claim extraction (spaCy)
│   └── evidence_retriever.py     # Phase 2: Wikipedia evidence retrieval
│
├── verification/                 # Phase 3 & 4
│   ├── __init__.py
│   ├── evidence_quality.py       # Phase 3: Evidence Quality Score (EQS)
│   └── nli_verifier.py           # Phase 4: NLI verification (BART-large-mnli)
│
├── correction/                   # Phase 5 & 6
│   ├── __init__.py
│   ├── claim_corrector.py        # Phase 5: Selective hallucination correction
│   └── response_reconstructor.py # Phase 6: Merge corrected + preserved claims
│
├── evaluation/                   # Metrics & Datasets
│   ├── __init__.py
│   ├── metrics.py                # All 8 evaluation metrics (HDA, P, R, F1, CSR, CPR, UMR, FRA)
│   └── dataset_loader.py         # FEVER / TruthfulQA / Custom CSV loaders
│
├── data/
│   └── custom_claims.csv         # Example custom dataset
│
├── results/                      # Auto-generated experiment results
└── logs/                         # Auto-generated log files
```

---

## Installation

### 1. Clone and set up environment

```bash
# Clone repository
git clone <your-repo-url>
cd AI_METRIC_ANALYSIS

# Create virtual environment
python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate    # Linux/macOS

# Install dependencies
pip install -r requirements.txt

# Download spaCy language model (required for claim extraction)
python -m spacy download en_core_web_sm
```

### 2. Configure environment (optional — for OpenAI correction)

```bash
# Copy the example env file
copy .env.example .env

# Edit .env and add your OpenAI API key (optional)
# Without it, the framework uses the local rule-based corrector
```

---

## Quick Start

### Run demo (no API key needed)

```bash
python run_experiment.py
```

### Run on FEVER dataset

```bash
python run_experiment.py --dataset fever --max-samples 30
```

### Run on TruthfulQA dataset

```bash
python run_experiment.py --dataset truthfulqa --max-samples 40
```

### Use OpenAI for higher-quality corrections

```bash
set OPENAI_API_KEY=sk-your-key-here
python run_experiment.py --correction-backend openai
```

### Compute and compare metrics

```bash
# From saved results
python compute_metrics.py --input results/fever_results.json

# Compare two experiments
python compute_metrics.py --compare results/fever_results.json results/truthfulqa_results.json

# Generate LaTeX table
python compute_metrics.py --input results/fever_results.json --latex

# Save metrics to CSV
python compute_metrics.py --input results/fever_results.json --csv metrics.csv

# Show bar chart
python compute_metrics.py --input results/fever_results.json --chart
```

---

## Use the Pipeline in Your Own Code

```python
from config import FrameworkConfig
from hallucination_pipeline import HallucinationCorrectionPipeline

# Configure (tweak thresholds as needed)
config = FrameworkConfig(
    css_supported=0.75,           # Claims above this → SUPPORTED
    css_insufficient=0.40,        # Claims below this → HALLUCINATED
    correction_backend="local",   # "openai" or "local"
    wikipedia_top_k=3,
)

# Initialize pipeline (loads NLI model once)
pipeline = HallucinationCorrectionPipeline(config)

# Your LLM-generated response
llm_response = """
Albert Einstein was a German physicist born in 1879.
He won the Nobel Prize in Physics in 1925.
He worked at Harvard University until his death.
"""

# Run the pipeline
result = pipeline.run(llm_response, query="Tell me about Einstein.")

# Inspect results
print("FINAL RESPONSE:", result.final_response)
print("CORRECTION REPORT:", result.reconstructed_response.correction_report)

# View claim-by-claim verification
for vr in result.verification_results:
    print(f"[{vr.label.value}] CSS={vr.css:.3f} | {vr.claim}")
```

---

## Evaluation Metrics & Metric Interpretation

The framework rigorously distinguishes between **Ground-Truth Metrics** (`evaluation_type: ground_truth`) and **Internal Verifier Metrics** (`evaluation_type: internal_verifier`) to prevent self-verification bias.

### Metric Interpretation

> [!IMPORTANT]
> **Ground-Truth Metrics** evaluate actual correctness against external benchmark annotations and target corrections.
> **Verifier Metrics** evaluate internal pipeline self-consistency using the Phase-7 NLI model.
> **Verifier metrics should NOT be interpreted as independent correctness measures**, as they measure whether the pipeline agrees with itself.

### 1. Ground-Truth Metrics (`evaluation_type: ground_truth`)
Evaluated against benchmark ground-truth labels and external gold correction targets:

| Metric | Formula | Description |
|---|---|---|
| **HDA (Accuracy)** | $(TP + TN) / N$ | Hallucination Detection Accuracy |
| **Precision** | $TP / (TP + FP)$ | Of detected hallucinations, fraction truly wrong |
| **Recall** | $TP / (TP + FN)$ | Of all ground-truth hallucinations, fraction caught |
| **F1 Score** | $2 \cdot P \cdot R / (P + R)$ | Harmonic mean of Precision and Recall |
| **CSR_GT** | $\text{Corrected}_{\text{GT\_matched}} / \text{Hallucinated}_{\text{GT}}$ | Ground Truth Correction Success Rate (matches gold correction text) |
| **CPR_GT** | $\text{Preserved}_{\text{Supported}} / \text{Total}_{\text{Supported\_GT}}$ | Claim Preservation Rate (preserves verified facts) |
| **UMR_GT** | $\text{Modified}_{\text{Supported}} / \text{Total}_{\text{Supported\_GT}}$ | Unnecessary Modification Rate ($UMR = 1 - CPR$) |
| **FRA_GT** | $(\text{Preserved}_{\text{Supp}} + \text{Corrected}_{\text{GT\_matched}}) / N$ | Final Response Accuracy against Ground Truth |

### 2. Internal Verifier Metrics (`evaluation_type: internal_verifier`)
Evaluated using Phase-7 fresh-retrieval NLI verification (`facebook/bart-large-mnli`):

| Metric | Formula | Description |
|---|---|---|
| **CSR_Verifier** | $\text{Accepted}_{\text{Phase7}} / \text{Total}_{\text{Corrected}}$ | Fraction of modified claims accepted by Phase-7 NLI |
| **Acceptance Rate** | $\text{Accepted}_{\text{Phase7}} / \text{Total}_{\text{Attempts}}$ | Fraction of correction attempts accepted |
| **Verification Pass Rate** | $\text{Passing}_{\text{Phase7}} / \text{Total}_{\text{Sent\_to\_Phase7}}$ | Pass rate for Phase-7 independent verification |
| **FRA_Verifier** | $(\text{Preserved} + \text{Accepted}_{\text{Phase7}}) / N$ | Response accuracy as judged by the internal verifier |

---

## Claim Support Score (CSS)

```
CSS = NLI_entailment_probability × Evidence_Quality_Score

Evidence_Quality_Score = source_reliability_weight × normalized_relevance

Source Weights:
  peer_reviewed  → 1.00
  government     → 0.90
  wikipedia      → 0.80
  news           → 0.60
  unknown        → 0.50

Classification:
  CSS ≥ 0.75         → SUPPORTED
  0.40 ≤ CSS < 0.75  → INSUFFICIENT_EVIDENCE
  CSS < 0.40         → HALLUCINATED
```

---

## Configuration

All parameters are centralized in [`config.py`](config.py):

```python
# Key parameters to tune
css_supported    = 0.75   # Lower → more permissive (less flagged)
css_insufficient = 0.40   # Lower → more claims marked as hallucinated
wikipedia_top_k  = 3      # More evidence → better coverage, slower
nli_device       = "cpu"  # Use "cuda" if you have a GPU
correction_backend = "local"  # "openai" for higher quality corrections
```

---

## NLI Model

This framework uses **`facebook/bart-large-mnli`** (1.6 GB, downloaded automatically on first run).

- A zero-shot classification model fine-tuned on MultiNLI
- Input: `evidence_passage` + `claim`
- Output: probabilities for `[entailment, neutral, contradiction]`
- We use the **entailment** probability as the raw claim support score

> **GPU:** Set `NLI_DEVICE = "cuda"` in `config.py` for ~10x faster inference.

---

## Datasets

| Dataset | Type | Labels | HuggingFace ID |
|---|---|---|---|
| **FEVER** | Fact verification | SUPPORTS / REFUTES | `fever/v1.0` |
| **TruthfulQA** | LLM hallucinations | correct / incorrect answers | `truthful_qa/generation` |
| **Custom CSV** | User-provided | 0 (supported) / 1 (hallucinated) | `data/custom_claims.csv` |

---

## Requirements

- Python 3.9+
- 8 GB RAM (for BART-large-mnli on CPU)
- Internet connection (for Wikipedia API + first-time model download)
- GPU optional but recommended for large-scale evaluation

---

## Extending the Framework

### Add a new evidence source

In `retrieval/evidence_retriever.py`, add a new retrieval method and register the source type. Then add its weight in `config.py`:

```python
EVIDENCE_WEIGHTS["arxiv"] = 0.95
```

### Swap the NLI model

In `config.py`, change:
```python
NLI_MODEL_NAME = "cross-encoder/nli-deberta-v3-base"  # faster alternative
```

### Custom correction prompt

In `correction/claim_corrector.py`, modify `_correct_with_openai()` to change the correction instruction.

---

## Citation

If you use this framework in your research, please cite:

```bibtex
@misc{hallucination_correction_2024,
  title  = {Selective Evidence-Guided Hallucination Correction Framework},
  year   = {2024},
  note   = {Conference paper prototype. GitHub: <your-repo-url>}
}
```

---

## License

MIT License. See `LICENSE` for details.
