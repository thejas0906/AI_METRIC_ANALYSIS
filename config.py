"""
config.py
=========
Central configuration file for the Selective Evidence-Guided
Hallucination Correction Framework.

All thresholds, model names, API settings, and evidence weights
are defined here for easy experimentation and tuning.
"""

import os
from dataclasses import dataclass, field
from typing import Dict, Optional


# ──────────────────────────────────────────────────────────────────
# NLI Model Configuration
# ──────────────────────────────────────────────────────────────────
NLI_MODEL_NAME = "facebook/bart-large-mnli"
"""
facebook/bart-large-mnli is a zero-shot NLI model.
Input:  premise (evidence) + hypothesis (claim)
Output: probabilities for [contradiction, neutral, entailment]
We use the entailment probability as our raw support score.
"""

NLI_MAX_LENGTH = 512          # Max token length for BART encoder
NLI_BATCH_SIZE = 4            # Batch size for NLI inference (laptop-friendly)
NLI_DEVICE = "cpu"            # Use "cuda" if GPU is available


# ──────────────────────────────────────────────────────────────────
# Claim Support Score (CSS) Thresholds
# ──────────────────────────────────────────────────────────────────
CSS_SUPPORTED_THRESHOLD    = 0.75   # CSS >= 0.75  → SUPPORTED
CSS_INSUFFICIENT_THRESHOLD = 0.40   # 0.40 <= CSS < 0.75 → INSUFFICIENT_EVIDENCE
# CSS < 0.40 → HALLUCINATED


# ──────────────────────────────────────────────────────────────────
# Evidence Source Reliability Weights
# ──────────────────────────────────────────────────────────────────
EVIDENCE_WEIGHTS: Dict[str, float] = {
    "wikipedia":       0.80,   # Wikipedia: broadly reliable but editable
    "government":      0.90,   # Official government sources
    "peer_reviewed":   1.00,   # Academic / peer-reviewed papers
    "news":            0.60,   # News articles (varying quality)
    "unknown":         0.50,   # Unknown / unclassified source
}
"""
These weights scale the raw NLI entailment probability to produce
a source-adjusted Claim Support Score (CSS):

    CSS = NLI_entailment_prob × source_weight
"""


# ──────────────────────────────────────────────────────────────────
# Evidence Retrieval Settings
# ──────────────────────────────────────────────────────────────────
WIKIPEDIA_LANGUAGE    = "en"    # Wikipedia language
WIKIPEDIA_TOP_K       = 3       # Number of top evidence passages to retrieve per claim
WIKIPEDIA_SUMMARY_LEN = 5       # Number of sentences per Wikipedia summary
WIKIPEDIA_USER_AGENT  = "SelectiveHallucinationCorrectionBot/1.0 (https://github.com/thejas0906/AI_METRIC_ANALYSIS; hallucination-research@example.com)"


# ──────────────────────────────────────────────────────────────────
# Claim Extraction Settings
# ──────────────────────────────────────────────────────────────────
SPACY_MODEL = "en_core_web_sm"  # Lightweight spaCy model (laptop-friendly)
MIN_CLAIM_LENGTH = 10           # Minimum character length for a valid claim
MAX_CLAIM_LENGTH = 300          # Maximum character length for a valid claim


# ──────────────────────────────────────────────────────────────────
# Correction Settings
# ──────────────────────────────────────────────────────────────────
MAX_CORRECTION_ITERATIONS = 2   # Max correction attempts before giving up
CORRECTION_BACKEND = "openai"   # "openai" | "local" (rule-based fallback)
OPENAI_MODEL = "gpt-3.5-turbo"  # Model to use for claim correction


# ──────────────────────────────────────────────────────────────────
# OpenAI API Key (loaded from environment)
# ──────────────────────────────────────────────────────────────────
OPENAI_API_KEY: Optional[str] = os.getenv("OPENAI_API_KEY", None)


# ──────────────────────────────────────────────────────────────────
# Evaluation Settings
# ──────────────────────────────────────────────────────────────────
EVALUATION_DATASET = "truthfulqa"  # "truthfulqa" | "fever" | "custom"
EXPERIMENT_RESULTS_DIR = "results"


# ──────────────────────────────────────────────────────────────────
# Logging
# ──────────────────────────────────────────────────────────────────
LOG_LEVEL = "INFO"   # DEBUG | INFO | WARNING | ERROR
LOG_FILE  = "logs/framework.log"


# ──────────────────────────────────────────────────────────────────
# Dataclass: FrameworkConfig (used for runtime config passing)
# ──────────────────────────────────────────────────────────────────
@dataclass
class FrameworkConfig:
    """
    Runtime configuration object.
    Instantiate this and pass to pipeline components for
    clean dependency injection and easy experimentation.
    """
    nli_model_name:           str            = NLI_MODEL_NAME
    nli_max_length:           int            = NLI_MAX_LENGTH
    nli_batch_size:           int            = NLI_BATCH_SIZE
    nli_device:               str            = NLI_DEVICE
    css_supported:            float          = CSS_SUPPORTED_THRESHOLD
    css_insufficient:         float          = CSS_INSUFFICIENT_THRESHOLD
    evidence_weights:         Dict[str, float] = field(default_factory=lambda: EVIDENCE_WEIGHTS.copy())
    wikipedia_language:       str            = WIKIPEDIA_LANGUAGE
    wikipedia_top_k:          int            = WIKIPEDIA_TOP_K
    wikipedia_summary_len:    int            = WIKIPEDIA_SUMMARY_LEN
    wikipedia_user_agent:     str            = WIKIPEDIA_USER_AGENT
    spacy_model:              str            = SPACY_MODEL
    min_claim_length:         int            = MIN_CLAIM_LENGTH
    max_claim_length:         int            = MAX_CLAIM_LENGTH
    max_correction_iterations: int           = MAX_CORRECTION_ITERATIONS
    correction_backend:       str            = CORRECTION_BACKEND
    openai_model:             str            = OPENAI_MODEL
    openai_api_key:           Optional[str]  = field(default_factory=lambda: OPENAI_API_KEY)
    evaluation_dataset:       str            = EVALUATION_DATASET
    results_dir:              str            = EXPERIMENT_RESULTS_DIR
    log_level:                str            = LOG_LEVEL
