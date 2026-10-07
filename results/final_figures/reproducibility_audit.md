# Framework Reproducibility & Version Control Audit

**Date:** October 7, 2026  
**Artifact:** [`results/final_figures/reproducibility_audit.md`](file:///c:/Users/kickg/OneDrive/Documents/AI_METRIC_ANALYSIS/results/final_figures/reproducibility_audit.md)  
**Scope:** Pipeline reproducibility, pickle cache audit, and `.gitignore` safety evaluation  

---

## 1. Audit of Serialized Pickle Files (`*.pkl`, `*.pickle`)

### Repository Inventory
- **File Found:** `results/audit_archive/precomputed_nli_data.pkl` (126 KB).
- **Location:** Safely archived in `results/audit_archive/`.

### Functional Role in Reproduction
- **Purpose:** Created by [`eval_css_thresholds.py`](file:///c:/Users/kickg/OneDrive/Documents/AI_METRIC_ANALYSIS/eval_css_thresholds.py) and [`calibrate_support_thresholds.py`](file:///c:/Users/kickg/OneDrive/Documents/AI_METRIC_ANALYSIS/calibrate_support_thresholds.py) to cache raw NLI sentence-pair logits and evidence quality assessments.
- **Reproducibility Assessment:**
  - In `eval_css_thresholds.py`, the code contains an automatic fallback: if `precomputed_nli_data.pkl` is absent, the script loads the benchmark samples from [`data/multiclaim_benchmark.json`](file:///c:/Users/kickg/OneDrive/Documents/AI_METRIC_ANALYSIS/data/multiclaim_benchmark.json), computes all retrieval and NLI inferences from scratch using `facebook/bart-large-mnli`, and re-creates the cache.
  - Figure generation script [`generate_final_figures.py`](file:///c:/Users/kickg/OneDrive/Documents/AI_METRIC_ANALYSIS/generate_final_figures.py) is completely independent and does **not** rely on pickle files.
- **GitIgnore Risk:**
  - A blanket `*.pkl` rule is an anti-pattern: it risks silently ignoring any future serialized evaluation checkpoints or essential models required for offline reproducibility.
  - **Correction Applied:** Removed blanket `*.pkl` and `*.pickle` rules. Replaced with path-specific exclusions (`results/audit_archive/*.pkl`, `results/cache/*.pkl`, `tmp/*.pkl`, etc.).

---

## 2. Categorization of Repository Files

### A. Required Tracked Files (Version Controlled)
1. **Source Code & Pipelines:**
   - [`config.py`](file:///c:/Users/kickg/OneDrive/Documents/AI_METRIC_ANALYSIS/config.py): System hyperparameters and calibrated thresholds ($\tau_{CSS} = 0.30$, $\tau_{CON} = 0.30$).
   - [`hallucination_pipeline.py`](file:///c:/Users/kickg/OneDrive/Documents/AI_METRIC_ANALYSIS/hallucination_pipeline.py): End-to-end 7-phase pipeline controller.
   - [`run_experiment.py`](file:///c:/Users/kickg/OneDrive/Documents/AI_METRIC_ANALYSIS/run_experiment.py): Benchmark execution script.
   - Modules: `retrieval/`, `verification/`, `correction/`, and `evaluation/`.
2. **Benchmark Datasets:**
   - [`data/multiclaim_benchmark.json`](file:///c:/Users/kickg/OneDrive/Documents/AI_METRIC_ANALYSIS/data/multiclaim_benchmark.json): 20 responses with 84 annotated atomic claims.
   - [`data/custom_claims.csv`](file:///c:/Users/kickg/OneDrive/Documents/AI_METRIC_ANALYSIS/data/custom_claims.csv): Domain-specific claim pairs.
   - [`data/wikipedia_cache.json`](file:///c:/Users/kickg/OneDrive/Documents/AI_METRIC_ANALYSIS/data/wikipedia_cache.json): Deterministic, offline Wikipedia API cache ensuring 100% reproducible retrieval without network dependency.
3. **Benchmark Results & Ground Truth:**
   - [`results/multiclaim_results.json`](file:///c:/Users/kickg/OneDrive/Documents/AI_METRIC_ANALYSIS/results/multiclaim_results.json): Per-response reports, claim spans, and ground-truth correction outcomes.
   - [`results/multiclaim_metrics.csv`](file:///c:/Users/kickg/OneDrive/Documents/AI_METRIC_ANALYSIS/results/multiclaim_metrics.csv): Evaluated accuracy, precision, recall, F1, CPR, and UMR metrics.
4. **Figure Generation & Deliverables:**
   - [`generate_final_figures.py`](file:///c:/Users/kickg/OneDrive/Documents/AI_METRIC_ANALYSIS/generate_final_figures.py): Deterministic visualization generator.
   - [`results/final_figures/confusion_matrix.png`](file:///c:/Users/kickg/OneDrive/Documents/AI_METRIC_ANALYSIS/results/final_figures/confusion_matrix.png)
   - [`results/final_figures/precision_recall_f1.png`](file:///c:/Users/kickg/OneDrive/Documents/AI_METRIC_ANALYSIS/results/final_figures/precision_recall_f1.png)
   - [`results/final_figures/classification_distribution.png`](file:///c:/Users/kickg/OneDrive/Documents/AI_METRIC_ANALYSIS/results/final_figures/classification_distribution.png)
   - [`results/final_figures/threshold_calibration.png`](file:///c:/Users/kickg/OneDrive/Documents/AI_METRIC_ANALYSIS/results/final_figures/threshold_calibration.png)
   - [`results/final_figures/before_after_calibration.png`](file:///c:/Users/kickg/OneDrive/Documents/AI_METRIC_ANALYSIS/results/final_figures/before_after_calibration.png)
   - [`results/final_figures/figure_generation_report.md`](file:///c:/Users/kickg/OneDrive/Documents/AI_METRIC_ANALYSIS/results/final_figures/figure_generation_report.md)
   - [`results/final_figures/benchmark_consistency_report.md`](file:///c:/Users/kickg/OneDrive/Documents/AI_METRIC_ANALYSIS/results/final_figures/benchmark_consistency_report.md)
   - [`results/final_figures/repository_cleanup_report.md`](file:///c:/Users/kickg/OneDrive/Documents/AI_METRIC_ANALYSIS/results/final_figures/repository_cleanup_report.md)

### B. Safe-to-Ignore Files
- **Archived Diagnostics & Interim Runs:** `results/audit_archive/` (26 files).
- **Execution & Runtime Logs:** `logs/`, `*.log`.
- **Python Bytecode & Caches:** `__pycache__/`, `*.pyc`, `*.pyo`, `.pytest_cache/`.
- **Local Scratch Directories:** `tmp/`, `temp/`, `scratch/`.
- **OS & Editor Metadata:** `.DS_Store`, `Thumbs.db`, `.vscode/`, `.idea/`.

---

## 3. Analysis of Incorrectly Ignored Files

During the audit, two previous rules posed risks to complete reproducibility:
1. **Blanket `results/*.json` and `results/*.csv`:** In previous configurations, this rule threatened to mask `multiclaim_results.json` and `multiclaim_metrics.csv`.  
   *Resolution:* Added explicit negation patterns `!results/multiclaim_results.json` and `!results/multiclaim_metrics.csv` along with `!results/final_figures/**`.
2. **Blanket `*.pkl` and `*.pickle`:** Would have suppressed any future serialized model artifact or offline evaluation checkpoint.  
   *Resolution:* Replaced with targeted directory-based rules: `results/audit_archive/*.pkl`, `results/cache/*.pkl`, `tmp/*.pkl`.

---

## 4. Final Recommendation on `.gitignore`

The `.gitignore` configuration in the repository is now fine-grained, secure, and reproducible:
- It excludes all runtime noise, diagnostic scratch files, and virtual environments.
- It tracks every script, dataset, offline cache (`wikipedia_cache.json`), configuration, metric table, and publication figure needed for any researcher to verify and reproduce the published results.
