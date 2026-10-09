# Forensic Benchmark Audit & Results Consistency Report

**Audit Date:** October 10, 2026  
**Auditor:** Automated Forensic Audit System (DeepMind Antigravity)  
**Target Repository:** `AI_METRIC_ANALYSIS`  
**Purpose:** Pre-publication verification of benchmark outputs, confusion matrices, threshold calibration, and figure alignment for IEEE manuscript submission.

---

## Executive Summary

A comprehensive, read-only forensic audit was performed across all experimental files, intermediate caches, visualization scripts, and serialized evaluation artifacts in the `AI_METRIC_ANALYSIS` repository.

### Key Audit Findings:
1. **Resolution of Conflicting Confusion Matrices (Result A vs. Result B):**
   - **Result A** ($\text{TP}=12, \text{FP}=3, \text{FN}=10, \text{TN}=59$, Precision $0.8000$, Recall $0.5455$, F1 $0.6486$, Accuracy $0.8452$) represents the **calibrated end-to-end framework** ($\tau_{CSS} = 0.30, \tau_{CON} = 0.30$).
   - **Result B** ($\text{TP}=13, \text{FP}=4, \text{FN}=9, \text{TN}=58$, Precision $0.7647$, Recall $0.5909$, F1 $0.6667$, Accuracy $0.8452$) represents an **earlier uncalibrated baseline run** where only the contradiction threshold was lowered ($\tau_{CON} = 0.30$), but the support threshold remained at the default uncalibrated value ($\tau_{CSS} = 0.75$).
   - **Result C (Active in `results/multiclaim_results.json`):** A subsequent run with expanded Wikipedia retrieval caching produced $\text{TP}=15, \text{FP}=4, \text{FN}=7, \text{TN}=58$ (F1 $0.7317$, Accuracy $0.8690$). The published figures in `results/final_figures/` strictly portray **Result A**, which must be explicitly documented in the paper to avoid confusion with `multiclaim_metrics.csv`.
2. **Independence and Provenance of Result A:**
   - Result A was independently reproduced and verified from `results/audit_archive/precomputed_nli_data.pkl` and `results/audit_archive/support_threshold_calibration_results.json`.
3. **Threshold Calibration Invariance:**
   - The plateau between $\tau_{CSS}=0.30$ and $0.35$ (both yielding identical binary metrics), as well as between $\tau_{CSS}=0.40$ and $0.45$, is mathematically proved: while multi-class decisions shift from `SUPPORTED` to `UNVERIFIABLE`, `CONTRADICTED` remains invariant at 15 and 17 claims respectively.
4. **Figure Integrity:**
   - All 5 publication figures in `results/final_figures/` (`confusion_matrix.png`, `precision_recall_f1.png`, `classification_distribution.png`, `threshold_calibration.png`, `before_after_calibration.png`) are 100% mutually consistent and accurately depict **Result A**.

---

## 1. Inventory of Authoritative Benchmark Files

The repository file tree was audited to establish what files actually exist:

| File Path | Status | Size | Content Description & Provenance |
| :--- | :---: | :---: | :--- |
| [`config.py`](file:///c:/Users/kickg/OneDrive/Documents/AI_METRIC_ANALYSIS/config.py) | **Active** | 9.2 KB | Framework configuration defining $\tau_{CSS}=0.30$, $\tau_{CON}=0.30$, and `facebook/bart-large-mnli`. |
| [`run_experiment.py`](file:///c:/Users/kickg/OneDrive/Documents/AI_METRIC_ANALYSIS/run_experiment.py) | **Active** | 14.9 KB | Experiment execution harness evaluating LLM responses. |
| [`results/multiclaim_results.json`](file:///c:/Users/kickg/OneDrive/Documents/AI_METRIC_ANALYSIS/results/multiclaim_results.json) | **Active** | 83.1 KB | Detailed outputs of a recent run (Result C: 46 SUP, 19 CON, 19 UNV, TP=15). |
| [`results/multiclaim_metrics.csv`](file:///c:/Users/kickg/OneDrive/Documents/AI_METRIC_ANALYSIS/results/multiclaim_metrics.csv) | **Active** | 1.6 KB | Metric summary table corresponding directly to `results/multiclaim_results.json`. |
| [`results/audit_archive/baseline_multiclaim_results.json`](file:///c:/Users/kickg/OneDrive/Documents/AI_METRIC_ANALYSIS/results/audit_archive/baseline_multiclaim_results.json) | **Archived** | 82.2 KB | Uncalibrated baseline ($\tau_{CON}=0.40, \tau_{CSS}=0.75$, 0 SUP, 2 CON, 82 UNV). |
| [`results/audit_archive/threshold030_multiclaim_results.json`](file:///c:/Users/kickg/OneDrive/Documents/AI_METRIC_ANALYSIS/results/audit_archive/threshold030_multiclaim_results.json) | **Archived** | 82.7 KB | **Source of Result B** ($\tau_{CON}=0.30, \tau_{CSS}=0.75$, 0 SUP, 17 CON, 67 UNV). |
| [`results/audit_archive/hybrid_multiclaim_results.json`](file:///c:/Users/kickg/OneDrive/Documents/AI_METRIC_ANALYSIS/results/audit_archive/hybrid_multiclaim_results.json) | **Archived** | 82.6 KB | Hybrid rule trial run ($\text{prob} \ge 0.80 \land \text{EQS} \ge 0.30$, TP=14). |
| [`results/audit_archive/support_threshold_calibration_results.json`](file:///c:/Users/kickg/OneDrive/Documents/AI_METRIC_ANALYSIS/results/audit_archive/support_threshold_calibration_results.json) | **Archived** | 1.5 KB | **Source of Result A** and calibration sweep ($\tau \in [0.30, 0.45]$). |
| [`results/audit_archive/css_thresholds_evaluation.json`](file:///c:/Users/kickg/OneDrive/Documents/AI_METRIC_ANALYSIS/results/audit_archive/css_thresholds_evaluation.json) | **Archived** | 1.8 KB | Extended CSS threshold sweep ($\tau \in [0.20, 0.75]$). |
| [`results/audit_archive/precomputed_nli_data.pkl`](file:///c:/Users/kickg/OneDrive/Documents/AI_METRIC_ANALYSIS/results/audit_archive/precomputed_nli_data.pkl) | **Archived** | 126.9 KB | Cached pairwise BART-NLI logits & EQS scores across all 84 claims. |
| [`results/final_figures/benchmark_consistency_report.md`](file:///c:/Users/kickg/OneDrive/Documents/AI_METRIC_ANALYSIS/results/final_figures/benchmark_consistency_report.md) | **Active** | 4.1 KB | Previous consistency audit report confirming Option A in figures. |
| [`results/final_figures/figure_generation_report.md`](file:///c:/Users/kickg/OneDrive/Documents/AI_METRIC_ANALYSIS/results/final_figures/figure_generation_report.md) | **Active** | 5.7 KB | Resolution and styling documentation for all 5 generated publication figures. |

---

## 2. Resolution of Conflicting Confusion Matrices

### Comparative Matrix

| Evaluation Dimension | **Result A**<br>*(IEEE Results Document)* | **Result B**<br>*(Earlier Benchmark Run)* | **Result C**<br>*(Active `multiclaim_results.json`)* |
| :--- | :---: | :---: | :---: |
| **True Positives (TP)** | **12** | 13 | 15 |
| **False Positives (FP)** | **3** | 4 | 4 |
| **False Negatives (FN)** | **10** | 9 | 7 |
| **True Negatives (TN)** | **59** | 58 | 58 |
| **Total Claims ($N$)** | **84** | **84** | **84** |
| **Precision** | **0.8000** | 0.7647 | 0.7895 |
| **Recall** | **0.5455** | 0.5909 | 0.6818 |
| **F1-Score** | **0.6486** | 0.6667 | 0.7317 |
| **Accuracy (HDA)** | **0.8452** | 0.8452 | 0.8690 |
| **CPR** | **0.9516** | 1.0000 | 0.9677 |
| **UMR** | **0.0484** | 0.0000 | 0.0323 |
| **Predicted Classes** | **29 SUP, 15 CON, 40 UNV** | 0 SUP, 17 CON, 67 UNV | 46 SUP, 19 CON, 19 UNV |
| **$\tau_{CON}$ (Contradiction)** | 0.30 | 0.30 | 0.30 |
| **$\tau_{CSS}$ (Support)** | **0.30 (Calibrated)** | **0.75 (Uncalibrated)** | 0.30 (Calibrated + Expanded Retrieval) |
| **Execution Script** | `calibrate_support_thresholds.py` | `compare_thresholds.py` | `run_experiment.py` (post-commit `b948270`) |
| **Primary Artifact** | `support_threshold_calibration_results.json` | `threshold030_multiclaim_results.json` | `multiclaim_results.json` |

### Provenance Analysis & Root Cause
1. **Why Result B has 0 SUPPORTED claims:**
   - In `compare_thresholds.py`, the experiment only varied `contradiction_threshold` (testing 0.40 vs 0.30). It did not modify `css_supported`, leaving it at the framework's legacy default of $0.75$.
   - Because Wikipedia evidence quality scores (EQS) cap at $\sim 0.33\text{--}0.38$, no claim achieved $\text{CSS} \ge 0.75$. Hence, **0 claims were classified as SUPPORTED**, and 67 claims were suppressed into `UNVERIFIABLE`.
2. **Why Result B has 17 CONTRADICTED claims while Result A has 15:**
   - In `NLIVerifier._classify()`, the pipeline evaluates `if css >= self.config.css_supported: return SUPPORTED` **before** checking contradiction.
   - When $\tau_{CSS}=0.75$ (Result B), two claims with $\text{CSS} \in [0.30, 0.35)$ failed the support check and fell through to the contradiction logic, where their contradiction score exceeded $0.30$.
   - When $\tau_{CSS}$ was calibrated to $0.30$ (Result A), those 2 claims were intercepted as `SUPPORTED`. One was a true hallucination ($\text{TP}: 13 \rightarrow 12$), and one was a factual claim ($\text{FP}: 4 \rightarrow 3$).
3. **Verdict on IEEE Paper Alignment:**
   - **Result A is the mathematically correct portrayal of the fully calibrated framework ($\tau_{CSS}=0.30, \tau_{CON}=0.30$).**
   - Result B represents a transitional ablation state where contradiction was calibrated without calibrating support.
   - All 5 publication figures in `results/final_figures/` are generated around **Result A**.

---

## 3. Independent Metric Recomputation

### Ground-Truth Label Definitions & Positive Class Policy
- **Dataset:** MultiClaim Benchmark (20 responses, 84 extracted atomic claim spans).
- **Ground-Truth Composition:**
  - Ground-Truth Hallucinations (`CONTRADICTED`): **22 claims** ($26.2\%$)
  - Ground-Truth Factual (`SUPPORTED`): **60 claims**
  - Ground-Truth Unverifiable (`UNVERIFIABLE`): **2 claims**
  - **Total Ground-Truth Non-Hallucinated / Factual:** $60 + 2 = \mathbf{62\text{ claims}}$ ($73.8\%$)
- **Binary Classification Policy:**
  - **Positive Class:** Predicted `CONTRADICTED` (claims flagged for hallucination correction).
  - **Negative Class:** Predicted `SUPPORTED` and `UNVERIFIABLE` (claims preserved without modification).
  - *Justification:* In selective correction, the framework does not rewrite unverifiable claims; therefore, only contradicted claims trigger downstream alteration.

### Verification of Claim Integrity
- Claims evaluated: exactly **84**
- Missing claims: **0**
- Duplicate claims: **0**
- Response IDs: `resp_001` through `resp_020` (continuous, verified)

### Exact Mathematical Formulas & Recomputed Values (Result A)

$$\text{TP} = 12, \quad \text{FP} = 3, \quad \text{FN} = 10, \quad \text{TN} = 59 \quad (N = 84)$$

$$\text{Accuracy} = \frac{\text{TP} + \text{TN}}{N} = \frac{12 + 59}{84} = \frac{71}{84} = \mathbf{0.845238} \quad (84.52\%)$$

$$\text{Precision} = \frac{\text{TP}}{\text{TP} + \text{FP}} = \frac{12}{12 + 3} = \frac{12}{15} = \mathbf{0.800000} \quad (80.00\%)$$

$$\text{Recall} = \frac{\text{TP}}{\text{TP} + \text{FN}} = \frac{12}{12 + 10} = \frac{12}{22} = \mathbf{0.545455} \quad (54.55\%)$$

$$\text{F1-Score} = \frac{2 \times \text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}} = \frac{2 \times 0.8 \times 0.545455}{0.8 + 0.545455} = \frac{0.872727}{1.345455} = \mathbf{0.648649} \quad (64.86\%)$$

$$\text{CPR (Claim Preservation Rate)} = \frac{\text{Preserved Factual Claims}}{\text{Total Factual Claims}} = \frac{62 - \text{FP}}{62} = \frac{59}{62} = \mathbf{0.951613} \quad (95.16\%)$$

$$\text{UMR (Unnecessary Modification Rate)} = \frac{\text{Modified Factual Claims}}{\text{Total Factual Claims}} = \frac{\text{FP}}{62} = \frac{3}{62} = \mathbf{0.048387} \quad (4.84\%)$$

$$\text{CPR} + \text{UMR} = 0.951613 + 0.048387 = \mathbf{1.000000} \quad (100.0\%)$$

Every recomputed metric matches Result A to the 6th decimal place.

---

## 4. Audit of Support Threshold Calibration

Calibration results across $\tau_{CSS} \in [0.30, 0.45]$ from `results/audit_archive/support_threshold_calibration_results.json`:

| $\tau_{CSS}$ Threshold | SUPPORTED | CONTRADICTED | UNVERIFIABLE | TP | FP | FN | TN | Precision | Recall | F1-Score | Accuracy | CPR | UMR |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **0.30** | **29** | **15** | **40** | **12** | **3** | **10** | **59** | **0.8000** | **0.5455** | **0.6486** | **0.8452** | **0.9516** | **0.0484** |
| **0.35** | 17 | 15 | 52 | 12 | 3 | 10 | 59 | 0.8000 | 0.5455 | 0.6486 | 0.8452 | 0.9516 | 0.0484 |
| **0.40** | 6 | 17 | 61 | 13 | 4 | 9 | 58 | 0.7647 | 0.5909 | 0.6667 | 0.8452 | 0.9355 | 0.0645 |
| **0.45** | 2 | 17 | 65 | 13 | 4 | 9 | 58 | 0.7647 | 0.5909 | 0.6667 | 0.8452 | 0.9355 | 0.0645 |

### Forensic Analysis of the Calibration Invariance
1. **Identical Metrics for $0.30$ and $0.35$:**
   - As $\tau_{CSS}$ increases from $0.30$ to $0.35$, 12 claims with $\text{CSS} \in [0.30, 0.35)$ lose their `SUPPORTED` label and transition into `UNVERIFIABLE`.
   - None of these 12 claims transition into `CONTRADICTED` because their contradiction probability is lower than the threshold ($\tau_{CON}=0.30$).
   - Because binary evaluation collapses both `SUPPORTED` and `UNVERIFIABLE` into the negative class, the confusion matrix ($\text{TP}=12, \text{FP}=3, \text{FN}=10, \text{TN}=59$) remains identical.
2. **Identical Metrics for $0.40$ and $0.45$:**
   - As $\tau_{CSS}$ increases from $0.40$ to $0.45$, 4 claims shift from `SUPPORTED` to `UNVERIFIABLE`. None transition to `CONTRADICTED`. The confusion matrix remains frozen at $\text{TP}=13, \text{FP}=4, \text{FN}=9, \text{TN}=58$.
3. **Computation Source:**
   - The calibration sweep was computed directly using `results/audit_archive/precomputed_nli_data.pkl`, which stores actual BART-large-MNLI premise-hypothesis sequence-pair inference outputs.
4. **Methodological Limitation to Acknowledge in IEEE Paper:**
   - The same 84 claims (20 multi-claim benchmark samples) were utilized both for threshold calibration and for reporting final performance.
   - *Recommendation for paper:* State clearly that threshold calibration was performed empirically on the benchmark dataset as an exploratory calibration study, and note that validation on an unseen held-out corpus is left for future work.

---

## 5. Visual Consistency Verification of Figures

All figures in [`results/final_figures/`](file:///c:/Users/kickg/OneDrive/Documents/AI_METRIC_ANALYSIS/results/final_figures/) were inspected:

| Figure File | Visual Components & Plotted Data | Alignment with Result A | Alignment with Result B | Audit Status |
| :--- | :--- | :---: | :---: | :---: |
| [`confusion_matrix.png`](file:///c:/Users/kickg/OneDrive/Documents/AI_METRIC_ANALYSIS/results/final_figures/confusion_matrix.png) | Heatmap: $\text{TP}=12, \text{FP}=3, \text{FN}=10, \text{TN}=59$. Annotations: $14.3\%, 3.6\%, 11.9\%, 70.2\%$. | **100% Match** | Contradicts (0% Match) | **PASSED** |
| [`precision_recall_f1.png`](file:///c:/Users/kickg/OneDrive/Documents/AI_METRIC_ANALYSIS/results/final_figures/precision_recall_f1.png) | Bars: Prec = $0.8000$ ($80.0\%$), Rec = $0.5455$ ($54.5\%$), F1 = $0.6486$ ($64.9\%$). | **100% Match** | Contradicts | **PASSED** |
| [`classification_distribution.png`](file:///c:/Users/kickg/OneDrive/Documents/AI_METRIC_ANALYSIS/results/final_figures/classification_distribution.png) | Exploded Pie: SUPPORTED ($29, 34.5\%$), CONTRADICTED ($15, 17.9\%$), UNVERIFIABLE ($40, 47.6\%$). | **100% Match** | Contradicts | **PASSED** |
| [`threshold_calibration.png`](file:///c:/Users/kickg/OneDrive/Documents/AI_METRIC_ANALYSIS/results/final_figures/threshold_calibration.png) | Curves: F1, CPR, UMR across $\tau \in [0.30, 0.45]$; optimal callout box at $\tau=0.30$ ($\text{CPR}=95.2\%, \text{UMR}=4.8\%$). | **100% Match** | N/A | **PASSED** |
| [`before_after_calibration.png`](file:///c:/Users/kickg/OneDrive/Documents/AI_METRIC_ANALYSIS/results/final_figures/before_after_calibration.png) | Grouped Bar: Baseline ($0, 17, 67$) vs. Calibrated ($29, 15, 40$); $+29$ recovered, $-27$ unverifiable reduction. | **100% Match** | Uses Result B as Baseline | **PASSED** |

**Audit Conclusion on Figures:**
All 5 figures are completely synchronized with **Result A**. In Figure 5, Result B is deliberately and correctly used as the uncalibrated "Baseline" ($\tau_{CSS}=0.75$) to showcase the effect of calibration.

---

## 6. Audit of Metric Definitions (CPR & UMR)

### Code Implementation in `evaluation/metrics.py` (lines 520–551)
```python
supported_outcomes = [o for o in outcomes if not o.ground_truth_label]
total_supported = len(supported_outcomes)

if total_supported == 0:
    return 1.0, 0.0

unnecessarily_modified = sum(1 for o in supported_outcomes if o.was_corrected)
preserved = total_supported - unnecessarily_modified
cpr = preserved / total_supported
umr = unnecessarily_modified / total_supported
```

### Explaining the Discrepancy ($\text{CPR}=1.0000$ vs. $\text{CPR}=0.9516$)
1. **Why $\text{CPR}=1.0000, \text{UMR}=0.0000$ appeared in the earlier report (`contradiction_thresholds_comparison.json`):**
   - In that trial run (`compare_thresholds.py`), the local correction backend failed to generate replacement text for the 4 false positives (`was_corrected = False`).
   - Because `was_corrected` was `False`, the code counted $0$ unnecessarily modified claims:
     $$\text{unnecessarily\_modified} = 0 \implies \text{CPR} = \frac{62 - 0}{62} = 1.0000, \quad \text{UMR} = 0.0000$$
2. **Why $\text{CPR}=0.9516, \text{UMR}=0.0484$ is reported in calibration:**
   - In `calibrate_support_thresholds.py`, the fact-substitution module successfully executed rule-based sentence modifications on the 3 false positives (`was_corrected = True`).
   - Consequently, all 3 false positives registered as modified factual claims:
     $$\text{unnecessarily\_modified} = 3 \implies \text{CPR} = \frac{62 - 3}{62} = 0.9516, \quad \text{UMR} = \frac{3}{62} = 0.0484$$
3. **Recommendation for IEEE Paper:**
   - Present $\text{CPR} = 0.9516$ ($95.2\%$) and $\text{UMR} = 0.0484$ ($4.8\%$). This represents the realistic, conservative evaluation where false positive detections actually undergo modification.

---

## 7. Actionable Guidance for IEEE Manuscript Writing

1. **Adopt Result A for the Main Results Table & Text:**
   - State clearly: Precision = **0.8000**, Recall = **0.5455**, F1 = **0.6486**, Accuracy = **0.8452**, CPR = **0.9516**, UMR = **0.0484**.
   - These numbers are 100% congruent with Figures 1 through 5.
2. **Clarify Result B as the Baseline:**
   - Present Result B ($\text{TP}=13, \text{FP}=4, \text{FN}=9, \text{TN}=58, \text{F1}=0.6667$, $0$ supported claims) as the **Uncalibrated Baseline** where contradiction threshold tuning alone was insufficient due to support score suppression.
3. **Note on `multiclaim_metrics.csv` (Result C):**
   - If questioned about `multiclaim_metrics.csv` (which records F1 = $0.7317$, TP = $15$), note that it represents an exploratory experiment incorporating an expanded offline Wikipedia cache. Keep Result A as the core benchmark baseline for paper reproducibility unless regenerating all figures.

---
*Report generated and validated against repository artifacts.*
