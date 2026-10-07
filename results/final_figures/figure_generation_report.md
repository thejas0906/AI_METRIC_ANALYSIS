# Publication Figure Generation Report

**Generated Date:** October 7, 2026  
**Module:** Selective Evidence-Guided Hallucination Correction Framework  
**Output Directory:** `results/final_figures/`  
**Image Resolution:** 300 DPI High-Resolution  

---

## 1. Figures Generated

| Figure | Output File | Dimensions & DPI | Key Visual Components |
|---|---|---|---|
| **Figure 1** | [`confusion_matrix.png`](file:///c:/Users/kickg/OneDrive/Documents/AI_METRIC_ANALYSIS/results/final_figures/confusion_matrix.png) | 6.5 × 5.5 in @ 300 DPI | 2×2 heatmap of binary hallucination detection ($N=84$ total claims), normalized percentages, explicit cell labels (TP=12, FP=3, FN=10, TN=59). |
| **Figure 2** | [`precision_recall_f1.png`](file:///c:/Users/kickg/OneDrive/Documents/AI_METRIC_ANALYSIS/results/final_figures/precision_recall_f1.png) | 6.5 × 5.0 in @ 300 DPI | Publication bar chart comparing Precision (0.8000), Recall (0.5455), and F1 Score (0.6486) with direct value annotations and $0.0-1.0$ y-axis. |
| **Figure 3** | [`classification_distribution.png`](file:///c:/Users/kickg/OneDrive/Documents/AI_METRIC_ANALYSIS/results/final_figures/classification_distribution.png) | 7.0 × 5.5 in @ 300 DPI | Exploded multi-class verification pie chart showing SUPPORTED (29 claims, 34.5%), CONTRADICTED (15 claims, 17.9%), and UNVERIFIABLE (40 claims, 47.6%) with legend. |
| **Figure 4** | [`threshold_calibration.png`](file:///c:/Users/kickg/OneDrive/Documents/AI_METRIC_ANALYSIS/results/final_figures/threshold_calibration.png) | 7.5 × 5.2 in @ 300 DPI | Metric sensitivity curves across CSS support thresholds ($\tau \in [0.30, 0.45]$) plotting F1, CPR, and UMR; vertical dashed line with callout box identifying optimal threshold at $\tau = 0.30$. |
| **Figure 5** | [`before_after_calibration.png`](file:///c:/Users/kickg/OneDrive/Documents/AI_METRIC_ANALYSIS/results/final_figures/before_after_calibration.png) | 7.5 × 5.2 in @ 300 DPI | Grouped comparative bar chart contrasting Baseline ($\tau = 0.75$) vs Calibrated ($\tau = 0.30$), illustrating the recovery of 29 SUPPORTED claims and 27-claim reduction in UNVERIFIABLE claims. |

---

## 2. Source Metrics Used

All values displayed in the figures match the benchmark evaluation:

### A. Detection Performance (Figures 1 & 2)
- **Total Multi-Claim Responses:** 20
- **Total Atomic Claims Evaluated:** 84
- **True Positives (TP):** 12 (Contradicted claims correctly flagged)
- **False Positives (FP):** 3 (Factual claims falsely flagged)
- **False Negatives (FN):** 10 (Contradicted claims missed due to insufficient evidence)
- **True Negatives (TN):** 59 (Factual claims preserved without alteration)
- **Precision:** 0.8000 (80.0%)
- **Recall:** 0.5455 (54.55%)
- **F1 Score:** 0.6486 (64.86%)

### B. Classification Distribution (Figure 3)
- **SUPPORTED:** 29 claims (34.5%)
- **CONTRADICTED:** 15 claims (17.9%)
- **UNVERIFIABLE:** 40 claims (47.6%)
- **Total:** 84 claims (100.0%)

### C. Threshold Calibration Sweep (Figure 4)
| $\tau_{CSS}$ Threshold | Detection F1 | Claim Preservation Rate (CPR) | Unnecessary Modification Rate (UMR) | Selection Rationale |
|:---:|:---:|:---:|:---:|---|
| **0.30** | **0.6486** | **0.9516** | **0.0484** | **Optimal calibration point: Maximizes CPR (95.2%) while maintaining strong F1** |
| 0.35 | 0.6486 | 0.9516 | 0.0484 | Identical performance boundary |
| 0.40 | 0.6667 | 0.9355 | 0.0645 | Marginal F1 gain (+0.0181) offset by safety degradation (-1.61% CPR) |
| 0.45 | 0.6667 | 0.9355 | 0.0645 | CPR drop remains below optimal safety target |

### D. Calibration Transition (Figure 5)
| Class | Baseline ($\tau=0.75$) | Calibrated ($\tau=0.30$) | Net Claim Delta |
|---|:---:|:---:|:---:|
| **SUPPORTED** | 0 | 29 | **+29 claims recovered** |
| **CONTRADICTED** | 17 | 15 | -2 claims (borderline items classified UNVERIFIABLE) |
| **UNVERIFIABLE** | 67 | 40 | **-27 claims resolved** |

---

## 3. Confirmation of Audit Files Archived

A total of **26 diagnostic and investigation artifacts** were relocated into [`results/audit_archive/`](file:///c:/Users/kickg/OneDrive/Documents/AI_METRIC_ANALYSIS/results/audit_archive/):

1. `all_unverifiable_diagnostic.json`
2. `audit7_unverifiable_diagnosis.json`
3. `baseline_multiclaim_results.json`
4. `categorized_unverifiable.json`
5. `contradiction_thresholds_comparison.csv`
6. `contradiction_thresholds_comparison.json`
7. `css_thresholds_evaluation.json`
8. `current_audit_results.json`
9. `demo_diagnostic_records.json`
10. `demo_results.json`
11. `extraction_audit_report.json`
12. `forensic_bucketed_claims.json`
13. `forensic_unverifiable_master.json`
14. `fresh_audit_summary.json`
15. `hybrid_multiclaim_results.json`
16. `new_forensic_audit_all_84.json`
17. `new_forensic_audit_master.json`
18. `new_unverifiable_claims.json`
19. `precomputed_nli_data.pkl`
20. `simulated_options.json`
21. `support_diagnostic_summary.json`
22. `support_threshold_calibration_results.json`
23. `temp_audit_40.txt`
24. `threshold030_multiclaim_results.json`
25. `threshold_grid_search.csv`
26. `unverifiable_audit_temp.json`

---

## 4. Files Skipped / Preserved in Place

The following primary benchmark files were strictly preserved in `results/` and **NOT** moved:
- [`multiclaim_results.json`](file:///c:/Users/kickg/OneDrive/Documents/AI_METRIC_ANALYSIS/results/multiclaim_results.json): Primary ground truth evaluation output containing response reports and correction outcomes.
- [`multiclaim_metrics.csv`](file:///c:/Users/kickg/OneDrive/Documents/AI_METRIC_ANALYSIS/results/multiclaim_metrics.csv): Benchmark metrics tracking accuracy, precision, recall, F1, CPR, UMR, and verifier consistency.
- `.gitkeep`: Retained to maintain git directory structure.
