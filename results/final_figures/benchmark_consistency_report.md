# Benchmark Consistency Verification Report

**Verification Date:** October 7, 2026  
**Artifact Inspected:** [`results/final_figures/confusion_matrix.png`](file:///c:/Users/kickg/OneDrive/Documents/AI_METRIC_ANALYSIS/results/final_figures/confusion_matrix.png)  
**Cross-Referenced Sources:**
- [`results/final_figures/figure_generation_report.md`](file:///c:/Users/kickg/OneDrive/Documents/AI_METRIC_ANALYSIS/results/final_figures/figure_generation_report.md)
- [`results/final_figures/threshold_calibration.png`](file:///c:/Users/kickg/OneDrive/Documents/AI_METRIC_ANALYSIS/results/final_figures/threshold_calibration.png)
- [`results/final_figures/precision_recall_f1.png`](file:///c:/Users/kickg/OneDrive/Documents/AI_METRIC_ANALYSIS/results/final_figures/precision_recall_f1.png)
- [`results/audit_archive/threshold030_multiclaim_results.json`](file:///c:/Users/kickg/OneDrive/Documents/AI_METRIC_ANALYSIS/results/audit_archive/threshold030_multiclaim_results.json)
- [`results/multiclaim_results.json`](file:///c:/Users/kickg/OneDrive/Documents/AI_METRIC_ANALYSIS/results/multiclaim_results.json)

---

## 1. Benchmark Run Verification

### Candidate Options Evaluated

| Run Variant | True Positives (TP) | False Positives (FP) | False Negatives (FN) | True Negatives (TN) | Total ($N$) | Precision | Recall | F1 Score | Status |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|---|
| **OPTION A**<br>*(Final Calibrated Run)* | **12** | **3** | **10** | **59** | **84** | **0.8000** | **0.5455** | **0.6486** | **CONFIRMED ACTIVE** in Figure 1 |
| **OPTION B**<br>*(Older Baseline Run)* | 13 | 4 | 9 | 58 | 84 | 0.7647 | 0.5909 | 0.6667 | Historical uncalibrated run |

---

## 2. Cross-Figure Consistency Analysis

All 5 figures in [`results/final_figures/`](file:///c:/Users/kickg/OneDrive/Documents/AI_METRIC_ANALYSIS/results/final_figures/) were verified against the underlying mathematical definitions:

1. **Figure 1 ([`confusion_matrix.png`](file:///c:/Users/kickg/OneDrive/Documents/AI_METRIC_ANALYSIS/results/final_figures/confusion_matrix.png)):**
   - Renders **$\text{TP} = 12$**, **$\text{FP} = 3$**, **$\text{FN} = 10$**, **$\text{TN} = 59$** (Total $N = 84$).
   - Matches Option A exactly.
2. **Figure 2 ([`precision_recall_f1.png`](file:///c:/Users/kickg/OneDrive/Documents/AI_METRIC_ANALYSIS/results/final_figures/precision_recall_f1.png)):**
   - $\text{Precision} = \frac{\text{TP}}{\text{TP} + \text{FP}} = \frac{12}{12 + 3} = \mathbf{0.8000}$ (80.0%)
   - $\text{Recall} = \frac{\text{TP}}{\text{TP} + \text{FN}} = \frac{12}{12 + 10} = \mathbf{0.5455}$ (54.55%)
   - $\text{F1 Score} = \frac{2 \times \text{Prec} \times \text{Rec}}{\text{Prec} + \text{Rec}} = \mathbf{0.6486}$ (64.86%)
   - Perfect mathematical consistency with Figure 1.
3. **Figure 3 ([`classification_distribution.png`](file:///c:/Users/kickg/OneDrive/Documents/AI_METRIC_ANALYSIS/results/final_figures/classification_distribution.png)):**
   - Predicted Contradicted = $\text{TP} + \text{FP} = 12 + 3 = \mathbf{15}$.
   - Shows SUPPORTED (29), CONTRADICTED (15), UNVERIFIABLE (40).
   - Consistent with Option A.
4. **Figure 4 ([`threshold_calibration.png`](file:///c:/Users/kickg/OneDrive/Documents/AI_METRIC_ANALYSIS/results/final_figures/threshold_calibration.png)):**
   - Plots calibration curve identifying optimal threshold $\tau_{CSS} = 0.30$ yielding $\text{F1} = \mathbf{0.6486}$.
   - Directly corresponds to Option A detection performance.
5. **Figure 5 ([`before_after_calibration.png`](file:///c:/Users/kickg/OneDrive/Documents/AI_METRIC_ANALYSIS/results/final_figures/before_after_calibration.png)):**
   - Final calibrated state corresponds to the 29 SUPPORTED, 15 CONTRADICTED, 40 UNVERIFIABLE partition.

---

## 3. Verification Outcome & Action

- **Verification Outcome:** Passed. Figure 1 correctly displays **Option A (TP=12, FP=3, FN=10, TN=59)**.
- **Regeneration Required:** **No**. The figure already uses the correct calibrated run, perfectly aligned with Figures 2, 3, 4, and 5.
- **Conclusion:** Complete internal consistency is preserved across all visualization artifacts in `results/final_figures/`.
