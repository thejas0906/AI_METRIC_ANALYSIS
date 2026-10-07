"""
generate_final_figures.py
=========================
Generates all 5 publication-ready benchmark figures for the
Selective Evidence-Guided Hallucination Correction Framework.

Output Directory: results/final_figures/
Resolution: 300 DPI
"""

import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

output_dir = os.path.join("results", "final_figures")
os.makedirs(output_dir, exist_ok=True)

# Set global publication styling
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['DejaVu Sans', 'Arial', 'Helvetica', 'Liberation Sans'],
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.labelweight': 'bold',
    'axes.titlesize': 13,
    'axes.titleweight': 'bold',
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.titlesize': 14,
    'figure.titleweight': 'bold',
    'figure.autolayout': False,
})

# ==============================================================================
# FIGURE 1: Confusion Matrix
# TP = 12, FP = 3, FN = 10, TN = 59 (Total = 84)
# ==============================================================================
def create_figure1_confusion_matrix():
    print("Generating Figure 1: Confusion Matrix...")
    fig, ax = plt.subplots(figsize=(6.5, 5.5), dpi=300)
    
    # Matrix layout:
    #                 Predicted Hallucinated    Predicted Factual
    # Actual Hallucinated       TP (12)                 FN (10)
    # Actual Factual            FP (3)                  TN (59)
    cm = np.array([[12, 10],
                   [3, 59]])
    
    # Custom annotations with rates
    total = 84
    annot = np.array([
        [f"True Positive (TP)\n{cm[0, 0]}\n({cm[0, 0]/total*100:.1f}%)",
         f"False Negative (FN)\n{cm[0, 1]}\n({cm[0, 1]/total*100:.1f}%)"],
        [f"False Positive (FP)\n{cm[1, 0]}\n({cm[1, 0]/total*100:.1f}%)",
         f"True Negative (TN)\n{cm[1, 1]}\n({cm[1, 1]/total*100:.1f}%)"]
    ])
    
    cmap = sns.light_palette("#1f77b4", as_cmap=True)
    sns.heatmap(cm, annot=annot, fmt="", cmap=cmap, cbar=True,
                xticklabels=["Hallucinated\n(Contradicted)", "Factual\n(Supported/Unverif)"],
                yticklabels=["Hallucinated\n(Contradicted)", "Factual\n(Supported/Unverif)"],
                annot_kws={"fontsize": 11, "weight": "medium"},
                linewidths=1.5, linecolor='white', square=True, ax=ax)
    
    ax.set_title("Hallucination Detection Confusion Matrix\n(Total Claims N = 84)", pad=16)
    ax.set_xlabel("Predicted Label", labelpad=10)
    ax.set_ylabel("Ground Truth Label", labelpad=10)
    
    plt.tight_layout()
    fig_path = os.path.join(output_dir, "confusion_matrix.png")
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  -> Saved {fig_path}")

# ==============================================================================
# FIGURE 2: Precision / Recall / F1 Bar Chart
# Precision = 0.8000, Recall = 0.5455, F1 Score = 0.6486
# ==============================================================================
def create_figure2_prf1():
    print("Generating Figure 2: Precision / Recall / F1 Bar Chart...")
    fig, ax = plt.subplots(figsize=(6.5, 5.0), dpi=300)
    
    metrics = ["Precision", "Recall", "F1 Score"]
    values = [0.8000, 0.5455, 0.6486]
    colors = ["#2b5c8f", "#d95f02", "#2ca02c"]
    
    bars = ax.bar(metrics, values, color=colors, width=0.52, edgecolor='#333333', linewidth=1.2, zorder=3)
    
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score (0.00 – 1.00)")
    ax.set_title("Hallucination Detection Performance Metrics\n(Benchmark Multi-Claim Evaluation)", pad=14)
    ax.grid(axis='y', linestyle='--', alpha=0.6, zorder=0)
    ax.set_axisbelow(True)
    
    # Value labels on top of bars
    for bar, val in zip(bars, values):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2.0, height + 0.025,
                f"{val:.4f}\n({val*100:.1f}%)",
                ha='center', va='bottom', fontsize=11, fontweight='bold', color='#1a1a1a')
        
    # Baseline reference note
    ax.axhline(y=0.5, color='#999999', linestyle=':', linewidth=1.0, zorder=1)
    
    plt.tight_layout()
    fig_path = os.path.join(output_dir, "precision_recall_f1.png")
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  -> Saved {fig_path}")

# ==============================================================================
# FIGURE 3: Classification Distribution Pie Chart
# SUPPORTED = 29, CONTRADICTED = 15, UNVERIFIABLE = 40 (Total = 84)
# ==============================================================================
def create_figure3_distribution():
    print("Generating Figure 3: Classification Distribution Pie Chart...")
    fig, ax = plt.subplots(figsize=(7.0, 5.5), dpi=300)
    
    labels = ["SUPPORTED", "CONTRADICTED", "UNVERIFIABLE"]
    counts = [29, 15, 40]
    total = sum(counts)
    colors = ["#2ca02c", "#d62728", "#7f7f7f"]
    explode = (0.04, 0.06, 0.02)
    
    def make_autopct(values):
        def my_autopct(pct):
            val = int(round(pct * total / 100.0))
            return f"{pct:.1f}%\n(n={val})"
        return my_autopct
    
    wedges, texts, autotexts = ax.pie(
        counts,
        explode=explode,
        labels=labels,
        colors=colors,
        autopct=make_autopct(counts),
        startangle=140,
        wedgeprops={"edgecolor": "white", "linewidth": 1.5, "antialiased": True},
        textprops={"fontsize": 11, "weight": "bold"},
        pctdistance=0.62
    )
    
    for at in autotexts:
        at.set_color('white')
        at.set_fontsize(10.5)
        at.set_weight('bold')
    
    ax.set_title("Distribution of Verification Decisions\n(Total Claims N = 84)", pad=16)
    
    legend_labels = [f"{lbl}: {cnt} claims ({cnt/total*100:.1f}%)" for lbl, cnt in zip(labels, counts)]
    ax.legend(wedges, legend_labels, title="Verification Classes", loc="center left", bbox_to_anchor=(1, 0, 0.5, 1), frameon=True)
    
    plt.tight_layout()
    fig_path = os.path.join(output_dir, "classification_distribution.png")
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  -> Saved {fig_path}")

# ==============================================================================
# FIGURE 4: Threshold Calibration Graph
# Thresholds: [0.30, 0.35, 0.40, 0.45]
# F1:  [0.6486, 0.6486, 0.6667, 0.6667]
# CPR: [0.9516, 0.9516, 0.9355, 0.9355]
# UMR: [0.0484, 0.0484, 0.0645, 0.0645]
# Recommended: 0.30
# ==============================================================================
def create_figure4_calibration():
    print("Generating Figure 4: Threshold Calibration Graph...")
    fig, ax = plt.subplots(figsize=(7.5, 5.2), dpi=300)
    
    thresholds = [0.30, 0.35, 0.40, 0.45]
    f1_scores   = [0.6486, 0.6486, 0.6667, 0.6667]
    cpr_scores  = [0.9516, 0.9516, 0.9355, 0.9355]
    umr_scores  = [0.0484, 0.0484, 0.0645, 0.0645]
    
    # Plot lines with distinct markers
    line_cpr, = ax.plot(thresholds, cpr_scores, marker='s', markersize=8, linewidth=2.4, color='#1f77b4', label='CPR (Claim Preservation Rate)')
    line_f1,  = ax.plot(thresholds, f1_scores,  marker='o', markersize=8, linewidth=2.4, color='#2ca02c', label='F1 Score (Detection)')
    line_umr, = ax.plot(thresholds, umr_scores, marker='^', markersize=8, linewidth=2.4, color='#d62728', label='UMR (Unnecessary Modification Rate)')
    
    # Highlight recommended threshold at 0.30
    ax.axvline(x=0.30, color='#888888', linestyle='--', linewidth=1.8, zorder=2)
    ax.annotate(
        "Optimal Threshold\n(τ = 0.30)\nMax CPR (95.2%)\nMin UMR (4.8%)",
        xy=(0.30, 0.9516),
        xytext=(0.32, 0.78),
        arrowprops=dict(facecolor='#1f77b4', shrink=0.08, width=1.5, headwidth=7),
        fontsize=10, fontweight='bold', bbox=dict(boxstyle="round,pad=0.4", fc="#f0f8ff", ec="#1f77b4", lw=1.2)
    )
    
    # Data labels on points
    for x, y in zip(thresholds, cpr_scores):
        ax.text(x, y + 0.02, f"{y:.4f}", ha='center', fontsize=9, color='#1f77b4', fontweight='bold')
    for x, y in zip(thresholds, f1_scores):
        ax.text(x, y - 0.035, f"{y:.4f}", ha='center', fontsize=9, color='#2ca02c', fontweight='bold')
    for x, y in zip(thresholds, umr_scores):
        ax.text(x, y + 0.02, f"{y:.4f}", ha='center', fontsize=9, color='#d62728', fontweight='bold')
        
    ax.set_xlim(0.28, 0.47)
    ax.set_ylim(0.0, 1.05)
    ax.set_xticks(thresholds)
    ax.set_xlabel("CSS Supported Threshold (τ_CSS)")
    ax.set_ylabel("Metric Value (0.00 – 1.00)")
    ax.set_title("Performance Trade-off Across CSS Support Thresholds\n(Optimal Safety Balance at τ = 0.30)", pad=14)
    ax.grid(True, linestyle=':', alpha=0.6)
    ax.legend(loc='center right', frameon=True, shadow=False)
    
    plt.tight_layout()
    fig_path = os.path.join(output_dir, "threshold_calibration.png")
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  -> Saved {fig_path}")

# ==============================================================================
# FIGURE 5: Before vs After Calibration
# Baseline: SUPPORTED = 0,  CONTRADICTED = 17, UNVERIFIABLE = 67
# Final:    SUPPORTED = 29, CONTRADICTED = 15, UNVERIFIABLE = 40
# ==============================================================================
def create_figure5_before_after():
    print("Generating Figure 5: Before vs After Calibration...")
    fig, ax = plt.subplots(figsize=(7.5, 5.2), dpi=300)
    
    categories = ["SUPPORTED", "CONTRADICTED", "UNVERIFIABLE"]
    baseline_counts = [0, 17, 67]
    final_counts    = [29, 15, 40]
    
    x = np.arange(len(categories))
    width = 0.35
    
    rects1 = ax.bar(x - width/2, baseline_counts, width, label='Baseline (τ = 0.75)',
                    color='#9e9e9e', edgecolor='#424242', linewidth=1.2, zorder=3)
    rects2 = ax.bar(x + width/2, final_counts, width, label='Calibrated (τ = 0.30)',
                    color='#1f77b4', edgecolor='#0d47a1', linewidth=1.2, zorder=3)
    
    ax.set_ylabel("Number of Claims (N = 84)")
    ax.set_title("Verification Decision Distribution: Baseline vs Calibrated Pipeline\n(Resolving Unverifiable Suppression)", pad=14)
    ax.set_xticks(x)
    ax.set_xticklabels(categories, fontweight='bold')
    ax.set_ylim(0, 80)
    ax.grid(axis='y', linestyle='--', alpha=0.6, zorder=0)
    ax.set_axisbelow(True)
    ax.legend(frameon=True)
    
    # Annotations on bars
    for r in rects1:
        h = r.get_height()
        ax.text(r.get_x() + r.get_width()/2., h + 1.2, f"{int(h)}",
                ha='center', va='bottom', fontsize=10, fontweight='bold', color='#424242')
        
    for r in rects2:
        h = r.get_height()
        ax.text(r.get_x() + r.get_width()/2., h + 1.2, f"{int(h)}",
                ha='center', va='bottom', fontsize=10, fontweight='bold', color='#0d47a1')
        
    # Key trend annotations
    # 1. Recovery of SUPPORTED
    ax.annotate("+29 Recovered\n(0 -> 29)",
                xy=(x[0] + width/2, 29),
                xytext=(x[0] + 0.15, 46),
                arrowprops=dict(facecolor='#2ca02c', shrink=0.08, width=1.4, headwidth=6),
                fontsize=9.5, fontweight='bold', color='#1b5e20',
                bbox=dict(boxstyle="round,pad=0.3", fc="#e8f5e9", ec="#2ca02c"))
    
    # 2. Reduction of UNVERIFIABLE
    ax.annotate("-27 Reduction\n(67 -> 40)",
                xy=(x[2] + width/2, 40),
                xytext=(x[2] - 0.28, 56),
                arrowprops=dict(facecolor='#d32f2f', shrink=0.08, width=1.4, headwidth=6),
                fontsize=9.5, fontweight='bold', color='#b71c1c',
                bbox=dict(boxstyle="round,pad=0.3", fc="#ffebee", ec="#d32f2f"))
    
    plt.tight_layout()
    fig_path = os.path.join(output_dir, "before_after_calibration.png")
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  -> Saved {fig_path}")

if __name__ == "__main__":
    create_figure1_confusion_matrix()
    create_figure2_prf1()
    create_figure3_distribution()
    create_figure4_calibration()
    create_figure5_before_after()
    print("\nAll 5 publication-ready figures successfully generated in results/final_figures/")
