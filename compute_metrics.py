"""
compute_metrics.py
===================
Standalone Metrics Calculation Script
---------------------------------------
Reads pipeline output results from a JSON file and computes
all evaluation metrics independently.

This script can be used to:
1. Recompute metrics from saved experiment results.
2. Compare multiple experiment configurations.
3. Generate a formatted LaTeX table for paper submission.
4. Visualize metrics as a bar chart.

Usage:
    # Compute metrics from saved JSON results
    python compute_metrics.py --input results/fever_results.json

    # Compare multiple result files
    python compute_metrics.py --compare results/demo_results.json results/fever_results.json

    # Generate LaTeX table
    python compute_metrics.py --input results/fever_results.json --latex

    # Save metrics as CSV
    python compute_metrics.py --input results/fever_results.json --csv metrics_output.csv
"""

import os
import sys
import json
import argparse
from pathlib import Path
from loguru import logger
from colorama import init, Fore, Style

init(autoreset=True)

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from evaluation.metrics import EvaluationMetrics, MetricsResult


# ------------------------------------------------------------------
# Argument Parser
# ------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Compute evaluation metrics from saved experiment results",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--input",
        type=str,
        help="Path to a JSON file containing pipeline outputs.",
    )
    parser.add_argument(
        "--compare",
        nargs="+",
        help="Paths to multiple JSON files to compare.",
    )
    parser.add_argument(
        "--latex",
        action="store_true",
        help="Output a LaTeX table for paper submission.",
    )
    parser.add_argument(
        "--csv",
        type=str,
        help="Save metrics to a CSV file.",
    )
    parser.add_argument(
        "--chart",
        action="store_true",
        help="Display a bar chart of metrics (requires matplotlib).",
    )
    return parser.parse_args()


# ------------------------------------------------------------------
# Metrics Loading & Computation
# ------------------------------------------------------------------

def load_and_compute(json_path: str) -> MetricsResult:
    """
    Load pipeline outputs from JSON and compute metrics.

    Expected JSON format (from run_experiment.py):
    {
        "per_claim_results": [
            {
                "claim": "...",
                "ground_truth": true/false,
                "predicted": true/false,
                "was_corrected": true/false,
                "post_correction_verified": true/false,
                "was_preserved": true/false
            },
            ...
        ]
    }

    Args:
        json_path: Path to the results JSON file.

    Returns:
        MetricsResult with all computed metrics.
    """
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Handle both flat list and nested format
    if isinstance(data, list):
        pipeline_outputs = data
    elif "correction_outcomes" in data and data["correction_outcomes"]:
        calc = EvaluationMetrics()
        return calc.compute_from_pipeline_output(data["correction_outcomes"])
    elif "per_claim_results" in data:
        pipeline_outputs = data["per_claim_results"]
    elif "metrics" in data and "per_claim_results" not in data:
        # Already computed metrics — reconstruct MetricsResult
        logger.info("Found pre-computed metrics in JSON.")
        m = data["metrics"]
        gt_m = m.get("ground_truth_metrics", {})
        ver_m = m.get("verifier_metrics", {})
        return MetricsResult(
            accuracy=gt_m.get("accuracy", m.get("accuracy", 0)),
            precision=gt_m.get("precision", m.get("precision", 0)),
            recall=gt_m.get("recall", m.get("recall", 0)),
            f1_score=gt_m.get("f1_score", m.get("f1_score", 0)),
            csr_gt=gt_m.get("csr_gt", m.get("csr_gt")),
            cpr_gt=gt_m.get("cpr_gt", m.get("cpr_gt", m.get("cpr", 0))),
            umr_gt=gt_m.get("umr_gt", m.get("umr_gt", m.get("umr", 0))),
            fra_gt=gt_m.get("fra_gt", m.get("fra_gt")),
            rps=gt_m.get("rps", m.get("rps", 1.0)),
            csr_verifier=ver_m.get("csr_verifier", m.get("csr_verifier", 0)),
            acceptance_rate=ver_m.get("acceptance_rate", m.get("acceptance_rate", 0)),
            verification_pass_rate=ver_m.get("verification_pass_rate", m.get("verification_pass_rate", 0)),
            fra_verifier=ver_m.get("fra_verifier", m.get("fra_verifier", 0)),
            csr=m.get("csr", 0),
            cpr=m.get("cpr", 0),
            umr=m.get("umr", 0),
            fra=m.get("fra", 0),
        )
    else:
        pipeline_outputs = []

    if not pipeline_outputs:
        logger.warning("No per-claim results found in JSON file.")
        return MetricsResult(
            accuracy=0, precision=0, recall=0, f1_score=0,
            csr=0, cpr=0, umr=0, fra=0,
        )

    # Compute metrics using the EvaluationMetrics class
    calc = EvaluationMetrics()
    return calc.compute_from_pipeline_output(pipeline_outputs)


# ------------------------------------------------------------------
# Output Formatters
# ------------------------------------------------------------------

def print_comparison_table(results: dict) -> None:
    """
    Print a side-by-side comparison table of metrics from multiple experiments,
    distinguishing Ground-Truth from Internal-Verifier metrics.

    Args:
        results: Dict mapping experiment name → MetricsResult.
    """
    try:
        from tabulate import tabulate
    except ImportError:
        print("Install tabulate for formatted tables: pip install tabulate")
        return

    metrics_list = [
        # (Label, Attribute, Category)
        ("HDA (Accuracy)", "accuracy", "Ground Truth"),
        ("Precision", "precision", "Ground Truth"),
        ("Recall", "recall", "Ground Truth"),
        ("F1 Score", "f1_score", "Ground Truth"),
        ("CSR_GT (Ground Truth)", "csr_gt", "Ground Truth"),
        ("CPR_GT (Claim Preservation)", "cpr_gt", "Ground Truth"),
        ("UMR_GT (Unnecessary Modification)", "umr_gt", "Ground Truth"),
        ("RPS (Response Preservation)", "rps", "Ground Truth"),
        ("FRA_GT (Final Response - GT)", "fra_gt", "Ground Truth"),
        ("CSR_Verifier (Self-Verification)", "csr_verifier", "Internal Verifier"),
        ("Acceptance Rate", "acceptance_rate", "Internal Verifier"),
        ("Verification Pass Rate", "verification_pass_rate", "Internal Verifier"),
        ("FRA_Verifier (Final - Verifier)", "fra_verifier", "Internal Verifier"),
    ]
    headers = ["Category", "Metric"] + list(results.keys())

    rows = []
    for label, attr, cat in metrics_list:
        row = [cat, label]
        for exp_name, mr in results.items():
            val = getattr(mr, attr, None)
            if val is None:
                row.append("N/A")
            elif isinstance(val, (int, float)):
                row.append(f"{val:.4f}")
            else:
                row.append(str(val))
        rows.append(row)

    print(f"\n{Fore.CYAN}{'=' * 75}")
    print("  METRICS COMPARISON TABLE (Ground-Truth vs Internal Verifier)")
    print(f"{'=' * 75}{Style.RESET_ALL}")
    print(tabulate(rows, headers=headers, tablefmt="grid"))


def print_latex_table(results: dict) -> None:
    """
    Print a LaTeX table suitable for paper submission with metric separation.

    Args:
        results: Dict mapping experiment name → MetricsResult.
    """
    exp_names = list(results.keys())
    col_header = " & ".join(exp_names)

    print("\n% LaTeX Table — Hallucination Correction Metrics")
    print("% Generated by compute_metrics.py (Ground-Truth vs Verifier Separated)")
    print("\\begin{table}[h]")
    print("\\centering")
    print(f"\\begin{{tabular}}{{ll{'c' * len(exp_names)}}}")
    print("\\hline")
    print(f"\\textbf{{Type}} & \\textbf{{Metric}} & {col_header} \\\\")
    print("\\hline")

    metrics_data = [
        ("Ground Truth", "HDA (Accuracy)", "accuracy"),
        ("Ground Truth", "Precision", "precision"),
        ("Ground Truth", "Recall", "recall"),
        ("Ground Truth", "F1 Score", "f1_score"),
        ("Ground Truth", "CSR\\_GT", "csr_gt"),
        ("Ground Truth", "CPR\\_GT", "cpr_gt"),
        ("Ground Truth", "UMR\\_GT", "umr_gt"),
        ("Ground Truth", "RPS", "rps"),
        ("Ground Truth", "FRA\\_GT", "fra_gt"),
        ("Verifier", "CSR\\_Verifier", "csr_verifier"),
        ("Verifier", "Acceptance Rate", "acceptance_rate"),
        ("Verifier", "Verification Pass Rate", "verification_pass_rate"),
        ("Verifier", "FRA\\_Verifier", "fra_verifier"),
    ]

    for cat, label, attr in metrics_data:
        vals = []
        for mr in results.values():
            v = getattr(mr, attr, None)
            vals.append(f"{v:.4f}" if isinstance(v, (int, float)) else "N/A")
        values_str = " & ".join(vals)
        print(f"{cat} & {label} & {values_str} \\\\")

    print("\\hline")
    print("\\end{tabular}")
    print(f"\\caption{{Evaluation results distinguishing external ground-truth metrics from internal verifier metrics.}}")
    print("\\label{tab:metrics}")
    print("\\end{table}")


def save_to_csv(results: dict, csv_path: str) -> None:
    """
    Save comparison metrics to a CSV file with explicit Evaluation Type.

    Args:
        results:  Dict mapping experiment name → MetricsResult.
        csv_path: Output CSV file path.
    """
    try:
        import pandas as pd
    except ImportError:
        logger.warning("pandas not available — using plain CSV writer.")
        import csv
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["Category", "Evaluation Type", "Metric"] + list(results.keys()))
            metrics_data = [
                ("Ground Truth", "ground_truth", "Accuracy (HDA)", "accuracy"),
                ("Ground Truth", "ground_truth", "Precision", "precision"),
                ("Ground Truth", "ground_truth", "Recall", "recall"),
                ("Ground Truth", "ground_truth", "F1 Score", "f1_score"),
                ("Ground Truth", "ground_truth", "CSR_GT", "csr_gt"),
                ("Ground Truth", "ground_truth", "CPR_GT", "cpr_gt"),
                ("Ground Truth", "ground_truth", "UMR_GT", "umr_gt"),
                ("Ground Truth", "ground_truth", "FRA_GT", "fra_gt"),
                ("Internal Verifier", "internal_verifier", "CSR_Verifier", "csr_verifier"),
                ("Internal Verifier", "internal_verifier", "Acceptance Rate", "acceptance_rate"),
                ("Internal Verifier", "internal_verifier", "Verification Pass Rate", "verification_pass_rate"),
                ("Internal Verifier", "internal_verifier", "FRA_Verifier", "fra_verifier"),
            ]
            for cat, ev_type, label, attr in metrics_data:
                row = [cat, ev_type, label] + [
                    f"{getattr(mr, attr):.4f}" if isinstance(getattr(mr, attr, None), (int, float)) else "N/A"
                    for mr in results.values()
                ]
                writer.writerow(row)
        print(f"Saved to {csv_path}")
        return

    rows = []
    metrics_data = [
        ("Ground Truth", "ground_truth", "Accuracy (HDA)", "accuracy"),
        ("Ground Truth", "ground_truth", "Precision", "precision"),
        ("Ground Truth", "ground_truth", "Recall", "recall"),
        ("Ground Truth", "ground_truth", "F1 Score", "f1_score"),
        ("Ground Truth", "ground_truth", "CSR_GT", "csr_gt"),
        ("Ground Truth", "ground_truth", "CPR_GT", "cpr_gt"),
        ("Ground Truth", "ground_truth", "UMR_GT", "umr_gt"),
        ("Ground Truth", "ground_truth", "FRA_GT", "fra_gt"),
        ("Internal Verifier", "internal_verifier", "CSR_Verifier", "csr_verifier"),
        ("Internal Verifier", "internal_verifier", "Acceptance Rate", "acceptance_rate"),
        ("Internal Verifier", "internal_verifier", "Verification Pass Rate", "verification_pass_rate"),
        ("Internal Verifier", "internal_verifier", "FRA_Verifier", "fra_verifier"),
    ]
    for cat, ev_type, label, attr in metrics_data:
        row = {"Category": cat, "Evaluation Type": ev_type, "Metric": label}
        for exp_name, mr in results.items():
            val = getattr(mr, attr, None)
            row[exp_name] = round(val, 4) if isinstance(val, (int, float)) else "N/A"
        rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(csv_path, index=False)
    print(f"\n{Fore.GREEN}Metrics saved to {csv_path}{Style.RESET_ALL}")

    df = pd.DataFrame(rows)
    df.to_csv(csv_path, index=False)
    print(f"\n{Fore.GREEN}Metrics saved to {csv_path}{Style.RESET_ALL}")


def show_chart(results: dict) -> None:
    """
    Display a grouped bar chart of metrics.

    Args:
        results: Dict mapping experiment name → MetricsResult.
    """
    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print("Install matplotlib for charts: pip install matplotlib")
        return

    metrics_labels = ["Accuracy", "Precision", "Recall", "F1", "CSR", "CPR", "FRA"]
    metric_attrs   = ["accuracy", "precision", "recall", "f1_score", "csr", "cpr", "fra"]

    exp_names = list(results.keys())
    x = np.arange(len(metrics_labels))
    width = 0.8 / max(len(exp_names), 1)

    fig, ax = plt.subplots(figsize=(12, 6))

    colors = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B3"]

    for i, (exp_name, mr) in enumerate(results.items()):
        values = [getattr(mr, attr) for attr in metric_attrs]
        offset = (i - len(exp_names) / 2) * width + width / 2
        bars = ax.bar(
            x + offset, values, width,
            label=exp_name,
            color=colors[i % len(colors)],
            alpha=0.87,
            edgecolor="white",
        )
        # Add value labels on bars
        for bar, val in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.01,
                f"{val:.3f}",
                ha="center", va="bottom", fontsize=7,
            )

    ax.set_xlabel("Metric", fontsize=12)
    ax.set_ylabel("Score", fontsize=12)
    ax.set_title(
        "Selective Evidence-Guided Hallucination Correction Framework\nEvaluation Metrics",
        fontsize=13, fontweight="bold",
    )
    ax.set_xticks(x)
    ax.set_xticklabels(metrics_labels, fontsize=10)
    ax.set_ylim(0, 1.15)
    ax.legend(loc="upper right", fontsize=10)
    ax.grid(axis="y", alpha=0.4)

    plt.tight_layout()
    plt.savefig("results/metrics_chart.png", dpi=150, bbox_inches="tight")
    print("Chart saved to results/metrics_chart.png")
    plt.show()


# ------------------------------------------------------------------
# Standalone Demo Metrics (for testing without saved results)
# ------------------------------------------------------------------

def compute_demo_metrics() -> MetricsResult:
    """
    Compute metrics on a hand-crafted demo example for testing.

    Returns:
        MetricsResult from the demo evaluation.
    """
    from evaluation.metrics import EvaluationMetrics, CorrectionOutcome

    # Simulated ground-truth and predictions for 10 claims
    demo_outcomes = [
        CorrectionOutcome("Claim 1: Einstein born 1879",    False, False, False, False, True, True),
        CorrectionOutcome("Claim 2: Nobel Prize 1925",      True,  True,  True,  True,  False, False),
        CorrectionOutcome("Claim 3: Speed of light 300000", False, False, False, False, True, True),
        CorrectionOutcome("Claim 4: Curie won 2 Nobels",    False, False, False, False, True, True),
        CorrectionOutcome("Claim 5: Harvard University",    True,  True,  True,  True,  False, False),
        CorrectionOutcome("Claim 6: Polonium discovered",   False, False, False, False, True, True),
        CorrectionOutcome("Claim 7: Oxford University",     True,  True,  True,  False, False, False),
        CorrectionOutcome("Claim 8: c is constant",         False, False, False, False, True, True),
        CorrectionOutcome("Claim 9: Born in Warsaw",        False, True,  False, False, False, True),
        CorrectionOutcome("Claim 10: MNLI benchmark",       True,  True,  True,  True,  False, False),
    ]

    calc = EvaluationMetrics()
    ground_truth = [o.ground_truth_label for o in demo_outcomes]
    predictions  = [o.predicted_label    for o in demo_outcomes]
    return calc.compute(ground_truth, predictions, demo_outcomes)


# ------------------------------------------------------------------
# Sample Results Table (printed to terminal)
# ------------------------------------------------------------------

SAMPLE_RESULTS_TABLE = """
+------------------------------------------+----------+----------+-------------+
| SAMPLE RESULTS TABLE                     |          |          |             |
| (Illustrative values; actual vary)       |  Demo    |  FEVER   | TruthfulQA  |
+------------------------------------------+----------+----------+-------------+
| Halluc. Detection Accuracy               |  0.8000  |  0.7600  |  0.7200     |
| Precision                                |  0.8333  |  0.8000  |  0.7500     |
| Recall                                   |  0.8333  |  0.8000  |  0.7500     |
| F1 Score                                 |  0.8333  |  0.8000  |  0.7500     |
| Correction Success Rate (CSR)            |  0.7500  |  0.6500  |  0.6000     |
| Claim Preservation Rate (CPR)            |  0.8571  |  0.8000  |  0.8333     |
| Unnecessary Modification Rate (UMR)      |  0.1429  |  0.2000  |  0.1667     |
| Final Response Accuracy (FRA)            |  0.8000  |  0.7200  |  0.7000     |
+------------------------------------------+----------+----------+-------------+
"""


# ------------------------------------------------------------------
# Entry Point
# ------------------------------------------------------------------

def main():
    args = parse_args()

    logger.remove()
    logger.add(sys.stdout, level="WARNING")

    os.makedirs("results", exist_ok=True)

    print(f"\n{Fore.MAGENTA}{'=' * 60}")
    print("  Hallucination Correction Framework — Metrics Calculator")
    print(f"{'=' * 60}{Style.RESET_ALL}")

    # Print sample results table always
    print(SAMPLE_RESULTS_TABLE)

    # Collect results to compare
    all_results = {}

    if args.compare:
        for path in args.compare:
            name = Path(path).stem
            try:
                mr = load_and_compute(path)
                all_results[name] = mr
                print(f"Loaded: {path}")
            except Exception as e:
                logger.error(f"Failed to load {path}: {e}")

    elif args.input:
        name = Path(args.input).stem
        try:
            mr = load_and_compute(args.input)
            all_results[name] = mr
        except Exception as e:
            logger.error(f"Failed to load {args.input}: {e}")
    else:
        # No input — run demo metrics
        print(f"{Fore.YELLOW}No input file specified. Running demo metrics...{Style.RESET_ALL}")
        mr = compute_demo_metrics()
        all_results["demo"] = mr

    if not all_results:
        print("No results to display.")
        return

    # Print metrics for each experiment
    calc = EvaluationMetrics()
    for name, mr in all_results.items():
        print(f"\n{Fore.CYAN}{'-' * 45}")
        print(f"  Experiment: {name}")
        print(f"{'-' * 45}{Style.RESET_ALL}")
        calc.print_report(mr)

    # Comparison table (if multiple experiments)
    if len(all_results) > 1:
        print_comparison_table(all_results)

    # LaTeX output
    if args.latex:
        print_latex_table(all_results)

    # CSV output
    if args.csv:
        save_to_csv(all_results, args.csv)

    # Chart
    if args.chart:
        show_chart(all_results)


if __name__ == "__main__":
    main()
