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
    elif "per_claim_results" in data:
        pipeline_outputs = data["per_claim_results"]
    elif "metrics" in data and "per_claim_results" not in data:
        # Already computed metrics — reconstruct MetricsResult
        logger.info("Found pre-computed metrics in JSON.")
        m = data["metrics"]
        return MetricsResult(
            accuracy=m.get("accuracy", 0),
            precision=m.get("precision", 0),
            recall=m.get("recall", 0),
            f1_score=m.get("f1_score", 0),
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
    Print a side-by-side comparison table of metrics from multiple experiments.

    Args:
        results: Dict mapping experiment name → MetricsResult.
    """
    try:
        from tabulate import tabulate
    except ImportError:
        print("Install tabulate for formatted tables: pip install tabulate")
        return

    metrics_names = ["Accuracy", "Precision", "Recall", "F1", "CSR", "CPR", "UMR", "FRA"]
    headers = ["Metric"] + list(results.keys())

    rows = []
    for i, metric_name in enumerate(metrics_names):
        row = [metric_name]
        for exp_name, mr in results.items():
            values = [
                mr.accuracy, mr.precision, mr.recall, mr.f1_score,
                mr.csr, mr.cpr, mr.umr, mr.fra,
            ]
            row.append(f"{values[i]:.4f}")
        rows.append(row)

    print(f"\n{Fore.CYAN}{'=' * 60}")
    print("  METRICS COMPARISON TABLE")
    print(f"{'=' * 60}{Style.RESET_ALL}")
    print(tabulate(rows, headers=headers, tablefmt="grid"))


def print_latex_table(results: dict) -> None:
    """
    Print a LaTeX table suitable for paper submission.

    Args:
        results: Dict mapping experiment name → MetricsResult.
    """
    exp_names = list(results.keys())
    col_header = " & ".join(exp_names)

    print("\n% LaTeX Table — Hallucination Correction Metrics")
    print("% Generated by compute_metrics.py")
    print("\\begin{table}[h]")
    print("\\centering")
    print(f"\\begin{{tabular}}{{l{'c' * len(exp_names)}}}")
    print("\\hline")
    print(f"\\textbf{{Metric}} & {col_header} \\\\")
    print("\\hline")

    metrics_data = [
        ("HDA (Accuracy)", "accuracy"),
        ("Precision", "precision"),
        ("Recall", "recall"),
        ("F1 Score", "f1_score"),
        ("Correction Success Rate (CSR)", "csr"),
        ("Claim Preservation Rate (CPR)", "cpr"),
        ("Unnecessary Modification Rate (UMR)", "umr"),
        ("Final Response Accuracy (FRA)", "fra"),
    ]

    for label, attr in metrics_data:
        values = " & ".join(
            f"{getattr(mr, attr):.4f}" for mr in results.values()
        )
        print(f"{label} & {values} \\\\")

    print("\\hline")
    print("\\end{tabular}")
    print(f"\\caption{{Evaluation results for the Selective Evidence-Guided Hallucination Correction Framework.}}")
    print("\\label{tab:metrics}")
    print("\\end{table}")


def save_to_csv(results: dict, csv_path: str) -> None:
    """
    Save comparison metrics to a CSV file.

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
            # Header
            writer.writerow(["Metric"] + list(results.keys()))
            metrics_data = [
                ("Accuracy",  "accuracy"),
                ("Precision", "precision"),
                ("Recall",    "recall"),
                ("F1 Score",  "f1_score"),
                ("CSR",       "csr"),
                ("CPR",       "cpr"),
                ("UMR",       "umr"),
                ("FRA",       "fra"),
            ]
            for label, attr in metrics_data:
                row = [label] + [f"{getattr(mr, attr):.4f}" for mr in results.values()]
                writer.writerow(row)
        print(f"Saved to {csv_path}")
        return

    rows = []
    metrics_data = [
        ("Accuracy",  "accuracy"),
        ("Precision", "precision"),
        ("Recall",    "recall"),
        ("F1 Score",  "f1_score"),
        ("CSR",       "csr"),
        ("CPR",       "cpr"),
        ("UMR",       "umr"),
        ("FRA",       "fra"),
    ]
    for label, attr in metrics_data:
        row = {"Metric": label}
        for exp_name, mr in results.items():
            row[exp_name] = round(getattr(mr, attr), 4)
        rows.append(row)

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
