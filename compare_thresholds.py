"""
compare_thresholds.py
=====================
Compare Contradiction Classification Thresholds on Multiclaim Benchmark:
1. Current baseline (threshold = 0.40, weighted)
2. Threshold = 0.30 (weighted)
3. Hybrid rule (if contradiction_prob >= 0.80 and EQS >= 0.30: CONTRADICTED)

Reports:
- Precision
- Recall
- F1
- HDA (Hallucination Detection Accuracy)
- CPR (Claim Preservation Rate)
- UMR (Unnecessary Modification Rate)
- Downstream CSR_GT and FRA_GT
"""

import os
import sys
import json
import time
import pandas as pd
from typing import Dict, Any, List
from colorama import init, Fore, Style
from loguru import logger

from config import FrameworkConfig
from hallucination_pipeline import HallucinationCorrectionPipeline
from evaluation.dataset_loader import DatasetLoader
from evaluation.metrics import EvaluationMetrics

init(autoreset=True)


def run_configuration(
    name: str,
    config: FrameworkConfig,
    samples: list,
    output_filename: str,
) -> Dict[str, Any]:
    print(f"\n{Fore.CYAN}{'=' * 70}")
    print(f"  RUNNING CONFIGURATION: {name}")
    print(f"  Mode: {config.contradiction_mode} | Threshold: {config.contradiction_threshold} | Hybrid Prob: {config.hybrid_contradiction_min_prob} | Hybrid EQS: {config.hybrid_contradiction_min_eqs}")
    print(f"{'=' * 70}{Style.RESET_ALL}\n")

    pipeline = HallucinationCorrectionPipeline(config)
    start_time = time.time()
    eval_results = pipeline.evaluate(samples, verbose=False)
    elapsed = time.time() - start_time

    metrics_res = eval_results["metrics"]

    # Save detailed configuration results
    os.makedirs("results", exist_ok=True)
    out_path = os.path.join("results", output_filename)
    results_data = {
        "config_name": name,
        "contradiction_mode": config.contradiction_mode,
        "contradiction_threshold": config.contradiction_threshold,
        "hybrid_prob": config.hybrid_contradiction_min_prob,
        "hybrid_eqs": config.hybrid_contradiction_min_eqs,
        "num_samples": len(samples),
        "elapsed_seconds": elapsed,
        "metrics": metrics_res.to_dict(),
        "response_reports": eval_results.get("response_reports", []),
        "per_claim_results": eval_results.get("pipeline_outputs", []),
        "correction_outcomes": [
            {
                "claim": o.claim,
                "ground_truth": o.ground_truth_label,
                "predicted": o.predicted_label,
                "was_corrected": o.was_corrected,
                "post_correction_verified": o.post_correction_verified,
                "was_preserved": o.was_preserved,
                "gold_label": o.gold_label,
                "gold_correction": o.gold_correction,
                "is_ground_truth_correct": o.is_ground_truth_correct,
            }
            for o in eval_results.get("correction_outcomes", [])
        ],
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results_data, f, indent=2, default=str)

    print(f"\n{Fore.GREEN}Completed {name} in {elapsed:.1f}s. Saved to {out_path}{Style.RESET_ALL}")
    return {
        "name": name,
        "metrics": metrics_res,
        "elapsed": elapsed,
        "outcomes": eval_results.get("correction_outcomes", []),
    }


def main():
    # Configure logging
    logger.remove()
    logger.add(
        sys.stdout,
        level="WARNING",
        format="<green>{time:HH:mm:ss}</green> | <level>{level}</level> | {message}",
    )

    print(f"\n{Fore.MAGENTA}{'=' * 75}")
    print("  CONTRADICTION THRESHOLD COMPARISON EXPERIMENT ON MULTICLAIM BENCHMARK")
    print(f"{'=' * 75}{Style.RESET_ALL}")

    loader = DatasetLoader()
    samples = loader.load_multiclaim_benchmark(max_samples=20)
    print(f"Loaded {len(samples)} multi-claim response samples from benchmark.")

    configs = [
        {
            "name": "1. Current Baseline (thresh=0.40, weighted)",
            "config": FrameworkConfig(
                contradiction_threshold=0.40,
                contradiction_mode="weighted",
            ),
            "file": "baseline_multiclaim_results.json",
        },
        {
            "name": "2. Threshold = 0.30 (weighted)",
            "config": FrameworkConfig(
                contradiction_threshold=0.30,
                contradiction_mode="weighted",
            ),
            "file": "threshold030_multiclaim_results.json",
        },
        {
            "name": "3. Hybrid Rule (prob>=0.80 & EQS>=0.30)",
            "config": FrameworkConfig(
                contradiction_threshold=0.40,
                contradiction_mode="hybrid",
                hybrid_contradiction_min_prob=0.80,
                hybrid_contradiction_min_eqs=0.30,
            ),
            "file": "hybrid_multiclaim_results.json",
        },
    ]

    all_runs = []
    for cfg in configs:
        res = run_configuration(
            name=cfg["name"],
            config=cfg["config"],
            samples=samples,
            output_filename=cfg["file"],
        )
        all_runs.append(res)

    # Compile Comparison Table
    table_rows = []
    for run in all_runs:
        m = run["metrics"]
        table_rows.append({
            "Configuration": run["name"],
            "Precision": f"{m.precision:.4f}",
            "Recall": f"{m.recall:.4f}",
            "F1 Score": f"{m.f1_score:.4f}",
            "HDA (Accuracy)": f"{m.accuracy:.4f}",
            "CPR": f"{m.cpr_gt:.4f}",
            "UMR": f"{m.umr_gt:.4f}",
            "TP": m.true_positives,
            "FP": m.false_positives,
            "FN": m.false_negatives,
            "TN": m.true_negatives,
            "CSR_GT": f"{m.csr_gt:.4f}" if m.csr_gt is not None else "N/A",
            "FRA_GT": f"{m.fra_gt:.4f}" if m.fra_gt is not None else "N/A",
        })

    df = pd.DataFrame(table_rows)
    csv_path = os.path.join("results", "contradiction_thresholds_comparison.csv")
    df.to_csv(csv_path, index=False)

    json_path = os.path.join("results", "contradiction_thresholds_comparison.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(table_rows, f, indent=2)

    print(f"\n\n{Fore.YELLOW}{'=' * 95}")
    print("  FINAL COMPARISON: CONTRADICTION CLASSIFICATION THRESHOLDS")
    print(f"{'=' * 95}{Style.RESET_ALL}\n")
    print(df[["Configuration", "Precision", "Recall", "F1 Score", "HDA (Accuracy)", "CPR", "UMR"]].to_string(index=False))
    print(f"\n{Fore.YELLOW}{'=' * 95}{Style.RESET_ALL}\n")
    print(df[["Configuration", "TP", "FP", "FN", "TN", "CSR_GT", "FRA_GT"]].to_string(index=False))
    print(f"\n{Fore.GREEN}Comparison saved to {csv_path} and {json_path}{Style.RESET_ALL}\n")


if __name__ == "__main__":
    main()
