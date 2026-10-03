"""
run_experiment.py
==================
Example Experiment Script
--------------------------
Demonstrates the full Selective Evidence-Guided Hallucination
Correction Framework on a set of example LLM responses.

This script:
1. Initializes the pipeline with default configuration.
2. Runs on hand-crafted examples (no API key required for demo).
3. Computes and prints evaluation metrics.
4. Saves results to the 'results/' directory as CSV and JSON.

Usage:
    # Basic demo (no API key needed)
    python run_experiment.py

    # With OpenAI for better corrections
    OPENAI_API_KEY=sk-xxx python run_experiment.py

    # Run on FEVER dataset (requires HuggingFace access)
    python run_experiment.py --dataset fever --max-samples 20

    # Run on TruthfulQA
    python run_experiment.py --dataset truthfulqa --max-samples 30
"""

import os
import sys
import json
import argparse
import time
from pathlib import Path
from loguru import logger
from colorama import init, Fore, Style

# Initialize colorama for Windows terminal colors
init(autoreset=True)

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

# ── Add project root to path ──────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent))

from config import FrameworkConfig
from hallucination_pipeline import HallucinationCorrectionPipeline
from evaluation.metrics import EvaluationMetrics
from evaluation.dataset_loader import DatasetLoader


# ──────────────────────────────────────────────────────────────────
# Example LLM Responses with Known Hallucinations
# ──────────────────────────────────────────────────────────────────

EXAMPLE_RESPONSES = [
    {
        "query": "Tell me about Albert Einstein.",
        "llm_response": (
            "Albert Einstein was a German physicist born in 1879 in Ulm, Germany. "
            "He developed the theory of relativity, which is one of the two pillars "
            "of modern physics. Einstein won the Nobel Prize in Physics in 1925 "
            "for his discovery of the photoelectric effect. "
            "He moved to the United States in 1933 and worked at Harvard University "
            "until his death in 1955."
            # Hallucinations: Nobel Prize year should be 1921, not 1925;
            # he worked at Princeton (IAS), not Harvard.
        ),
        "notes": "Contains 2 hallucinations: wrong Nobel year, wrong institution."
    },
    {
        "query": "What is the speed of light?",
        "llm_response": (
            "The speed of light in a vacuum is approximately 300,000 kilometers "
            "per second, or about 186,000 miles per second. "
            "This speed is denoted by the letter 'c' and is a fundamental "
            "constant in physics. Nothing with mass can travel faster than light. "
            "Einstein's theory of special relativity established that the speed "
            "of light is constant for all observers."
        ),
        "notes": "Mostly factual — should have high CSS scores."
    },
    {
        "query": "Who was Marie Curie?",
        "llm_response": (
            "Marie Curie was a Polish-French physicist and chemist who lived from "
            "1867 to 1934. She was the first woman to win a Nobel Prize, and the "
            "only person to win Nobel Prizes in two different sciences. "
            "She discovered the elements polonium and radium. "
            "Curie was born in Warsaw, which was then part of the Russian Empire. "
            "She conducted her Nobel Prize-winning research at Oxford University."
            # Hallucination: She worked at the University of Paris (Sorbonne), not Oxford.
        ),
        "notes": "Contains 1 hallucination: wrong university."
    },
]


# ──────────────────────────────────────────────────────────────────
# Argument Parser
# ──────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="Run Hallucination Correction Framework experiments",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--dataset",
        choices=["demo", "fever", "truthfulqa", "custom", "multiclaim"],
        default="demo",
        help="Dataset to evaluate on (default: demo)",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=20,
        help="Max dataset samples to evaluate (default: 20)",
    )
    parser.add_argument(
        "--output-dir",
        default="results",
        help="Directory to save results (default: results/)",
    )
    parser.add_argument(
        "--correction-backend",
        choices=["openai", "local"],
        default="local",
        help="Correction backend: 'openai' or 'local' (default: local)",
    )
    parser.add_argument(
        "--css-supported",
        type=float,
        default=0.75,
        help="CSS threshold for SUPPORTED label (default: 0.75)",
    )
    parser.add_argument(
        "--css-insufficient",
        type=float,
        default=0.40,
        help="CSS threshold for INSUFFICIENT label (default: 0.40)",
    )
    parser.add_argument(
        "--no-verbose",
        action="store_true",
        help="Suppress detailed pipeline output",
    )
    return parser.parse_args()


# ──────────────────────────────────────────────────────────────────
# Main Experiment Runner
# ──────────────────────────────────────────────────────────────────

def run_demo(pipeline: HallucinationCorrectionPipeline, output_dir: str, verbose: bool):
    """
    Run the demo experiment on hand-crafted example responses.

    Args:
        pipeline:   Initialized HallucinationCorrectionPipeline.
        output_dir: Directory to save results.
        verbose:    If True, print detailed output.
    """
    print(f"\n{Fore.CYAN}{'=' * 60}")
    print(f"  DEMO MODE: Hand-crafted LLM Response Examples")
    print(f"{'=' * 60}{Style.RESET_ALL}\n")

    all_results = []

    for i, example in enumerate(EXAMPLE_RESPONSES, 1):
        print(f"\n{Fore.YELLOW}Example {i}/{len(EXAMPLE_RESPONSES)}")
        print(f"Query: {example['query']}")
        print(f"Notes: {example['notes']}{Style.RESET_ALL}")
        print(f"\nOriginal Response:\n{example['llm_response']}\n")

        # Run the pipeline
        result = pipeline.run(
            llm_response=example["llm_response"],
            query=example["query"],
            verbose=verbose,
        )

        # Display correction report
        if result.reconstructed_response:
            print(result.reconstructed_response.correction_report)

        print(f"\n{Fore.GREEN}Final Response:")
        print(result.final_response)
        print(Style.RESET_ALL)

        # Collect results for saving
        all_results.append({
            "example_id": i,
            "query": example["query"],
            "original_response": example["llm_response"],
            "notes": example["notes"],
            "extracted_claims": result.extracted_claims,
            "verification_summary": {
                vr.claim[:50]: vr.label.value
                for vr in result.verification_results
            },
            "final_response": result.final_response,
            "elapsed_seconds": result.pipeline_metadata.get("elapsed_seconds", 0),
        })

    # Save results
    save_results(all_results, output_dir, filename="demo_results.json")
    print(f"\n{Fore.GREEN}Demo complete. Results saved to {output_dir}/demo_results.json{Style.RESET_ALL}")


def run_dataset_evaluation(
    pipeline: HallucinationCorrectionPipeline,
    dataset_name: str,
    max_samples: int,
    output_dir: str,
    verbose: bool,
):
    """
    Run evaluation on a benchmark dataset.

    Args:
        pipeline:     Initialized pipeline.
        dataset_name: "fever" | "truthfulqa"
        max_samples:  Maximum samples to evaluate.
        output_dir:   Directory to save results.
        verbose:      If True, show progress.
    """
    print(f"\n{Fore.CYAN}{'=' * 60}")
    print(f"  DATASET EVALUATION: {dataset_name.upper()} ({max_samples} samples)")
    print(f"{'=' * 60}{Style.RESET_ALL}\n")

    # Load dataset
    loader = DatasetLoader()
    try:
        if dataset_name == "multiclaim":
            samples = loader.load_multiclaim_benchmark(max_samples=max_samples)
        else:
            samples = loader.load(dataset_name, max_samples=max_samples)
    except Exception as e:
        logger.warning(f"Dataset loading failed: {e}. Using fallback samples.")
        # Use offline fallback
        samples = loader._load_truthfulqa_fallback()
        samples = samples[:max_samples]

    print(f"Loaded {len(samples)} samples from {dataset_name}")

    # Run evaluation
    print(f"\nRunning pipeline on {len(samples)} samples...")
    eval_results = pipeline.evaluate(samples, verbose=verbose)

    # Print response-level report & summary metrics
    metrics_calc = EvaluationMetrics()
    if eval_results.get("response_reports"):
        metrics_calc.print_response_level_report(eval_results["response_reports"])
    metrics_calc.print_report(eval_results["metrics"])

    # Save results with explicit ground-truth vs verifier separation
    results_data = {
        "dataset": dataset_name,
        "num_samples": len(samples),
        "metrics": eval_results["metrics"].to_dict(),
        "response_reports": eval_results.get("response_reports", []),
        "per_claim_results": eval_results["pipeline_outputs"],
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

    filename = f"{dataset_name}_results.json"
    save_results(results_data, output_dir, filename=filename)

    # Save metrics table as CSV
    try:
        df = metrics_calc.to_dataframe(eval_results["metrics"])
        csv_path = os.path.join(output_dir, f"{dataset_name}_metrics.csv")
        df.to_csv(csv_path, index=False)
        print(f"\n{Fore.GREEN}Metrics saved to {csv_path}{Style.RESET_ALL}")
    except Exception as e:
        logger.warning(f"Could not save CSV: {e}")

    print(f"\n{Fore.GREEN}Evaluation complete! Results saved to {output_dir}/{filename}{Style.RESET_ALL}")
    return eval_results


def save_results(data: dict, output_dir: str, filename: str):
    """Save results dictionary to a JSON file."""
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, filename)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, default=str)
    logger.info(f"Results saved to {path}")


# ──────────────────────────────────────────────────────────────────
# Entry Point
# ──────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    # Configure logging
    logger.remove()
    logger.add(
        sys.stdout,
        level="INFO",
        format="<green>{time:HH:mm:ss}</green> | <level>{level}</level> | {message}",
    )
    os.makedirs("logs", exist_ok=True)
    logger.add("logs/experiment.log", level="DEBUG", rotation="10 MB")

    print(f"\n{Fore.MAGENTA}{'=' * 60}")
    print("  Selective Evidence-Guided Hallucination Correction")
    print("  Framework — Experiment Runner")
    print(f"{'=' * 60}{Style.RESET_ALL}\n")

    # Build configuration
    config = FrameworkConfig(
        correction_backend=args.correction_backend,
        css_supported=args.css_supported,
        css_insufficient=args.css_insufficient,
        openai_api_key=os.getenv("OPENAI_API_KEY"),
    )

    # Initialize pipeline
    print(f"Initializing pipeline (NLI model: {config.nli_model_name})...")
    pipeline = HallucinationCorrectionPipeline(config)

    verbose = not args.no_verbose

    # Run appropriate mode
    if args.dataset == "demo":
        run_demo(pipeline, args.output_dir, verbose)
    else:
        run_dataset_evaluation(
            pipeline,
            dataset_name=args.dataset,
            max_samples=args.max_samples,
            output_dir=args.output_dir,
            verbose=verbose,
        )


if __name__ == "__main__":
    main()
