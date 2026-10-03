"""
dataset_loader.py
==================
Dataset Loading Module for Evaluation
---------------------------------------
Loads and preprocesses benchmark datasets for evaluating the
Hallucination Correction Framework.

Supported Datasets:
-------------------
1. TruthfulQA  — Questions designed to elicit hallucinations from LLMs.
                 Labels: correct/incorrect answer pairs.
                 Source: HuggingFace datasets ('truthful_qa', 'generation')

2. FEVER       — Fact verification dataset with SUPPORTS/REFUTES/NEI labels.
                 We map: SUPPORTS → not hallucinated, REFUTES → hallucinated.
                 Source: HuggingFace datasets ('fever', 'v1.0')

3. Custom CSV  — User-provided CSV with 'claim' and 'label' columns.
                 label: 1 = hallucinated, 0 = supported

Each loader returns a list of DatasetSample objects ready for
the evaluation pipeline.

Design Decisions:
-----------------
- We limit dataset size by default (MAX_SAMPLES=50) for fast
  laptop-friendly experimentation. Pass max_samples=None to load all.
- TruthfulQA doesn't have binary per-claim labels, so we use the
  'correct_answers' field to derive a ground-truth "supported" flag.
- FEVER is the cleanest dataset for binary claim verification.
"""

import csv
import os
from dataclasses import dataclass, field
from typing import List, Optional, Literal
from loguru import logger


# ──────────────────────────────────────────────────────────────────
# Data Structure
# ──────────────────────────────────────────────────────────────────

@dataclass
class DatasetSample:
    """
    A single evaluation sample.

    Attributes:
        claim:           The factual claim or question.
        ground_truth:    True if the claim is hallucinated / incorrect.
        context:         Original question or context (optional).
        source:          Dataset name (e.g., "fever", "truthfulqa").
        metadata:        Additional dataset-specific fields.
    """
    claim:        str
    ground_truth: bool           # True = hallucinated
    context:      str            = ""
    source:       str            = "unknown"
    metadata:     dict           = field(default_factory=dict)


@dataclass
class GoldClaimAnnotation:
    """
    Gold standard annotation for an atomic factual claim within a complete response.

    Attributes:
        claim_text:      The exact claim text or proposition.
        gold_label:      "SUPPORTED", "CONTRADICTED", or "UNVERIFIABLE".
        gold_correction: Optional expected correction text if contradicted.
    """
    claim_text:      str
    gold_label:      str   # "SUPPORTED" | "CONTRADICTED" | "UNVERIFIABLE"
    gold_correction: Optional[str] = None


@dataclass
class ResponseEvaluationSample:
    """
    A complete LLM-generated response for a given question containing multiple
    factual claims, paired with per-claim gold annotations.

    Attributes:
        question:       The prompt or question given to the LLM.
        response:       The full multi-sentence LLM response text.
        gold_claims:    List of GoldClaimAnnotation for claims in this response.
        source:         Dataset or benchmark source.
        metadata:       Additional dataset-specific fields.
    """
    question:    str
    response:    str
    gold_claims: List[GoldClaimAnnotation] = field(default_factory=list)
    source:      str                       = "benchmark"
    metadata:    dict                      = field(default_factory=dict)
    response_id: str                       = ""

    # Backward compatibility properties
    @property
    def claim(self) -> str:
        return self.response

    @property
    def context(self) -> str:
        return self.question

    @property
    def ground_truth(self) -> bool:
        return any(gc.gold_label == "CONTRADICTED" for gc in self.gold_claims)


# Alias for research specification
MultiClaimResponseSample = ResponseEvaluationSample


# ──────────────────────────────────────────────────────────────────
# Dataset Loader
# ──────────────────────────────────────────────────────────────────

class DatasetLoader:
    """
    Loads benchmark datasets for hallucination evaluation.

    Usage:
        loader = DatasetLoader()
        samples = loader.load("fever", max_samples=50)
        # → List[DatasetSample]
    """

    def load(
        self,
        dataset_name: str,
        max_samples: Optional[int] = 50,
        split: str = "validation",
        custom_path: Optional[str] = None,
    ) -> List[DatasetSample]:
        """
        Load a dataset by name.

        Args:
            dataset_name:  "fever" | "truthfulqa" | "custom"
            max_samples:   Maximum number of samples to load. None = all.
            split:         Dataset split: "train", "validation", "test"
            custom_path:   Path to CSV file for "custom" dataset.

        Returns:
            List of DatasetSample objects.
        """
        loaders = {
            "fever":      self._load_fever,
            "truthfulqa": self._load_truthfulqa,
            "custom":     self._load_custom,
            "multiclaim": lambda split, max_samples: self.load_multiclaim_benchmark(max_samples=max_samples),
        }

        loader_fn = loaders.get(dataset_name.lower())
        if loader_fn is None:
            raise ValueError(
                f"Unknown dataset: '{dataset_name}'. "
                f"Supported: {list(loaders.keys())}"
            )

        if dataset_name.lower() == "custom":
            samples = loader_fn(custom_path or "data/custom_claims.csv", max_samples)
        elif dataset_name.lower() == "multiclaim":
            samples = loader_fn(split, max_samples)
        else:
            samples = loader_fn(split, max_samples)

        logger.info(f"Loaded {len(samples)} samples from '{dataset_name}'.")
        return samples

    # ──────────────────────────────────────────────────────────────
    # FEVER Loader
    # ──────────────────────────────────────────────────────────────

    def _load_fever(self, split: str, max_samples: Optional[int]) -> List[DatasetSample]:
        """
        Load the FEVER dataset from HuggingFace.

        FEVER Label Mapping:
            "SUPPORTS"        → ground_truth = False  (claim is supported)
            "REFUTES"         → ground_truth = True   (claim is hallucinated)
            "NOT ENOUGH INFO" → ground_truth = False  (treat as insufficient, not hallucinated)

        Args:
            split:       "train" | "labelled_dev" | "paper_dev"
            max_samples: Maximum number of samples.

        Returns:
            List of DatasetSample objects.
        """
        try:
            from datasets import load_dataset
        except ImportError:
            raise ImportError(
                "HuggingFace datasets library required. "
                "Install: pip install datasets"
            )

        logger.info(f"Loading FEVER dataset (split={split})...")

        # FEVER splits: 'train', 'labelled_dev', 'paper_dev', 'paper_test'
        fever_split = "labelled_dev" if split == "validation" else split

        try:
            dataset = load_dataset("fever", "v1.0", split=fever_split, trust_remote_code=True)
        except Exception as e:
            logger.warning(f"FEVER v1.0 failed ({e}), trying default config...")
            dataset = load_dataset("fever", split=fever_split, trust_remote_code=True)

        samples = []
        for item in dataset:
            label_str = item.get("label", "NOT ENOUGH INFO")

            # Map FEVER labels to binary hallucination flag
            if label_str == "SUPPORTS":
                is_hallucinated = False
            elif label_str == "REFUTES":
                is_hallucinated = True
            else:
                # "NOT ENOUGH INFO" → skip (ambiguous)
                continue

            samples.append(DatasetSample(
                claim=item["claim"],
                ground_truth=is_hallucinated,
                context="",
                source="fever",
                metadata={"fever_id": item.get("id", ""), "label": label_str},
            ))

            if max_samples and len(samples) >= max_samples:
                break

        return samples

    # ──────────────────────────────────────────────────────────────
    # TruthfulQA Loader
    # ──────────────────────────────────────────────────────────────

    def _load_truthfulqa(self, split: str, max_samples: Optional[int]) -> List[DatasetSample]:
        """
        Load TruthfulQA from HuggingFace.

        TruthfulQA provides multiple correct and incorrect answer
        strings per question. We simulate claim verification by:
        - Treating 'best_answer' → not hallucinated (ground_truth=False)
        - Taking one 'incorrect_answers' entry → hallucinated (ground_truth=True)

        This doubles the dataset size (one true + one false sample per question).

        Args:
            split:       "validation" | "train"
            max_samples: Maximum number of samples (total).

        Returns:
            List of DatasetSample objects.
        """
        try:
            from datasets import load_dataset
        except ImportError:
            raise ImportError("Install: pip install datasets")

        logger.info(f"Loading TruthfulQA dataset (split={split})...")

        try:
            dataset = load_dataset("truthful_qa", "generation", split=split, trust_remote_code=True)
        except Exception as e:
            logger.error(f"Failed to load TruthfulQA: {e}")
            return self._load_truthfulqa_fallback()

        samples = []
        for item in dataset:
            question = item.get("question", "")
            best_answer = item.get("best_answer", "")
            incorrect_answers = item.get("incorrect_answers", [])

            # True (supported) claim
            if best_answer:
                samples.append(DatasetSample(
                    claim=best_answer,
                    ground_truth=False,   # supported
                    context=question,
                    source="truthfulqa",
                    metadata={"question": question, "type": "correct"},
                ))

            # False (hallucinated) claim — take first incorrect answer
            if incorrect_answers:
                samples.append(DatasetSample(
                    claim=incorrect_answers[0],
                    ground_truth=True,    # hallucinated
                    context=question,
                    source="truthfulqa",
                    metadata={"question": question, "type": "incorrect"},
                ))

            if max_samples and len(samples) >= max_samples:
                break

        return samples

    @staticmethod
    def _load_truthfulqa_fallback() -> List[DatasetSample]:
        """
        Hand-crafted fallback TruthfulQA-style samples for offline use.
        Used when HuggingFace access is unavailable.
        """
        samples_data = [
            # (claim, ground_truth, question)
            ("The Great Wall of China is visible from space with the naked eye.",
             True, "Can you see the Great Wall of China from space?"),
            ("The Great Wall of China is not clearly visible from low Earth orbit without aid.",
             False, "Can you see the Great Wall of China from space?"),
            ("Napoleon Bonaparte was very short for his era, standing about 5 feet 2 inches.",
             True, "How tall was Napoleon?"),
            ("Napoleon Bonaparte was average height for his time, around 5 feet 7 inches.",
             False, "How tall was Napoleon?"),
            ("Albert Einstein failed mathematics as a child.",
             True, "Did Einstein fail math?"),
            ("Albert Einstein excelled at mathematics from an early age.",
             False, "Did Einstein fail math?"),
            ("Lightning never strikes the same place twice.",
             True, "Does lightning strike the same place twice?"),
            ("Lightning can and does strike the same place multiple times.",
             False, "Does lightning strike the same place twice?"),
            ("Humans only use 10% of their brains.",
             True, "What percentage of the brain do humans use?"),
            ("Humans use virtually all of their brain; the 10% myth is a misconception.",
             False, "What percentage of the brain do humans use?"),
        ]

        return [
            DatasetSample(
                claim=claim,
                ground_truth=gt,
                context=question,
                source="truthfulqa_fallback",
            )
            for claim, gt, question in samples_data
        ]

    # ──────────────────────────────────────────────────────────────
    # Custom CSV Loader
    # ──────────────────────────────────────────────────────────────

    def _load_custom(self, csv_path: str, max_samples: Optional[int]) -> List[DatasetSample]:
        """
        Load a custom CSV dataset.

        Expected CSV format:
            claim, label, context (optional)

        Where:
            label = 1 → hallucinated (ground_truth = True)
            label = 0 → supported   (ground_truth = False)

        Args:
            csv_path:    Path to CSV file.
            max_samples: Maximum samples to load.

        Returns:
            List of DatasetSample objects.
        """
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"Custom dataset CSV not found: {csv_path}")

        samples = []
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                claim   = row.get("claim", "").strip()
                label   = row.get("label", "0").strip()
                context = row.get("context", "").strip()

                if not claim:
                    continue

                samples.append(DatasetSample(
                    claim=claim,
                    ground_truth=(label in ("1", "true", "True", "hallucinated")),
                    context=context,
                    source="custom",
                ))

                if max_samples and len(samples) >= max_samples:
                    break

        return samples

    def load_multiclaim_benchmark(
        self,
        max_samples: Optional[int] = None,
        json_path: Optional[str] = None,
    ) -> List[ResponseEvaluationSample]:
        """
        Load multi-claim benchmark responses with ground-truth per-claim annotations.

        Each sample represents a full LLM response to a question with multiple
        claims that are independently annotated as SUPPORTED, CONTRADICTED, or UNVERIFIABLE.

        Args:
            max_samples: Maximum number of multi-claim samples to return.
            json_path:   Optional custom path to multiclaim benchmark JSON.

        Returns:
            List of ResponseEvaluationSample objects.
        """
        # Attempt to load from JSON benchmark file
        if json_path is None:
            candidate = os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                "data",
                "multiclaim_benchmark.json",
            )
            if os.path.exists(candidate):
                json_path = candidate

        if json_path and os.path.exists(json_path):
            import json
            logger.info(f"Loading multi-claim benchmark from JSON: {json_path}")
            with open(json_path, "r", encoding="utf-8") as f:
                raw_data = json.load(f)

            benchmark = []
            for item in raw_data:
                gold_claims = [
                    GoldClaimAnnotation(
                        claim_text=gc["claim_text"],
                        gold_label=gc["gold_label"],
                        gold_correction=gc.get("gold_correction"),
                    )
                    for gc in item.get("gold_claims", [])
                ]
                metadata = item.get("metadata", {})
                if "domain" in item:
                    metadata["domain"] = item["domain"]
                resp_id = item.get("response_id", "")
                if resp_id:
                    metadata["response_id"] = resp_id

                benchmark.append(
                    ResponseEvaluationSample(
                        question=item.get("question", ""),
                        response=item.get("response", ""),
                        gold_claims=gold_claims,
                        source=item.get("domain", "multiclaim_benchmark"),
                        metadata=metadata,
                        response_id=resp_id,
                    )
                )

            if max_samples:
                benchmark = benchmark[:max_samples]
            logger.info(f"Loaded {len(benchmark)} multi-claim response samples.")
            return benchmark

        # Fallback to hardcoded examples if JSON not found
        benchmark = [
            ResponseEvaluationSample(
                question="Tell me about Albert Einstein.",
                response=(
                    "Albert Einstein was born in 1879 in Germany. "
                    "He worked as a patent clerk in Switzerland. "
                    "He received the Nobel Prize in Literature in 1921. "
                    "He spent his later career at the Institute for Advanced Study in Princeton."
                ),
                gold_claims=[
                    GoldClaimAnnotation(
                        claim_text="Albert Einstein was born in 1879 in Germany.",
                        gold_label="SUPPORTED",
                    ),
                    GoldClaimAnnotation(
                        claim_text="He worked as a patent clerk in Switzerland.",
                        gold_label="SUPPORTED",
                    ),
                    GoldClaimAnnotation(
                        claim_text="He received the Nobel Prize in Literature in 1921.",
                        gold_label="CONTRADICTED",
                        gold_correction="He received the Nobel Prize in Physics in 1921.",
                    ),
                    GoldClaimAnnotation(
                        claim_text="He spent his later career at the Institute for Advanced Study in Princeton.",
                        gold_label="SUPPORTED",
                    ),
                ],
                source="multiclaim_benchmark",
                metadata={"topic": "biography"},
            ),
            ResponseEvaluationSample(
                question="What were the achievements of Marie Curie?",
                response=(
                    "Marie Curie was born in Warsaw, Poland. "
                    "She won two Nobel Prizes in Physics and Chemistry. "
                    "She was the first female professor at Harvard University. "
                    "Her personal laboratory notebooks remain radioactive today."
                ),
                gold_claims=[
                    GoldClaimAnnotation(
                        claim_text="Marie Curie was born in Warsaw, Poland.",
                        gold_label="SUPPORTED",
                    ),
                    GoldClaimAnnotation(
                        claim_text="She won two Nobel Prizes in Physics and Chemistry.",
                        gold_label="SUPPORTED",
                    ),
                    GoldClaimAnnotation(
                        claim_text="She was the first female professor at Harvard University.",
                        gold_label="CONTRADICTED",
                        gold_correction="She was the first female professor at the University of Paris (Sorbonne).",
                    ),
                    GoldClaimAnnotation(
                        claim_text="Her personal laboratory notebooks remain radioactive today.",
                        gold_label="SUPPORTED",
                    ),
                ],
                source="multiclaim_benchmark",
                metadata={"topic": "science"},
            ),
            ResponseEvaluationSample(
                question="Tell me about the Apollo 11 mission.",
                response=(
                    "Apollo 11 landed humans on the Moon in July 1969. "
                    "Neil Armstrong was the commander of the mission. "
                    "Apollo 11 was launched by the Soviet Union space program. "
                    "The astronauts returned safely to Earth."
                ),
                gold_claims=[
                    GoldClaimAnnotation(
                        claim_text="Apollo 11 landed humans on the Moon in July 1969.",
                        gold_label="SUPPORTED",
                    ),
                    GoldClaimAnnotation(
                        claim_text="Neil Armstrong was the commander of the mission.",
                        gold_label="SUPPORTED",
                    ),
                    GoldClaimAnnotation(
                        claim_text="Apollo 11 was launched by the Soviet Union space program.",
                        gold_label="CONTRADICTED",
                        gold_correction="Apollo 11 was launched by the United States NASA space program.",
                    ),
                    GoldClaimAnnotation(
                        claim_text="The astronauts returned safely to Earth.",
                        gold_label="SUPPORTED",
                    ),
                ],
                source="multiclaim_benchmark",
                metadata={"topic": "space"},
            ),
        ]

        if max_samples:
            return benchmark[:max_samples]
        return benchmark
