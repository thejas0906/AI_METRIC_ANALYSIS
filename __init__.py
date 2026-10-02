"""
__init__.py  (project root)
============================
Selective Evidence-Guided Hallucination Correction Framework
-------------------------------------------------------------
Top-level package exposing the main pipeline and configuration.

Quick start:
    from hallucination_pipeline import HallucinationCorrectionPipeline
    from config import FrameworkConfig

    config = FrameworkConfig()
    pipeline = HallucinationCorrectionPipeline(config)
    result = pipeline.run("Einstein won the Nobel Prize in 1925.")
    print(result.final_response)
"""

__version__ = "1.0.0"
__author__  = "Hallucination Correction Research"

__all__ = [
    "HallucinationCorrectionPipeline",
    "PipelineResult",
    "FrameworkConfig",
]

# Lazy imports — heavy dependencies (spaCy, torch, transformers) are only
# loaded when actually used, not at package import time.
def __getattr__(name):
    if name == "HallucinationCorrectionPipeline":
        from hallucination_pipeline import HallucinationCorrectionPipeline
        return HallucinationCorrectionPipeline
    if name == "PipelineResult":
        from hallucination_pipeline import PipelineResult
        return PipelineResult
    if name == "FrameworkConfig":
        from config import FrameworkConfig
        return FrameworkConfig
    raise AttributeError(f"module 'AI_METRIC_ANALYSIS' has no attribute {name!r}")
