# conftest.py
# =============================================================
# pytest configuration — prevents pytest from treating the
# project root as a package (which would trigger __init__.py
# and pull in heavy ML dependencies before tests run).
# =============================================================
import sys
import os

# Ensure the project root is on sys.path for test imports
sys.path.insert(0, os.path.dirname(__file__))

collect_ignore = ["__init__.py", "hallucination_pipeline.py"]
