"""
Entry point to run the full Task 1 cleaning pipeline.
Run from the project root:
    python run_pipeline.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.cleaning import run_cleaning_pipeline

if __name__ == "__main__":
    run_cleaning_pipeline()
