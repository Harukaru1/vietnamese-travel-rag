"""Shared paths and constants."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"            # Wikipedia parquet shards (not committed)
PROCESSED_DIR = DATA_DIR / "processed"  # selected articles and chunks (not committed)
RESULTS_DIR = ROOT / "results"         # small aggregated outputs (committed)

HF_DATASET = "wikimedia/wikipedia"
HF_CONFIG = "20231101.vi"
N_SHARDS = 4

ARTICLES_PATH = PROCESSED_DIR / "articles.jsonl"
