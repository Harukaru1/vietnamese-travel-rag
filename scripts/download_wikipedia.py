"""Download the Vietnamese Wikipedia dump (2023-11-01) from the Hugging Face Hub.

Source: https://huggingface.co/datasets/wikimedia/wikipedia (config 20231101.vi).
Text is licensed CC BY-SA 4.0; attribution is kept per article via its URL.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from huggingface_hub import hf_hub_download  # noqa: E402

from travel_rag.config import HF_CONFIG, HF_DATASET, N_SHARDS, RAW_DIR  # noqa: E402


def main() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    for i in range(N_SHARDS):
        filename = f"{HF_CONFIG}/train-{i:05d}-of-{N_SHARDS:05d}.parquet"
        path = hf_hub_download(HF_DATASET, filename, repo_type="dataset", local_dir=RAW_DIR)
        print("downloaded", path)


if __name__ == "__main__":
    main()
