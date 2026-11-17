"""Measure CPU embedding speed and token lengths for candidate models.

Embeds a fixed random sample of chunks with each model and extrapolates the
time to embed the whole chunk file. Writes results/embedding_benchmark.json.
"""

import json
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch  # noqa: E402

from travel_rag.build_chunks import chunks_path  # noqa: E402
from travel_rag.chunking import Chunk  # noqa: E402
from travel_rag.config import RESULTS_DIR  # noqa: E402
from travel_rag.embeddings import Embedder  # noqa: E402

MODELS = ["BAAI/bge-m3", "intfloat/multilingual-e5-base", "intfloat/multilingual-e5-small"]
SAMPLE = 128


def main() -> None:
    chunks = [Chunk(**json.loads(line)) for line in chunks_path(200, 40).open(encoding="utf-8")]
    random.seed(0)
    sample = [c.embedding_text() for c in random.sample(chunks, SAMPLE)]
    rows = []
    for name in MODELS:
        emb = Embedder(name)
        emb.embed_passages(sample[:8], show_progress=False)  # warm-up
        start = time.perf_counter()
        emb.embed_passages(sample, show_progress=False)
        sec = time.perf_counter() - start
        tokens = sorted(emb.count_tokens(t) for t in sample)
        rows.append({
            "model": name,
            "dim": emb.dim,
            "chunks_per_second": round(SAMPLE / sec, 2),
            "est_minutes_full_s200_o40": round(len(chunks) / (SAMPLE / sec) / 60, 1),
            "tokens_per_chunk_median": tokens[len(tokens) // 2],
            "tokens_per_chunk_max": tokens[-1],
            "words_per_chunk_median": sorted(len(t.split()) for t in sample)[SAMPLE // 2],
        })
        print(rows[-1], flush=True)
        del emb
    out = {"cpu_threads": torch.get_num_threads(), "sample_size": SAMPLE,
           "chunk_file": "chunks_s200_o40", "n_chunks": len(chunks), "results": rows}
    RESULTS_DIR.mkdir(exist_ok=True)
    (RESULTS_DIR / "embedding_benchmark.json").write_text(json.dumps(out, indent=2) + "\n")


if __name__ == "__main__":
    main()
