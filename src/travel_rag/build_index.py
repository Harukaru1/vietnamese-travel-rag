"""Embed a chunk file and save a FAISS index.

Usage:
    python -m travel_rag.build_index --size 200 --overlap 40 [--model BAAI/bge-m3]

Output directory index/<model>_s<size>_o<overlap>/ holds index.faiss,
embeddings.npy, chunks.jsonl (row i <-> vector i) and meta.json.
"""

from __future__ import annotations

import argparse
import json
import shutil
import time

import faiss
import numpy as np

from travel_rag.build_chunks import chunks_path
from travel_rag.chunking import Chunk
from travel_rag.embeddings import Embedder, model_slug
from travel_rag.retriever import build_faiss, index_dir


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", type=int, default=200)
    parser.add_argument("--overlap", type=int, default=40)
    parser.add_argument("--model", default="BAAI/bge-m3")
    parser.add_argument("--max-seq-length", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()

    src = chunks_path(args.size, args.overlap)
    chunks = [json.loads(line) for line in src.open(encoding="utf-8")]
    texts = [Chunk(**c).embedding_text() for c in chunks]

    embedder = Embedder(args.model, args.max_seq_length, args.batch_size)
    start = time.perf_counter()
    vectors = embedder.embed_passages(texts)
    seconds = time.perf_counter() - start

    out = index_dir(model_slug(args.model), args.size, args.overlap)
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / "embeddings.npy", vectors)
    faiss.write_index(build_faiss(vectors), str(out / "index.faiss"))
    shutil.copy(src, out / "chunks.jsonl")
    meta = {
        "model_name": args.model, "dim": int(vectors.shape[1]),
        "max_seq_length": args.max_seq_length, "chunk_size": args.size,
        "overlap": args.overlap, "n_chunks": len(chunks),
        "embed_seconds": round(seconds, 1),
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
