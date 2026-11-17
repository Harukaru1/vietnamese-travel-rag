"""FAISS index over chunk embeddings, plus a search function.

Index type: IndexFlatIP = exact (brute-force) inner-product search. With
~10k chunks x 1024 dims (~40 MB) a full scan takes a few milliseconds, so an
approximate index (HNSW, IVF) would add recall loss and tuning for no gain.
Approximate indexes matter at millions of vectors.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import faiss
import numpy as np

from travel_rag.config import ROOT
from travel_rag.embeddings import Embedder

INDEX_ROOT = ROOT / "index"  # not committed


def index_dir(model_slug: str, size: int, overlap: int) -> Path:
    return INDEX_ROOT / f"{model_slug}_s{size}_o{overlap}"


def build_faiss(vectors: np.ndarray) -> faiss.Index:
    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)
    return index


@dataclass
class SearchResult:
    rank: int
    score: float  # cosine similarity, in [-1, 1]
    chunk: dict


class Retriever:
    def __init__(self, directory: Path, embedder: Embedder | None = None):
        meta = json.loads((directory / "meta.json").read_text())
        self.meta = meta
        self.index = faiss.read_index(str(directory / "index.faiss"))
        with (directory / "chunks.jsonl").open(encoding="utf-8") as f:
            self.chunks = [json.loads(line) for line in f]
        assert self.index.ntotal == len(self.chunks), "index and chunk file out of sync"
        self.embedder = embedder or Embedder(meta["model_name"], max_seq_length=meta["max_seq_length"])

    def search(self, query: str, k: int = 5) -> list[SearchResult]:
        return self.search_many([query], k)[0]

    def search_many(self, queries: list[str], k: int = 5) -> list[list[SearchResult]]:
        q = self.embedder.embed_queries(queries)
        scores, ids = self.index.search(q, k)
        return [
            [SearchResult(rank=r + 1, score=float(s), chunk=self.chunks[i])
             for r, (s, i) in enumerate(zip(row_s, row_i)) if i != -1]
            for row_s, row_i in zip(scores, ids)
        ]
