"""BM25 lexical baseline with the same interface as Retriever.

Used to check how much of the dense retrievers' score comes from word overlap
between the generated questions and their source chunks.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from rank_bm25 import BM25Okapi

from travel_rag.chunking import Chunk
from travel_rag.retriever import SearchResult

TOKEN = re.compile(r"\w+", re.UNICODE)


def tokenize(text: str) -> list[str]:
    return TOKEN.findall(text.lower())


class BM25Retriever:
    def __init__(self, chunk_file: Path):
        with chunk_file.open(encoding="utf-8") as f:
            self.chunks = [json.loads(line) for line in f]
        self.bm25 = BM25Okapi([tokenize(Chunk(**c).embedding_text()) for c in self.chunks])

    def search_many(self, queries: list[str], k: int = 5) -> list[list[SearchResult]]:
        out = []
        for q in queries:
            scores = self.bm25.get_scores(tokenize(q))
            top = sorted(range(len(scores)), key=lambda i: -scores[i])[:k]
            out.append([SearchResult(rank=r + 1, score=float(scores[i]), chunk=self.chunks[i])
                        for r, i in enumerate(top)])
        return out

    def search(self, query: str, k: int = 5) -> list[SearchResult]:
        return self.search_many([query], k)[0]
