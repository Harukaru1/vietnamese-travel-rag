"""Hybrid retrieval (BM25 + dense, fused with RRF) and an optional reranker.

Why: BM25 and the dense retriever fail on different questions. BM25 is good at
rare exact words (names, numbers, dish names); dense search is good at
paraphrase. Fusing the two ranked lists can recover both kinds of hit.

Reciprocal Rank Fusion (Cormack et al., 2009):
    score(d) = sum over retrievers of 1 / (RRF_K + rank_of_d)
It uses ranks only, so the very different score scales of BM25 (unbounded)
and cosine similarity (-1..1) need no normalisation. RRF_K = 60 is the value
from the paper; it damps the advantage of the very top ranks.

Reranker: a cross-encoder reads (query, passage) together and scores
relevance, which is more accurate than comparing two separate embeddings but
far slower, so it is only applied to the top `rerank_depth` fused candidates.

Both classes expose search / search_many / chunks like Retriever.
"""

from __future__ import annotations

from travel_rag.bm25 import BM25Retriever
from travel_rag.chunking import Chunk
from travel_rag.retriever import Retriever, SearchResult

RRF_K = 60
DEPTH = 50  # candidates taken from each retriever before fusion


def rrf_fuse(rankings: list[list[str]], k: int = RRF_K) -> list[tuple[str, float]]:
    scores: dict[str, float] = {}
    for ranking in rankings:
        for rank, cid in enumerate(ranking, start=1):
            scores[cid] = scores.get(cid, 0.0) + 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda x: -x[1])


class HybridRetriever:
    def __init__(self, dense: Retriever, bm25: BM25Retriever, depth: int = DEPTH):
        dense_ids = [c["chunk_id"] for c in dense.chunks]
        assert dense_ids == [c["chunk_id"] for c in bm25.chunks], "dense and BM25 use different chunks"
        self.dense, self.bm25, self.depth = dense, bm25, depth
        self.chunks = dense.chunks
        self.by_id = {c["chunk_id"]: c for c in self.chunks}

    def search_many(self, queries: list[str], k: int = 5) -> list[list[SearchResult]]:
        dense = self.dense.search_many(queries, self.depth)
        lexical = self.bm25.search_many(queries, self.depth)
        out = []
        for d, b in zip(dense, lexical):
            fused = rrf_fuse([[r.chunk["chunk_id"] for r in d], [r.chunk["chunk_id"] for r in b]])[:k]
            out.append([SearchResult(rank=i + 1, score=s, chunk=self.by_id[cid])
                        for i, (cid, s) in enumerate(fused)])
        return out

    def search(self, query: str, k: int = 5) -> list[SearchResult]:
        return self.search_many([query], k)[0]


class RerankedRetriever:
    """Re-score the top `rerank_depth` candidates of a base retriever with a cross-encoder."""

    def __init__(self, base, model_name: str = "BAAI/bge-reranker-v2-m3", rerank_depth: int = 20,
                 max_length: int = 512, batch_size: int = 16):
        from sentence_transformers import CrossEncoder

        self.base, self.depth, self.batch_size = base, rerank_depth, batch_size
        self.chunks = base.chunks
        self.model_name = model_name
        self.model = CrossEncoder(model_name, max_length=max_length)

    def search_many(self, queries: list[str], k: int = 5) -> list[list[SearchResult]]:
        candidates = self.base.search_many(queries, max(k, self.depth))
        out = []
        for q, cands in zip(queries, candidates):
            pairs = [(q, Chunk(**r.chunk).embedding_text()) for r in cands]
            scores = self.model.predict(pairs, batch_size=self.batch_size, show_progress_bar=False)
            order = sorted(range(len(cands)), key=lambda i: -scores[i])[:k]
            out.append([SearchResult(rank=r + 1, score=float(scores[i]), chunk=cands[i].chunk)
                        for r, i in enumerate(order)])
        return out

    def search(self, query: str, k: int = 5) -> list[SearchResult]:
        return self.search_many([query], k)[0]
