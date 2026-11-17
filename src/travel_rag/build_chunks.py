"""Chunk the selected articles and write chunks + summary statistics.

Usage:
    python -m travel_rag.build_chunks                      # default 200/40
    python -m travel_rag.build_chunks --size 100 --overlap 20
"""

from __future__ import annotations

import argparse
import json
import statistics

from travel_rag.chunking import chunk_article
from travel_rag.config import ARTICLES_PATH, PROCESSED_DIR, RESULTS_DIR


def chunks_path(size: int, overlap: int):
    return PROCESSED_DIR / f"chunks_s{size}_o{overlap}.jsonl"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", type=int, default=200, help="max words per chunk")
    parser.add_argument("--overlap", type=int, default=40, help="max words shared with previous chunk")
    parser.add_argument("--min-words", type=int, default=20)
    args = parser.parse_args()

    articles = [json.loads(line) for line in ARTICLES_PATH.open(encoding="utf-8")]
    chunks = [c for a in articles for c in chunk_article(a, args.size, args.overlap, args.min_words)]

    out = chunks_path(args.size, args.overlap)
    with out.open("w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps(c.to_dict(), ensure_ascii=False) + "\n")

    lengths = [c.n_words for c in chunks]
    stats = {
        "chunk_size": args.size,
        "overlap": args.overlap,
        "min_words": args.min_words,
        "n_articles": len(articles),
        "n_chunks": len(chunks),
        "words_total": sum(lengths),
        "words_per_chunk_mean": round(statistics.mean(lengths), 1),
        "words_per_chunk_median": statistics.median(lengths),
        "words_per_chunk_min": min(lengths),
        "words_per_chunk_max": max(lengths),
        "chunks_per_article_mean": round(len(chunks) / len(articles), 1),
    }
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    stats_path = RESULTS_DIR / f"chunk_stats_s{args.size}_o{args.overlap}.json"
    stats_path.write_text(json.dumps(stats, indent=2) + "\n", encoding="utf-8")
    print(f"{len(chunks)} chunks -> {out}")
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
