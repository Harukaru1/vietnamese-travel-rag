"""Command-line search over a built index.

Usage:
    python -m travel_rag.search "Phở có nguồn gốc từ đâu?" --k 5
    python -m travel_rag.search "..." --model intfloat/multilingual-e5-small
"""

from __future__ import annotations

import argparse
import textwrap

from travel_rag.embeddings import model_slug
from travel_rag.retriever import Retriever, index_dir


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("query", nargs="+")
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--model", default="BAAI/bge-m3")
    parser.add_argument("--size", type=int, default=200)
    parser.add_argument("--overlap", type=int, default=40)
    args = parser.parse_args()

    retriever = Retriever(index_dir(model_slug(args.model), args.size, args.overlap))
    for query in args.query:
        print(f"\n### {query}")
        for r in retriever.search(query, args.k):
            c = r.chunk
            where = f"{c['title']} › {c['section']}" if c["section"] else c["title"]
            print(f"{r.rank}. [{r.score:.3f}] {where}  ({c['chunk_id']})")
            print(textwrap.indent(textwrap.shorten(c["text"], 220), "   "))


if __name__ == "__main__":
    main()
