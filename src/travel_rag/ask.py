"""Ask a question from the command line.

    python -m travel_rag.ask "Phở có nguồn gốc từ đâu?" [--k 5] [--closed-book]
"""

from __future__ import annotations

import argparse

from travel_rag.embeddings import model_slug
from travel_rag.llm import DEFAULT_MODEL, GeminiClient
from travel_rag.rag import RAGPipeline, answer_closed_book
from travel_rag.retriever import Retriever, index_dir


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("questions", nargs="+")
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--model", default="BAAI/bge-m3", help="embedding model")
    parser.add_argument("--llm", default=DEFAULT_MODEL)
    parser.add_argument("--size", type=int, default=200)
    parser.add_argument("--overlap", type=int, default=40)
    parser.add_argument("--closed-book", action="store_true", help="also print the no-RAG answer")
    args = parser.parse_args()

    llm = GeminiClient(model=args.llm)
    rag = RAGPipeline(Retriever(index_dir(model_slug(args.model), args.size, args.overlap)), llm, args.k)
    for q in args.questions:
        a = rag.answer(q)
        print(f"\nQ: {q}\nA: {a.answer}")
        for c in a.citations:
            where = f"{c.title} › {c.section}" if c.section else c.title
            print(f"   [{c.number}] {where} ({c.chunk_id}) {c.url}")
        if a.invalid_citations:
            print(f"   (dropped invalid citations: {a.invalid_citations})")
        if args.closed_book:
            print(f"No-RAG: {answer_closed_book(q, llm)}")
    print(f"\nLLM stats: {llm.stats}")


if __name__ == "__main__":
    main()
