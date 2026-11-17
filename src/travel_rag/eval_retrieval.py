"""Retrieval evaluation: recall@k and MRR for several chunkings and embedders.

Gold passages. Each question stores its evidence span (copied verbatim from a
200-word chunk). For any chunking, a chunk is relevant if it belongs to the
same article and contains the whole evidence span. With small chunks the span
can be split across two chunks; then every chunk of that article holding at
least MIN_OVERLAP of the evidence words counts as relevant. How many items
needed this fallback is reported.

Metrics (per question, then averaged):
  recall@k    1 if any relevant chunk is in the top k, else 0 (often called hit rate)
  MRR@10      1 / rank of the first relevant chunk, 0 if none in the top 10
  article@k   1 if any chunk of the right article is in the top k
95% confidence intervals come from a bootstrap over questions, because with
~140 questions a difference of a few points can be noise.

Usage:
    python -m travel_rag.eval_retrieval              # all available indexes
"""

from __future__ import annotations

import json
import random
from pathlib import Path

from travel_rag.bm25 import BM25Retriever
from travel_rag.build_chunks import chunks_path
from travel_rag.config import RESULTS_DIR, ROOT
from travel_rag.retriever import INDEX_ROOT, Retriever

QA_PATH = ROOT / "eval" / "qa_final.jsonl"
KS = (1, 3, 5, 10)
MIN_OVERLAP = 0.6
BOOTSTRAP = 2000
SEED = 0

CONFIGS = [  # (index directory, label)
    ("multilingual-e5-small_s100_o20", "e5-small, 100/20"),
    ("multilingual-e5-small_s200_o40", "e5-small, 200/40"),
    ("multilingual-e5-small_s400_o80", "e5-small, 400/80"),
    ("bge-m3_s200_o40", "bge-m3, 200/40"),
    ("bm25:200:40", "BM25, 200/40"),
]


def norm(text: str) -> str:
    return " ".join(text.split()).lower()


def gold_chunks(item: dict, article_chunks: list[dict]) -> tuple[set[str], str]:
    """Return (relevant chunk ids, how they were found: 'exact' / 'overlap' / 'none')."""
    ev = norm(item["evidence"]).strip(" .")
    exact = {c["chunk_id"] for c in article_chunks if ev in norm(c["text"])}
    if exact:
        return exact, "exact"
    ev_words = ev.split()
    partial = set()
    for c in article_chunks:
        words = set(norm(c["text"]).split())
        if sum(w in words for w in ev_words) / len(ev_words) >= MIN_OVERLAP:
            partial.add(c["chunk_id"])
    return partial, "overlap" if partial else "none"


def bootstrap_ci(values: list[float], rng: random.Random) -> tuple[float, float]:
    n = len(values)
    means = sorted(sum(values[rng.randrange(n)] for _ in range(n)) / n for _ in range(BOOTSTRAP))
    return means[int(0.025 * BOOTSTRAP)], means[int(0.975 * BOOTSTRAP) - 1]


def evaluate(retriever, items: list[dict]) -> list[dict]:
    by_article: dict[str, list[dict]] = {}
    for c in retriever.chunks:
        by_article.setdefault(c["article_id"], []).append(c)

    results = retriever.search_many([it["question"] for it in items], k=max(KS))
    per_q = []
    for it, res in zip(items, results):
        gold, how = gold_chunks(it, by_article.get(it["article_id"], []))
        ids = [r.chunk["chunk_id"] for r in res]
        first = next((i + 1 for i, cid in enumerate(ids) if cid in gold), None)
        art_first = next((i + 1 for i, r in enumerate(res) if r.chunk["article_id"] == it["article_id"]), None)
        row = {"qid": it["qid"], "gold_mapping": how, "n_gold": len(gold), "first_hit_rank": first,
               "mrr@10": 1 / first if first else 0.0}
        for k in KS:
            row[f"recall@{k}"] = float(first is not None and first <= k)
            row[f"article@{k}"] = float(art_first is not None and art_first <= k)
        per_q.append(row)
    return per_q


def summarise(per_q: list[dict], rng: random.Random) -> dict:
    out = {"n": len(per_q)}
    for key in [f"recall@{k}" for k in KS] + ["mrr@10", "article@5"]:
        vals = [r[key] for r in per_q]
        lo, hi = bootstrap_ci(vals, rng)
        out[key] = {"mean": round(sum(vals) / len(vals), 3), "ci95": [round(lo, 3), round(hi, 3)]}
    out["gold_mapping"] = {m: sum(r["gold_mapping"] == m for r in per_q) for m in ("exact", "overlap", "none")}
    return out


PAIRS = [  # (a, b): report mean(a - b) with a paired bootstrap CI
    ("bge-m3_s200_o40", "multilingual-e5-small_s200_o40"),
    ("bge-m3_s200_o40", "bm25:200:40"),
    ("multilingual-e5-small_s200_o40", "bm25:200:40"),
    ("multilingual-e5-small_s100_o20", "multilingual-e5-small_s400_o80"),
]


def paired_differences(per_question: dict) -> dict:
    """Resample questions jointly for both systems, so the CI reflects the
    difference on the same questions (tighter than comparing two separate CIs)."""
    out = {}
    for a, b in PAIRS:
        if a not in per_question or b not in per_question:
            continue
        ra = {r["qid"]: r for r in per_question[a]}
        rb = {r["qid"]: r for r in per_question[b]}
        qids = sorted(ra)
        for metric in ("mrr@10", "recall@5"):
            diffs = [ra[q][metric] - rb[q][metric] for q in qids]
            lo, hi = bootstrap_ci(diffs, random.Random(SEED))
            out[f"{a} - {b} | {metric}"] = {"mean": round(sum(diffs) / len(diffs), 3),
                                            "ci95": [round(lo, 3), round(hi, 3)]}
    return out


def main() -> None:
    items = [json.loads(line) for line in QA_PATH.open(encoding="utf-8")]
    subsets = {"all": items, "reviewed": [it for it in items if it["reviewed"]]}
    report = {"questions": {k: len(v) for k, v in subsets.items()}, "configs": {}}
    per_question = {}
    embedders = {}
    for dirname, label in CONFIGS:
        if dirname.startswith("bm25:"):
            _, size, overlap = dirname.split(":")
            retriever = BM25Retriever(chunks_path(int(size), int(overlap)))
            meta = {"n_chunks": len(retriever.chunks)}
        else:
            directory: Path = INDEX_ROOT / dirname
            if not (directory / "meta.json").exists():
                print(f"skip {label}: index not built")
                continue
            meta = json.loads((directory / "meta.json").read_text())
            retriever = Retriever(directory, embedder=embedders.get(meta["model_name"]))
            embedders[meta["model_name"]] = retriever.embedder  # reuse the loaded model
        per_q = evaluate(retriever, items)
        per_question[dirname] = per_q
        rng = random.Random(SEED)
        reviewed_ids = {it["qid"] for it in subsets["reviewed"]}
        report["configs"][label] = {
            "index": dirname, "n_chunks": meta["n_chunks"],
            "all": summarise(per_q, rng),
            "reviewed": summarise([r for r in per_q if r["qid"] in reviewed_ids], rng),
        }
        s = report["configs"][label]["all"]
        print(f"{label:18s} R@1 {s['recall@1']['mean']:.3f}  R@5 {s['recall@5']['mean']:.3f}  "
              f"R@10 {s['recall@10']['mean']:.3f}  MRR {s['mrr@10']['mean']:.3f}  gold {s['gold_mapping']}")

    report["paired_differences"] = paired_differences(per_question)

    RESULTS_DIR.mkdir(exist_ok=True)
    (RESULTS_DIR / "retrieval_eval.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    (RESULTS_DIR / "retrieval_per_question.json").write_text(json.dumps(per_question, indent=1) + "\n")


if __name__ == "__main__":
    main()
