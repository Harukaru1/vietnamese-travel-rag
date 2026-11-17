"""Does query rewriting fix retrieval for follow-up questions?

Test set (eval/conversations.jsonl), built from the reviewed QA items:
  turn 1  a question about the same place/dish/festival, another aspect
  turn 2  the QA question rephrased with the entity name replaced by a
          reference ("ở đó", "món này", ...); the entity name must not occur
The dialogues are written by gemini-3.1-flash-lite, a different model from the
rewriter, and checked in code (the follow-up must not contain the title).

For turn 2 we compare retrieval of the QA item's gold chunk (bge-m3, 200/40):
  raw        the follow-up as typed
  rewritten  the follow-up rewritten with the conversation (turn 1 is
             answered by the real RAG pipeline, so the history is realistic)
  oracle     the original standalone question (upper bound)

Usage: python -m travel_rag.eval_conversation
"""

from __future__ import annotations

import json
import random

from travel_rag.config import RESULTS_DIR, ROOT
from travel_rag.conversation import Turn, rewrite_question
from travel_rag.eval_retrieval import QA_PATH, bootstrap_ci, gold_chunks
from travel_rag.llm import GeminiClient
from travel_rag.rag import RAGPipeline
from travel_rag.retriever import INDEX_ROOT, Retriever

DIALOGUES_PATH = ROOT / "eval" / "conversations.jsonl"
N = 30
SEED = 5
K = 5
BUILDER_MODEL = "gemini-3.1-flash-lite"

BUILD_SYSTEM = """You create short two-turn test conversations in Vietnamese for a \
travel chatbot. You get a standalone question about an entity (a place, dish or \
festival) and the entity's name.

Write:
- "first_turn": a natural first question about the SAME entity, naming it, but \
asking about a different aspect than the standalone question.
- "follow_up": the standalone question rewritten as a natural follow-up that \
refers to the entity only indirectly ("ở đó", "nơi này", "món này", "lễ hội đó", \
"chỗ đó"...). It must not contain the entity's name, and it must ask for the same \
information as the standalone question."""

BUILD_SCHEMA = {
    "type": "OBJECT",
    "properties": {"first_turn": {"type": "STRING"}, "follow_up": {"type": "STRING"}},
    "required": ["first_turn", "follow_up"],
}


def build_dialogues(items: list[dict]) -> list[dict]:
    builder = GeminiClient(model=BUILDER_MODEL)
    rng = random.Random(SEED)
    reviewed = [it for it in items if it["reviewed"]]
    others = [it for it in items if not it["reviewed"]]
    rng.shuffle(reviewed)
    rng.shuffle(others)
    seen, out = set(), []
    for it in reviewed + others:  # human-reviewed items first
        if it["article_id"] in seen:
            continue
        o = builder.generate_json(
            f"Entity: {it['title']}\nStandalone question: {it['question']}", BUILD_SYSTEM, BUILD_SCHEMA)
        follow = o["follow_up"].strip()
        name = it["title"].split(" (")[0].lower()
        if name in follow.lower() or not o["first_turn"].strip():
            continue  # reference not removed: discard rather than repair
        seen.add(it["article_id"])
        out.append({"qid": it["qid"], "title": it["title"], "first_turn": o["first_turn"].strip(),
                    "follow_up": follow, "standalone": it["question"], "reviewed": it["reviewed"]})
        if len(out) == N:
            break
    with DIALOGUES_PATH.open("w", encoding="utf-8") as f:
        for d in out:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")
    return out


def main() -> None:
    items = {json.loads(l)["qid"]: json.loads(l) for l in QA_PATH.open(encoding="utf-8")}
    dialogues = ([json.loads(l) for l in DIALOGUES_PATH.open(encoding="utf-8")]
                 if DIALOGUES_PATH.exists() else build_dialogues(list(items.values())))
    retriever = Retriever(INDEX_ROOT / "bge-m3_s200_o40")
    pipeline = RAGPipeline(retriever, GeminiClient(), k=K)
    by_article: dict[str, list[dict]] = {}
    for c in retriever.chunks:
        by_article.setdefault(c["article_id"], []).append(c)

    rows = []
    for d in dialogues:
        it = items[d["qid"]]
        gold, _ = gold_chunks(it, by_article[it["article_id"]])
        first = pipeline.answer(d["first_turn"])
        rewritten = rewrite_question(d["follow_up"], [Turn(d["first_turn"], first.answer)], pipeline.llm)
        row = {"qid": d["qid"], "rewritten": rewritten,
               "entity_in_rewrite": d["title"].split(" (")[0].lower() in rewritten.lower()}
        for name, query in (("raw", d["follow_up"]), ("rewritten", rewritten), ("oracle", d["standalone"])):
            ids = [r.chunk["chunk_id"] for r in retriever.search(query, 10)]
            rank = next((i + 1 for i, c in enumerate(ids) if c in gold), None)
            row[f"{name}_recall@{K}"] = float(rank is not None and rank <= K)
            row[f"{name}_mrr@10"] = 1 / rank if rank else 0.0
        rows.append(row)

    report = {"n": len(rows), "index": "bge-m3_s200_o40",
              "entity_in_rewrite_rate": round(sum(r["entity_in_rewrite"] for r in rows) / len(rows), 3)}
    for name in ("raw", "rewritten", "oracle"):
        for m in (f"recall@{K}", "mrr@10"):
            vals = [r[f"{name}_{m}"] for r in rows]
            lo, hi = bootstrap_ci(vals, random.Random(SEED))
            report[f"{name}_{m}"] = {"mean": round(sum(vals) / len(vals), 3), "ci95": [round(lo, 3), round(hi, 3)]}
    diffs = [r[f"rewritten_recall@{K}"] - r[f"raw_recall@{K}"] for r in rows]
    lo, hi = bootstrap_ci(diffs, random.Random(SEED))
    report[f"rewritten_minus_raw_recall@{K}"] = {"mean": round(sum(diffs) / len(diffs), 3),
                                                 "ci95": [round(lo, 3), round(hi, 3)]}
    RESULTS_DIR.mkdir(exist_ok=True)
    (RESULTS_DIR / "conversation_eval.json").write_text(json.dumps(report, indent=2) + "\n")
    with (RESULTS_DIR / "conversation_per_item.jsonl").open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
