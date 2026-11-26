"""Build supervised fine-tuning data for grounded answering (QLoRA on Qwen2.5).

Task taught: the same job Gemini does in the RAG pipeline. Input = the RAG
system prompt + 5 numbered retrieved passages + a question. Output = JSON
{"found", "answer", "citations"} in the RAG_SCHEMA format, so the fine-tuned
model can be scored with exactly the same parser and judge as Gemini.

Leakage control. The 147 articles used by the evaluation set (eval/qa.jsonl,
including dropped items) are excluded: no training question is written from
them and none of their chunks appears in a training context. The model
should learn the behaviour (read, cite, abstain), not the eval facts.

Examples:
  answerable  a question written by Gemini from one chunk (2 per chunk, evidence
              checked verbatim, same rules as the eval set). Context = bge-m3
              top 5; if the source chunk was not retrieved it replaces a random
              slot, so every answerable example contains its evidence.
              Target: found=true, the short answer, citations = the slots
              that contain the evidence.
  abstain     the same question with every chunk of its source article and
              every chunk containing the evidence removed from the context
              (filled with the next retrieved chunks). Target: found=false.
              Caveat: another article could still contain the answer, so a
              few abstention targets may be wrong; this is a known label noise.

The eval prompts (eval/qa_final.jsonl + eval/unanswerable.jsonl) are written
with the same bge-m3 top-5 contexts used in eval_answers.py, so Gemini, base
Qwen and fine-tuned Qwen are compared on identical inputs.

Output (not committed, upload the folder to Kaggle as a dataset):
  data/sft/train.jsonl, data/sft/val.jsonl, data/sft/eval_prompts.jsonl
  data/sft/summary.json  (also copied to results/sft_data_summary.json)

Usage:  python -m travel_rag.make_sft_data [--chunks 360]
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter

from travel_rag.config import DATA_DIR, RESULTS_DIR, ROOT
from travel_rag.eval_retrieval import QA_PATH
from travel_rag.llm import DailyQuotaError, GeminiClient, LLMError
from travel_rag.make_eval_set import BANNED_IN_QUESTION, QUESTION_TYPES, SYSTEM, evidence_in_chunk, normalise
from travel_rag.rag import RAG_SYSTEM, build_prompt
from travel_rag.retriever import INDEX_ROOT, Retriever, SearchResult

SFT_DIR = DATA_DIR / "sft"
INDEX = "bge-m3_s200_o40"
K = 5
POOL = 20           # retrieved candidates before filtering
MIN_WORDS = 80
ABSTAIN_SHARE = 0.25
VAL_SHARE = 0.05
SEED = 21

GEN_SYSTEM = SYSTEM.replace("write ONE question", "write TWO different questions (about different facts)") \
    .replace("The question must", "Each question must")
GEN_SCHEMA = {
    "type": "OBJECT",
    "properties": {"items": {"type": "ARRAY", "items": {
        "type": "OBJECT",
        "properties": {"usable": {"type": "BOOLEAN"}, "question": {"type": "STRING"},
                       "answer": {"type": "STRING"}, "evidence": {"type": "STRING"}},
        "required": ["usable", "question", "answer", "evidence"],
        "propertyOrdering": ["usable", "question", "answer", "evidence"],
    }}},
    "required": ["items"],
}


def target(found: bool, answer: str = "", citations: list[int] | None = None) -> str:
    return json.dumps({"found": found, "answer": answer if found else "", "citations": citations or []},
                      ensure_ascii=False)


def contains(chunk: dict, evidence: str) -> bool:
    return normalise(evidence).strip(" .").lower() in normalise(chunk["text"]).lower()


def record(rid: str, question: str, ctx: list[dict], tgt: str | None, **meta) -> dict:
    results = [SearchResult(rank=i + 1, score=0.0, chunk=c) for i, c in enumerate(ctx)]
    return {"id": rid, "system": RAG_SYSTEM, "prompt": build_prompt(question, results),
            "target": tgt, "context_chunk_ids": [c["chunk_id"] for c in ctx], **meta}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--chunks", type=int, default=360, help="source chunks to write questions from")
    args = parser.parse_args()
    rng = random.Random(SEED)

    retriever = Retriever(INDEX_ROOT / INDEX)
    by_id = {c["chunk_id"]: c for c in retriever.chunks}
    eval_articles = {json.loads(l)["article_id"] for l in (ROOT / "eval" / "qa.jsonl").open(encoding="utf-8")}

    # one source chunk per article first, then a second round, for diversity
    pool = [c for c in retriever.chunks if c["article_id"] not in eval_articles and c["n_words"] >= MIN_WORDS]
    rng.shuffle(pool)
    first, rest, seen = [], [], set()
    for c in pool:
        (rest if c["article_id"] in seen else first).append(c)
        seen.add(c["article_id"])
    sources = (first + rest)[:args.chunks]

    llm = GeminiClient()
    qa, rejected = [], Counter()
    for i, chunk in enumerate(sources):
        types = f"{QUESTION_TYPES[i % 6]} and {QUESTION_TYPES[(i + 3) % 6]}"
        where = f"{chunk['title']} — {chunk['section']}" if chunk["section"] else chunk["title"]
        try:
            out = llm.generate_json(f"Preferred question types: {types}.\n\nPassage ({where}):\n{chunk['text']}",
                                    GEN_SYSTEM, GEN_SCHEMA)
        except DailyQuotaError:
            print("daily quota reached; rerun later, cached calls are reused")
            break
        except LLMError as exc:
            rejected["llm_error"] += 1
            print("LLM error:", exc)
            continue
        for it in out.get("items", [])[:2]:
            if not it.get("usable"):
                rejected["not_usable"] += 1
            elif not evidence_in_chunk(it["evidence"], chunk["text"]):
                rejected["evidence_not_verbatim"] += 1
            elif BANNED_IN_QUESTION.search(it["question"]):
                rejected["not_self_contained"] += 1
            else:
                qa.append({"question": it["question"].strip(), "answer": it["answer"].strip(),
                           "evidence": normalise(it["evidence"]), "source": chunk})

    retrieved = retriever.search_many([q["question"] for q in qa], k=POOL)
    rows = []
    for n, (q, res) in enumerate(zip(qa, retrieved)):
        src = q["source"]
        cands = [r.chunk for r in res if r.chunk["article_id"] not in eval_articles]
        if rng.random() < ABSTAIN_SHARE:
            ctx = [c for c in cands if c["article_id"] != src["article_id"] and not contains(c, q["evidence"])][:K]
            if len(ctx) == K:
                rows.append(record(f"t{n:04d}a", q["question"], ctx, target(False), kind="abstain",
                                   source_chunk_id=src["chunk_id"]))
                continue
        ctx = cands[:K]
        if not any(contains(c, q["evidence"]) for c in ctx):
            ctx[rng.randrange(K)] = by_id[src["chunk_id"]]
        cites = [i + 1 for i, c in enumerate(ctx) if contains(c, q["evidence"])]
        rows.append(record(f"t{n:04d}", q["question"], ctx, target(True, q["answer"], cites), kind="answerable",
                           source_chunk_id=src["chunk_id"], gold_retrieved=src["chunk_id"] in
                           [r.chunk["chunk_id"] for r in res[:K]]))

    # split by source article, so validation questions come from unseen articles
    arts = sorted({by_id[r["source_chunk_id"]]["article_id"] for r in rows})
    rng.shuffle(arts)
    val_arts = set(arts[:max(1, round(len(arts) * VAL_SHARE))])
    train = [r for r in rows if by_id[r["source_chunk_id"]]["article_id"] not in val_arts]
    val = [r for r in rows if by_id[r["source_chunk_id"]]["article_id"] in val_arts]

    # evaluation prompts: identical contexts to eval_answers.py (dense top 5, no filtering)
    items = [json.loads(l) for l in QA_PATH.open(encoding="utf-8")]
    unans = [json.loads(l) for l in (ROOT / "eval" / "unanswerable.jsonl").open(encoding="utf-8")]
    qs = items + unans
    ctxs = retriever.search_many([q["question"] for q in qs], k=K)
    evals = [record(q["qid"], q["question"], [r.chunk for r in ctx], None,
                    kind="unanswerable" if q in unans else "answerable") for q, ctx in zip(qs, ctxs)]

    SFT_DIR.mkdir(parents=True, exist_ok=True)
    for name, data in (("train", train), ("val", val), ("eval_prompts", evals)):
        with (SFT_DIR / f"{name}.jsonl").open("w", encoding="utf-8") as f:
            for r in data:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    summary = {
        "source_chunks_used": len(sources), "qa_kept": len(qa), "rejected": dict(rejected),
        "train": dict(Counter(r["kind"] for r in train)), "val": dict(Counter(r["kind"] for r in val)),
        "eval_prompts": dict(Counter(r["kind"] for r in evals)),
        "answerable_with_gold_inserted": sum(r["kind"] == "answerable" and not r["gold_retrieved"] for r in rows),
        "eval_articles_excluded": len(eval_articles), "generator": llm.model, "llm_stats": llm.stats,
    }
    (SFT_DIR / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    RESULTS_DIR.mkdir(exist_ok=True)
    (RESULTS_DIR / "sft_data_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
