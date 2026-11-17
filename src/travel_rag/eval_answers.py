"""Answer evaluation: RAG vs closed-book (no retrieval), judged by an LLM.

Conditions (same generator, gemini-flash-lite-latest, temperature 0):
  rag          top-5 chunks from bge-m3 / 200-word chunks, grounded prompt
  closed_book  question only

Judge: a different model (gemini-3-flash-preview) compares each answer with
the reference answer and its evidence and returns one label:
  correct    states the reference answer (wording may differ), no contradiction
  partial    some of the key information, or hedged / incomplete
  incorrect  a different or contradicting answer
  abstained  declines to answer
A different model is used to reduce self-preference bias. The judge is itself
validated against human labels (scripts/judge_agreement.py).

Also measured:
  * abstention on 20 unanswerable questions (string match, no judge needed)
  * citation hit: does a cited chunk contain the gold evidence?
  * accuracy split by whether retrieval put the gold chunk in the context

Subsets: all 141 items; the 44 human-reviewed items; and the reviewed items
whose source fact is also supported by an independent source ("fact-checked").

Usage:  python -m travel_rag.eval_answers
"""

from __future__ import annotations

import json
import random
from concurrent.futures import ThreadPoolExecutor

from travel_rag.config import RESULTS_DIR, ROOT
from travel_rag.eval_retrieval import QA_PATH, bootstrap_ci, gold_chunks
from collections import Counter

from travel_rag.llm import DailyQuotaError, GeminiClient, LLMError
from travel_rag.rag import NOT_FOUND, RAGPipeline, answer_closed_book
from travel_rag.retriever import INDEX_ROOT, Retriever

UNANSWERABLE_PATH = ROOT / "eval" / "unanswerable.jsonl"
INDEX = "bge-m3_s200_o40"
K = 5
# One judge for every item, so labels are comparable. Free-tier daily quotas
# ruled out stronger models: gemini-3-flash-preview and gemini-3.6-flash allow
# about 20 requests a day. gemini-3.5-flash-lite was avoided because the
# generator alias (gemini-flash-lite-latest) most likely points to it, which
# would make the model grade its own answers. gemini-3.1-flash-lite is a
# separate release. A small judge can be wrong, which is why it is checked
# against human labels.
JUDGE_MODELS = ["gemini-3.1-flash-lite"]
WORKERS = 2
SEED = 0

JUDGE_SYSTEM = """You grade answers of a Vietnamese question-answering system.
You get a question, a reference answer, the evidence sentence the reference \
comes from, and a candidate answer. Judge only whether the candidate conveys \
the reference answer; do not use your own knowledge about the topic.

Labels:
- correct: the candidate states the key information of the reference answer \
(different wording, extra detail or extra correct context are fine) and does \
not contradict it.
- partial: the candidate gives only part of the key information, or hedges \
between the right answer and others.
- incorrect: the candidate gives a different answer or contradicts the reference.
- abstained: the candidate declines (e.g. "Không có trong tài liệu", "Tôi không biết").

Return the label and a one-sentence reason in English."""

JUDGE_SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "label": {"type": "STRING", "enum": ["correct", "partial", "incorrect", "abstained"]},
        "reason": {"type": "STRING"},
    },
    "required": ["label", "reason"],
    "propertyOrdering": ["reason", "label"],
}

ABSTAIN_MARKERS = [NOT_FOUND.lower().rstrip("."), "tôi không biết"]


def is_abstention(answer: str) -> bool:
    a = answer.lower()
    return any(m in a for m in ABSTAIN_MARKERS)


class Judge:
    def __init__(self, models: list[str]):
        self.clients = [GeminiClient(model=m) for m in models]
        self.exhausted: set[str] = set()

    def __call__(self, item: dict, answer: str) -> dict:
        prompt = (f"Question: {item['question']}\nReference answer: {item['answer']}\n"
                  f"Evidence: {item['evidence']}\nCandidate answer: {answer}")
        for client in self.clients:
            if client.model in self.exhausted:
                continue
            try:
                out = client.generate_json(prompt, JUDGE_SYSTEM, JUDGE_SCHEMA)
                return {**out, "judge_model": client.model}
            except DailyQuotaError:
                self.exhausted.add(client.model)
            except LLMError as exc:  # keep the run going; failures are reported
                return {"label": "error", "reason": str(exc)[:200], "judge_model": client.model}
        return {"label": "error", "reason": "all judge models out of daily quota", "judge_model": None}


def score(labels: list[str]) -> dict:
    errors = labels.count("error")
    labels = [lab for lab in labels if lab != "error"]
    n = len(labels)
    out = {lab: round(labels.count(lab) / n, 3) for lab in ("correct", "partial", "incorrect", "abstained")}
    vals = [1.0 if lab == "correct" else 0.0 for lab in labels]
    lo, hi = bootstrap_ci(vals, random.Random(SEED))
    out["accuracy_ci95"] = [round(lo, 3), round(hi, 3)]
    out["n"] = n
    out["judge_errors_excluded"] = errors
    return out


def main() -> None:
    items = [json.loads(line) for line in QA_PATH.open(encoding="utf-8")]
    unanswerable = [json.loads(line) for line in UNANSWERABLE_PATH.open(encoding="utf-8")]
    gen = GeminiClient()
    judge = Judge(JUDGE_MODELS)
    retriever = Retriever(INDEX_ROOT / INDEX)
    rag = RAGPipeline(retriever, gen, k=K)
    by_article: dict[str, list[dict]] = {}
    for c in retriever.chunks:
        by_article.setdefault(c["article_id"], []).append(c)

    all_q = [it["question"] for it in items] + [u["question"] for u in unanswerable]
    contexts = retriever.search_many(all_q, k=K)

    def run_answerable(args):
        it, ctx = args
        gold, _ = gold_chunks(it, by_article.get(it["article_id"], []))
        ra = rag.answer(it["question"], results=ctx)
        cb = answer_closed_book(it["question"], gen)
        return {
            "qid": it["qid"], "category": it["category"], "reviewed": it["reviewed"],
            "fact_status": it["fact_status"],
            "gold_in_context": any(r.chunk["chunk_id"] in gold for r in ctx),
            "rag_answer": ra.answer, "rag_found": ra.found,
            "rag_citations": [c.chunk_id for c in ra.citations],
            "citation_hit": any(c.chunk_id in gold for c in ra.citations) if ra.citations else None,
            "invalid_citations": ra.invalid_citations,
            "closed_book_answer": cb,
            "rag_judge": judge(it, ra.answer),
            "closed_book_judge": judge(it, cb),
        }

    def run_unanswerable(args):
        u, ctx = args
        ra = rag.answer(u["question"], results=ctx)
        cb = answer_closed_book(u["question"], gen)
        return {"qid": u["qid"], "kind": u["kind"], "rag_answer": ra.answer,
                "rag_abstained": not ra.found, "closed_book_answer": cb,
                "closed_book_abstained": is_abstention(cb)}

    with ThreadPoolExecutor(WORKERS) as pool:
        ans = list(pool.map(run_answerable, zip(items, contexts[:len(items)])))
        una = list(pool.map(run_unanswerable, zip(unanswerable, contexts[len(items):])))

    subsets = {
        "all": ans,
        "reviewed": [a for a in ans if a["reviewed"]],
        "fact_checked": [a for a in ans if a["reviewed"] and a["fact_status"] == "supported"],
    }
    judge_used = Counter(r[f"{c}_judge"]["judge_model"] for r in ans for c in ("rag", "closed_book"))
    report = {"generator": gen.model, "judge_models_used": dict(judge_used), "retrieval": f"{INDEX}, k={K}",
              "answerable": {}, "unanswerable": {}}
    for name, rows in subsets.items():
        report["answerable"][name] = {
            "rag": score([r["rag_judge"]["label"] for r in rows]),
            "closed_book": score([r["closed_book_judge"]["label"] for r in rows]),
        }
    rows = ans
    pair = {"both_correct": 0, "rag_only": 0, "closed_book_only": 0, "neither": 0}
    for r in rows:
        a, b = r["rag_judge"]["label"] == "correct", r["closed_book_judge"]["label"] == "correct"
        pair["both_correct" if a and b else "rag_only" if a else "closed_book_only" if b else "neither"] += 1
    report["answerable"]["paired_all"] = pair
    hit = [r for r in rows if r["gold_in_context"]]
    miss = [r for r in rows if not r["gold_in_context"]]
    report["answerable"]["rag_by_retrieval"] = {
        "gold_in_context": score([r["rag_judge"]["label"] for r in hit]),
        "gold_not_in_context": score([r["rag_judge"]["label"] for r in miss]) if miss else None,
    }
    cited = [r for r in rows if r["rag_found"]]
    report["answerable"]["citations"] = {
        "answers_with_citations": sum(bool(r["rag_citations"]) for r in cited),
        "answers_found": len(cited),
        "citation_hit_rate": round(sum(bool(r["citation_hit"]) for r in cited) / len(cited), 3),
        "invalid_citation_numbers": sum(len(r["invalid_citations"]) for r in rows),
    }
    report["unanswerable"] = {
        "n": len(una),
        "rag_abstain_rate": round(sum(u["rag_abstained"] for u in una) / len(una), 3),
        "closed_book_abstain_rate": round(sum(u["closed_book_abstained"] for u in una) / len(una), 3),
        "rag_answered": [u["qid"] for u in una if not u["rag_abstained"]],
    }
    report["llm_stats"] = {"generator": gen.stats, **{c.model: c.stats for c in judge.clients}}

    RESULTS_DIR.mkdir(exist_ok=True)
    (RESULTS_DIR / "answer_eval.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    with (RESULTS_DIR / "answers_per_item.jsonl").open("w", encoding="utf-8") as f:
        for r in ans + una:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
