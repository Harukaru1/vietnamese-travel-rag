"""Score Qwen2.5 base vs QLoRA predictions (from the Kaggle notebook) next to Gemini.

All three systems saw identical prompts: the same RAG system prompt and the
same bge-m3 top-5 passages (data/sft/eval_prompts.jsonl, built with the
contexts of eval_answers.py). They are graded by the same judge model and
prompt as eval_answers.py (gemini-3.1-flash-lite), and Gemini's labels are read
from results/answers_per_item.jsonl, so the numbers are comparable.

Extra metrics for small models:
  valid_json      output parses as the RAG JSON object (the fine-tune should fix format)
  false_abstain   answerable questions where the model said found=false
  citation_hit    a cited passage contains the gold evidence
Unparseable output is still judged: its raw text is the answer.

Usage:  python -m travel_rag.eval_finetuned --pred-dir results/finetune
        (expects predictions_base.jsonl and predictions_qlora.jsonl there)
"""

from __future__ import annotations

import argparse
import json
import random
from concurrent.futures import ThreadPoolExecutor

from travel_rag.build_chunks import chunks_path
from travel_rag.config import RESULTS_DIR, ROOT
from travel_rag.eval_answers import JUDGE_MODELS, Judge, is_abstention, score
from travel_rag.eval_retrieval import QA_PATH, bootstrap_ci, gold_chunks
from travel_rag.make_sft_data import SFT_DIR
from travel_rag.rag import NOT_FOUND

SEED = 0


def parse(output: str) -> tuple[dict | None, str, bool, list[int]]:
    """Return (parsed object or None, answer text, found, citations)."""
    try:
        obj = json.loads(output[output.index("{"):output.rindex("}") + 1])
        if not isinstance(obj, dict) or "found" not in obj:
            raise ValueError
    except ValueError:
        text = output.strip()
        return None, text, bool(text) and not is_abstention(text), []
    found = bool(obj.get("found")) and bool(str(obj.get("answer", "")).strip())
    cites = [c for c in obj.get("citations") or [] if isinstance(c, int)]
    return obj, str(obj.get("answer", "")).strip() if found else NOT_FOUND, found, cites


def paired(a: list[float], b: list[float]) -> dict:
    d = [x - y for x, y in zip(a, b)]
    lo, hi = bootstrap_ci(d, random.Random(SEED))
    return {"mean": round(sum(d) / len(d), 3), "ci95": [round(lo, 3), round(hi, 3)]}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred-dir", default=str(RESULTS_DIR / "finetune"))
    args = parser.parse_args()

    items = {json.loads(l)["qid"]: json.loads(l) for l in QA_PATH.open(encoding="utf-8")}
    prompts = {json.loads(l)["id"]: json.loads(l) for l in (SFT_DIR / "eval_prompts.jsonl").open(encoding="utf-8")}
    chunks = [json.loads(l) for l in chunks_path(200, 40).open(encoding="utf-8")]
    by_article: dict[str, list[dict]] = {}
    for c in chunks:
        by_article.setdefault(c["article_id"], []).append(c)
    gemini = {json.loads(l)["qid"]: json.loads(l) for l in (RESULTS_DIR / "answers_per_item.jsonl").open(encoding="utf-8")}
    judge = Judge(JUDGE_MODELS)

    systems = {}
    for name in ("base", "qlora"):
        with open(f"{args.pred_dir}/predictions_{name}.jsonl", encoding="utf-8") as f:
            preds = {json.loads(l)["id"]: json.loads(l)["output"] for l in f}
        assert set(preds) == set(prompts), f"{name}: predictions do not match eval prompts"

        def run(qid, preds=preds):
            obj, answer, found, cites = parse(preds[qid])
            ctx = prompts[qid]["context_chunk_ids"]
            row = {"qid": qid, "kind": prompts[qid]["kind"], "valid_json": obj is not None, "found": found,
                   "answer": answer, "citations": cites,
                   "invalid_citations": [c for c in cites if not 1 <= c <= len(ctx)]}
            if row["kind"] == "answerable":
                it = items[qid]
                gold, _ = gold_chunks(it, by_article.get(it["article_id"], []))
                row["citation_hit"] = (any(ctx[c - 1] in gold for c in cites if 1 <= c <= len(ctx))
                                       if found and cites else None)
                row["judge"] = judge(it, answer)
            return row

        with ThreadPoolExecutor(2) as pool:
            systems[name] = list(pool.map(run, sorted(prompts)))

    report = {"judge": JUDGE_MODELS[0], "n_prompts": len(prompts), "systems": {}}
    correct = {}
    for name, rows in systems.items():
        ans = [r for r in rows if r["kind"] == "answerable"]
        una = [r for r in rows if r["kind"] == "unanswerable"]
        found = [r for r in ans if r["found"]]
        correct[name] = {r["qid"]: float(r["judge"]["label"] == "correct") for r in ans}
        report["systems"][name] = {
            "answerable": score([r["judge"]["label"] for r in ans]),
            "valid_json_rate": round(sum(r["valid_json"] for r in rows) / len(rows), 3),
            "false_abstain_rate": round(sum(not r["found"] for r in ans) / len(ans), 3),
            "citation_hit_rate": round(sum(bool(r["citation_hit"]) for r in found) / len(found), 3) if found else None,
            "invalid_citation_numbers": sum(len(r["invalid_citations"]) for r in rows),
            "unanswerable_abstain_rate": round(sum(not r["found"] for r in una) / len(una), 3),
        }
    g_ans = [gemini[q] for q in sorted(items)]
    correct["gemini"] = {r["qid"]: float(r["rag_judge"]["label"] == "correct") for r in g_ans}
    report["systems"]["gemini (reference, eval_answers.py)"] = {
        "answerable": score([r["rag_judge"]["label"] for r in g_ans]),
        "false_abstain_rate": round(sum(not r["rag_found"] for r in g_ans) / len(g_ans), 3),
        "citation_hit_rate": round(sum(bool(r["citation_hit"]) for r in g_ans if r["rag_found"])
                                   / sum(r["rag_found"] for r in g_ans), 3),
        "unanswerable_abstain_rate": round(sum(gemini[q]["rag_abstained"] for q in gemini if q.startswith("u"))
                                           / sum(q.startswith("u") for q in gemini), 3),
    }
    qids = sorted(items)
    report["paired_accuracy_differences"] = {
        "qlora - base": paired([correct["qlora"][q] for q in qids], [correct["base"][q] for q in qids]),
        "qlora - gemini": paired([correct["qlora"][q] for q in qids], [correct["gemini"][q] for q in qids]),
    }
    report["llm_stats"] = {c.model: c.stats for c in judge.clients}
    (RESULTS_DIR / "finetune_eval.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    with (RESULTS_DIR / "finetune_per_item.jsonl").open("w", encoding="utf-8") as f:
        for name, rows in systems.items():
            for r in rows:
                f.write(json.dumps({"system": name, **r}, ensure_ascii=False) + "\n")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
